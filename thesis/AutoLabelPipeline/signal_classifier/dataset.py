"""
SlidingWindowDataset — PyTorch Dataset for SignalClassifier training and inference.

Reads the metadata.json produced by extract_features.py and the corresponding
.pt feature tensors, then yields fixed-length temporal windows with class labels.

Key design rules
----------------
* Train / val / test split is performed at the RECORDING level (the timestamp+name
  prefix of sequence_id), so no physical scenario appears in multiple splits.
* Horizontally-flipped sequences (is_flipped=True) are included in the training
  split only.  Val and test use only unflipped originals for unbiased evaluation.
* Transition windows: where the ground-truth per-frame labels do not sufficiently
  agree within the window — are silently skipped during training.  The threshold
  is controlled by label_purity_threshold (default 0.80).
* Caching: tensors are loaded from disk on demand.  OS page cache handles repeated
  access efficiently; no in-process tensor cache to avoid multi-worker memory bloat.

Labels: 0=none, 1=left, 2=right, 3=hazard
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import torch
from torch.utils.data import Dataset


LABEL_MAP   = {"none": 0, "left": 1, "right": 2, "hazard": 3}
LABEL_NAMES = ["none", "left", "right", "hazard"]


class _Window:
    """struct describing one sliding-window training sample."""
    __slots__ = ("feature_file", "start", "label_int", "camera", "sequence_id")

    def __init__(
        self,
        feature_file: str,
        start: int,
        label_int: int,
        camera: str,
        sequence_id: str,
    ):
        self.feature_file = feature_file
        self.start        = start
        self.label_int    = label_int
        self.camera       = camera
        self.sequence_id  = sequence_id


class SlidingWindowDataset(Dataset):
    """
    Parameters
    ----------
    metadata_path         : path to metadata.json from extract_features.py
    split                 : "train", "val", or "test"
    window_size           : T; frames per window (default 10)
    stride                : step between windows (use train_stride or inference_stride from config)
    label_purity_threshold: fraction of frames in a window that must share the majority
                            label for the window to be kept (training only)
    train_ratio           : fraction of recordings allocated to training
    val_ratio             : fraction allocated to validation (remainder -> test)
    split_seed            : seed for reproducible recording-level shuffle
    """

    def __init__(
        self,
        metadata_path: str,
        split: str,
        window_size: int = 10,
        stride: int = 5,
        label_purity_threshold: float = 0.80,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        split_seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.window_size = window_size
        self.split       = split

        with open(metadata_path) as f:
            meta: Dict = json.load(f)

        # ── Recording-level split ──────────────────────────────────────────
        # Each recording is a unique (camera, recording) pair so that front
        # and back cameras of the same drive are split together.
        all_recordings = sorted(
            {(e["camera"], e["recording"]) for e in meta.values() if not e.get("is_flipped", False)}
        )
        rng = random.Random(split_seed)
        rng.shuffle(all_recordings)

        n          = len(all_recordings)
        n_train    = int(n * train_ratio)
        n_val      = int(n * val_ratio)

        if split == "train":
            chosen_recordings = set(all_recordings[:n_train])
        elif split == "val":
            chosen_recordings = set(all_recordings[n_train : n_train + n_val])
        else:
            chosen_recordings = set(all_recordings[n_train + n_val :])

        # ── Build window list ──────────────────────────────────────────────
        self.windows: List[_Window] = []
        is_training = split == "train"

        for key, entry in meta.items():
            camera    = entry.get("camera", "")
            recording = entry.get("recording", "")
            is_flipped = entry.get("is_flipped", False)

            # Flipped copies only participate in training
            if is_flipped and not is_training:
                continue

            if (camera, recording) not in chosen_recordings:
                continue

            per_frame_labels: List[int] = entry.get("per_frame_labels", [])
            num_frames: int             = entry.get("num_frames", len(per_frame_labels))
            feature_file: str           = entry["feature_file"]
            sequence_id: str            = entry.get("sequence_id", key)

            start = 0
            while start <= num_frames - window_size:
                window_labels = per_frame_labels[start : start + window_size]

                if not window_labels:
                    start += stride
                    continue

                # Advance pointer based on dynamic stride (oversampling active signals)
                # If there's an active signal (1, 2, 3), take a step of 1 to oversample (during train).
                has_signal = any(lbl in (1, 2, 3) for lbl in window_labels)
                current_stride = 1 if (has_signal and is_training) else stride

                majority_label = max(set(window_labels), key=window_labels.count)
                purity = window_labels.count(majority_label) / len(window_labels)

                # Skip transition windows during training
                if is_training and purity < label_purity_threshold:
                    start += current_stride
                    continue

                self.windows.append(
                    _Window(
                        feature_file=feature_file,
                        start=start,
                        label_int=majority_label,
                        camera=camera,
                        sequence_id=sequence_id,
                    )
                )
                start += current_stride

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """
        Returns
        -------
        features : FloatTensor  [T, P, D]
        label    : LongTensor   scalar
        """
        w = self.windows[idx]
        # Load tensor from disk on every access; OS page cache handles efficiency.
        full_features: torch.Tensor = torch.load(w.feature_file, weights_only=True)
        window = full_features[w.start : w.start + self.window_size].float()
        metadata = {
            "camera": w.camera,
            "feature_file": w.feature_file,
            "start": w.start,
            "label": w.label_int,
            "sequence_id": w.sequence_id,
        }
        return window, torch.tensor(w.label_int, dtype=torch.long), metadata

    # ── Utilities ──────────────────────────────────────────────────────────────

    def class_counts(self) -> List[int]:
        """Return number of windows per class in this split (order: none/left/right/hazard)."""
        counts = [0, 0, 0, 0]
        for w in self.windows:
            counts[w.label_int] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """
        Return inverse-frequency class weights as a FloatTensor [4].
        Useful for torch.nn.CrossEntropyLoss(weight=...).
        """
        counts = self.class_counts()
        total  = sum(counts)
        weights = [total / (4 * max(c, 1)) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


class HumanTestDataset(Dataset):
    """
    Dataset for per-frame evaluation on human-labeled sequences.
    Loads full sequences and returns per-frame labels (not sequence-level).
    
    Enables frame-level accuracy computation, which gracefully handles sequences
    with label transitions (e.g., none -> left -> none).

    Parameters
    ----------
    front_csv, back_csv  : paths to human-labeled CSVs
    label_column         : name of column holding human labels
    metadata_path        : path to metadata.json (to find .pt feature files)
    crop_base_dir        : base directory for crop paths
    min_frames           : discard sequences shorter than this
    """

    def __init__(
        self,
        front_csv: Optional[str],
        back_csv: Optional[str],
        label_column: str,
        metadata_path: str,
        crop_base_dir: str,
        min_frames: int = 10,
    ):
        self.sequences = []  # list of dicts with full per-frame info

        # Load metadata to find .pt files
        with open(metadata_path) as f:
            meta: Dict = json.load(f)
        file_to_entry = {e["feature_file"]: e for e in meta.values() if not e.get("is_flipped", False)}

        # Load human-labeled CSVs
        for csv_path, camera in [(front_csv, "front"), (back_csv, "back")]:
            if not csv_path or not Path(csv_path).exists():
                continue
            
            df = pd.read_csv(csv_path)
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in {csv_path}")

            # Group by sequence
            for seq_id, group in df.groupby("sequence_id"):
                if len(group) < min_frames:
                    continue

                # Get the feature file from metadata
                feature_file = None
                for ff, entry in file_to_entry.items():
                    if entry.get("sequence_id") == seq_id and entry.get("camera") == camera:
                        feature_file = ff
                        break

                if not feature_file:
                    continue

                # Parse per-frame labels
                per_frame_labels = []
                for _, row in group.iterrows():
                    lbl = normalize_label(row[label_column])
                    per_frame_labels.append(lbl if lbl is not None else 0)

                self.sequences.append({
                    "sequence_id":      seq_id,
                    "camera":           camera,
                    "feature_file":     feature_file,
                    "per_frame_labels": per_frame_labels,
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns
        -------
        features         : FloatTensor  [T, P, D]
        per_frame_labels : LongTensor   [T]  (one label per frame)
        sequence_id      : str
        """
        seq = self.sequences[idx]
        features = torch.load(seq["feature_file"], weights_only=True).float()
        per_frame_labels = torch.tensor(seq["per_frame_labels"], dtype=torch.long)
        return features, per_frame_labels, seq["sequence_id"]


def normalize_label(raw) -> Optional[int]:
    """Convert raw label string to int (0-3), or None if unrecognised."""
    if raw is None or (isinstance(raw, float) and __import__("pandas").isna(raw)):
        return None
    s = str(raw).strip().lower()
    label_map = {"none": 0, "left": 1, "right": 2, "hazard": 3, "both": 3}
    return label_map.get(s)


def build_datasets(
    metadata_path: str,
    window_cfg: dict,
    data_cfg: dict,
) -> Tuple["SlidingWindowDataset", "SlidingWindowDataset", "SlidingWindowDataset"]:
    """
    Convenience factory that constructs train, val, and test datasets from config dicts.

    Parameters
    ----------
    metadata_path : path to metadata.json
    window_cfg    : cfg["window"]  sub-dict
    data_cfg      : cfg["data"]    sub-dict

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    common = dict(
        metadata_path         = metadata_path,
        window_size           = window_cfg["size"],
        label_purity_threshold= window_cfg.get("label_purity_threshold", 0.80),
        train_ratio           = data_cfg.get("train_ratio", 0.70),
        val_ratio             = data_cfg.get("val_ratio",   0.15),
        split_seed            = data_cfg.get("split_seed",  42),
    )

    train_ds = SlidingWindowDataset(split="train", stride=window_cfg["train_stride"],     **common)
    val_ds   = SlidingWindowDataset(split="val",   stride=window_cfg["inference_stride"], **common)
    test_ds  = SlidingWindowDataset(split="test",  stride=window_cfg["inference_stride"], **common)

    return train_ds, val_ds, test_ds
