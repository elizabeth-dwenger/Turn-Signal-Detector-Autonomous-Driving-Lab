"""
Stage 1 — Offline DINOv2 feature extraction.

For each sequence in the front/back CSVs:
  1. Load crop images from disk.
  2. Resize each crop to 224x224 (simple stretch — no letterboxing).
  3. Run through frozen DINOv2 ViT-S/14 to get 256 patch tokens * D=384.
  4. Spatially downsample: 256 -> P tokens via adaptive average pooling.
  5. Save feature tensor [T, P, 384] as a .pt file.
  6. Optionally save a horizontally-flipped copy with left/right label swapped.

All per-sequence metadata (per-frame labels, split group, file paths) is written
to a single metadata.json that later scripts (dataset.py, train.py) read.

Run this ONCE before any training.  DINOv2 is never used again after this step.

Usage:
    python -m signal_classifier.extract_features --config signal_classifier/config.yaml
"""

import argparse
import io
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Label helpers ─────────────────────────────────────────────────────────────

LABEL_MAP = {"none": 0, "left": 1, "right": 2, "hazard": 3, "both": 3}
FLIP_LABEL = {0: 0, 1: 2, 2: 1, 3: 3}   # left <-> right swap; none and hazard unchanged


def normalize_label(raw) -> Optional[int]:
    """
    Convert a raw CSV label string to an integer class index.
    "both" is treated as "hazard" (index 3).
    Returns None for unrecognised values.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().lower()
    return LABEL_MAP.get(s)


# ── DINOv2 helpers ────────────────────────────────────────────────────────────

def load_dino(model_name_or_path: str, device: str):
    """Load frozen DINOv2 and its image processor from HuggingFace or a local path."""
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        log.error("transformers is not installed.  Run: pip install transformers")
        sys.exit(1)

    log.info(f"Loading DINOv2 from: {model_name_or_path}")
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def extract_patch_tokens(
    images: List[Image.Image],
    processor,
    model,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Forward-pass a list of PIL images through DINOv2.

    Returns
    -------
    Tensor [N, 256, 384] - patch tokens (CLS token removed).
    """
    all_tokens: List[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        # last_hidden_state: [B, 257, D] - index 0 is the CLS token
        patch_tokens = model(**inputs).last_hidden_state[:, 1:, :]   # [B, 256, D]
        all_tokens.append(patch_tokens.cpu())
    return torch.cat(all_tokens, dim=0)   # [N, 256, D]


# ── Path resolution (mirrors existing ImageLoader logic) ─────────────────────

def resolve_crop_path(path_str: str, crop_base_dir: Path) -> Optional[Path]:
    """
    Try several strategies to locate a crop image, matching the behaviour of
    the existing ImageLoader._resolve_path in src/data/image_loader.py.
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    rel = crop_base_dir / p
    if rel.exists():
        return rel
    if p.exists():
        return p
    direct = crop_base_dir / p.name
    if direct.exists():
        return direct
    return None


def apply_photometric_augmentation(
    image: Image.Image,
    cfg: dict,
    rng: random.Random,
    strong: bool = False,
) -> Image.Image:
    """Apply mild PIL-based augmentation before DINO feature extraction."""
    out = image
    magnitude = 0.18 if strong else 0.10

    if cfg.get("augment_brightness_contrast", False):
        brightness = 1.0 + rng.uniform(-magnitude, magnitude)
        contrast = 1.0 + rng.uniform(-magnitude, magnitude)
        out = ImageEnhance.Brightness(out).enhance(brightness)
        out = ImageEnhance.Contrast(out).enhance(contrast)

    if cfg.get("augment_gamma", False):
        gamma = rng.uniform(0.85, 1.15) if strong else rng.uniform(0.92, 1.08)
        lut = [min(255, max(0, int(((i / 255.0) ** gamma) * 255.0))) for i in range(256)]
        out = out.point(lut * 3)

    if cfg.get("augment_blur", False) and rng.random() < (0.35 if strong else 0.20):
        out = out.filter(ImageFilter.GaussianBlur(radius=0.8 if strong else 0.5))

    if cfg.get("augment_noise", False):
        noise_scale = 6.0 if strong else 3.0
        tensor = torch.tensor(list(out.getdata()), dtype=torch.float32).view(out.size[1], out.size[0], 3)
        noise = torch.randn_like(tensor) * noise_scale
        tensor = (tensor + noise).clamp(0, 255).to(torch.uint8)
        out = Image.fromarray(tensor.numpy(), mode="RGB")

    if cfg.get("augment_jpeg", False) and rng.random() < 0.25:
        buffer = io.BytesIO()
        quality = rng.randint(55, 80) if strong else rng.randint(70, 88)
        out.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")

    return out


# ── Core per-sequence processing ─────────────────────────────────────────────

def process_one_sequence(
    group: pd.DataFrame,
    crop_base_dir: Path,
    label_column: str,
    output_path: Path,
    processor,
    model,
    device: str,
    batch_size: int,
    spatial_tokens: int,
    min_frames: int,
    do_flip: bool,
    flip_output_path: Optional[Path],
    augment_cfg: dict,
) -> Optional[Dict]:
    """
    Extract features for one sequence and (optionally) its horizontal mirror.

    Returns a dict of metadata entries suitable for metadata.json,
    or None if the sequence should be discarded.
    """
    group = group.sort_values("frame_id")

    if len(group) < min_frames:
        return None

    # Build per-frame label list
    per_frame_labels: List[int] = []
    for _, row in group.iterrows():
        lbl = normalize_label(row.get(label_column))
        per_frame_labels.append(lbl if lbl is not None else 0)   # default none

    label_counter = Counter(per_frame_labels)
    dominant_label = label_counter.most_common(1)[0][0] if label_counter else 0
    # Without global class stats at extraction time, treat active-signal sequences as the
    # minority-focused subset when minority_only_aug is enabled.
    use_stronger_aug = augment_cfg.get("minority_only_aug", False) and dominant_label in (1, 2, 3)
    should_augment = augment_cfg.get("augment_photometric", False) and (
        not augment_cfg.get("minority_only_aug", False) or dominant_label in (1, 2, 3)
    )
    seq_rng = random.Random(str(group["sequence_id"].iloc[0]))

    # Load and resize crops
    images: List[Image.Image] = []
    frame_ids: List[int] = []
    for _, row in group.iterrows():
        cp = resolve_crop_path(str(row["crop_path"]), crop_base_dir)
        if cp is None:
            log.warning(f"Crop not found: {row['crop_path']}")
            continue
        try:
            img = Image.open(cp).convert("RGB").resize((224, 224), Image.BILINEAR)
            if should_augment:
                img = apply_photometric_augmentation(img, augment_cfg, seq_rng, strong=use_stronger_aug)
            images.append(img)
            frame_ids.append(int(row["frame_id"]))
        except Exception as e:
            log.warning(f"Failed to load {cp}: {e}")

    if len(images) < min_frames:
        return None

    # Trim per_frame_labels to match successfully loaded images
    # (some frames may have been skipped due to missing crops)
    loaded_frame_ids = set(frame_ids)
    per_frame_labels_filtered = [
        normalize_label(row.get(label_column)) or 0
        for _, row in group.iterrows()
        if int(row["frame_id"]) in loaded_frame_ids
    ]

    # Extract features
    tokens = extract_patch_tokens(images, processor, model, device, batch_size)  # [T, 256, D]

    # Save original features (tensor only — safe for any torch.load settings)
    torch.save(tokens, output_path)

    seq_id = str(group["sequence_id"].iloc[0])
    recording = seq_id.split("/")[0]   # e.g. "2024-03-25-15-40-16_mapping_tartu"
    label_counts = dict(Counter(per_frame_labels_filtered))

    entry_orig = {
        "sequence_id":       seq_id,
        "feature_file":      str(output_path),
        "per_frame_labels":  per_frame_labels_filtered,
        "frame_ids":         frame_ids,
        "num_frames":        len(images),
        "is_flipped":        False,
        "recording":         recording,
        "label_counts":      label_counts,
    }
    result = {"original": entry_orig}

    # Offline horizontal-flip augmentation
    if do_flip and flip_output_path is not None:
        flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        flip_tokens = extract_patch_tokens(flipped_images, processor, model, device, batch_size)
        torch.save(flip_tokens, flip_output_path)

        flipped_labels = [FLIP_LABEL[l] for l in per_frame_labels_filtered]
        flip_label_counts = dict(Counter(flipped_labels))

        entry_flip = {
            "sequence_id":       seq_id,
            "feature_file":      str(flip_output_path),
            "per_frame_labels":  flipped_labels,
            "frame_ids":         frame_ids,
            "num_frames":        len(images),
            "is_flipped":        True,
            "recording":         recording,
            "label_counts":      flip_label_counts,
        }
        result["flipped"] = entry_flip

    return result


# ── CSV processing ────────────────────────────────────────────────────────────

def process_csv(
    csv_path: Path,
    camera_tag: str,
    crop_base_dir: Path,
    label_column: str,
    output_dir: Path,
    processor,
    model,
    device: str,
    batch_size: int,
    spatial_tokens: int,
    min_frames: int,
    augment_flip: bool,
    augment_cfg: dict,
) -> Dict:
    """
    Process one camera CSV and return all metadata entries.
    Keys in returned dict are globally unique: "{camera_tag}__{safe_seq_id}".
    """
    log.info(f"Loading [{camera_tag}] CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if "sequence_id" not in df.columns:
        log.error("CSV missing required 'sequence_id' column.")
        return {}
    if label_column not in df.columns:
        log.warning(f"Label column '{label_column}' not found; all labels will be 'none'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries: Dict = {}
    skipped = 0

    groups = list(df.groupby("sequence_id"))
    for seq_id, group in tqdm(groups, desc=f"  {camera_tag}", unit="seq"):
        safe = seq_id.replace("/", "__").replace(" ", "_")
        out_path  = output_dir / f"{safe}.pt"
        flip_path = output_dir / f"{safe}_hflip.pt" if augment_flip else None

        result = process_one_sequence(
            group=group,
            crop_base_dir=crop_base_dir,
            label_column=label_column,
            output_path=out_path,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
            spatial_tokens=spatial_tokens,
            min_frames=min_frames,
            do_flip=augment_flip,
            flip_output_path=flip_path,
            augment_cfg=augment_cfg,
        )

        if result is None:
            skipped += 1
            continue

        orig_key = f"{camera_tag}__{safe}"
        result["original"]["camera"] = camera_tag
        all_entries[orig_key] = result["original"]

        if "flipped" in result:
            flip_key = f"{camera_tag}__{safe}_hflip"
            result["flipped"]["camera"] = camera_tag
            all_entries[flip_key] = result["flipped"]

    log.info(
        f"  [{camera_tag}] done — {len(groups) - skipped} sequences saved "
        f"({skipped} discarded as too short or missing crops)."
    )
    return all_entries


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 features for SignalClassifier (Stage 1)."
    )
    parser.add_argument(
        "--config",
        default="signal_classifier/config.yaml",
        help="Path to config.yaml (default: signal_classifier/config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    fc = cfg["features"]

    device = fc.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    crop_base_dir  = Path(dc["crop_base_dir"])
    label_column   = dc.get("label_column", "predicted_label")
    min_frames     = dc.get("min_sequence_frames", 10)
    spatial_tokens = fc.get("spatial_tokens", 32)
    batch_size     = fc.get("batch_size", 64)
    augment_flip   = fc.get("augment_flip", True)
    output_root    = Path(fc["output_dir"])

    dino_source = fc.get("dino_model_path") or fc["dino_model_name"]
    processor, model = load_dino(dino_source, device)

    all_metadata: Dict = {}

    csv_sources = [
        ("front", dc.get("front_csv")),
        ("back",  dc.get("back_csv")),
    ]
    for camera_tag, csv_path in csv_sources:
        if not csv_path or not Path(csv_path).exists():
            log.warning(f"CSV not found for [{camera_tag}]: {csv_path!r} — skipping.")
            continue
        entries = process_csv(
            csv_path=Path(csv_path),
            camera_tag=camera_tag,
            crop_base_dir=crop_base_dir,
            label_column=label_column,
            output_dir=output_root / camera_tag,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
            spatial_tokens=spatial_tokens,
            min_frames=min_frames,
            augment_flip=augment_flip,
            augment_cfg=fc,
        )
        all_metadata.update(entries)

    meta_path = output_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    log.info(f"metadata.json written to {meta_path}")
    log.info(f"Total sequences in index: {len(all_metadata)}")

    # Print a quick label summary
    label_totals: Counter = Counter()
    for entry in all_metadata.values():
        label_totals.update(entry.get("label_counts", {}))
    log.info("Label distribution across all windows:")
    for lbl, cnt in sorted(label_totals.items()):
        log.info(f"  class {lbl}: {cnt} frames")


if __name__ == "__main__":
    main()
