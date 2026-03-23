"""
Inference script for SignalClassifier.

Per-frame evaluation for all splits:
- Autolabeled splits (test/val/train): Slides windows, aggregates frame-level predictions, 
  evaluates against per-frame pseudo-labels
- Human test split: Slides windows, aggregates frame-level predictions, 
  evaluates against human-labeled ground truth frames

This approach gracefully handles sequences with label transitions and gives fine-grained
per-frame accuracy metrics.

Usage:
    python -m signal_classifier.inference --config signal_classifier/config.yaml [--split {test|val|train|human_test}]

Output formats: CSV or JSON (set in config inference.output_format).

CSV columns for all splits:
    sequence_id, camera, recording, frame_index, predicted, ground_truth/human_label, 
    correct, confidence, none_prob, left_prob, right_prob, hazard_prob
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .dataset import SlidingWindowDataset, HumanTestDataset, LABEL_NAMES, LABEL_MAP
from .model import SignalClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Inference over one sequence ───────────────────────────────────────────────

@torch.no_grad()
def predict_sequence_framewise(
    feature_tensor: torch.Tensor,   # [T_full, P, D]
    model: nn.Module,
    window_size: int,
    stride: int,
    device: str,
) -> torch.Tensor:
    """
    Slide a window over the full sequence and aggregate per-frame predictions.
    For each frame, average predictions from all windows that include it.
    
    Returns
    -------
    per_frame_probs : FloatTensor [T_full, 4] - softmax probabilities per frame
    """
    model.eval()
    T_full = feature_tensor.shape[0]
    per_frame_counts = torch.zeros(T_full, 4)  # accumulate counts per frame-class
    per_frame_probs = torch.zeros(T_full, 4)   # accumulate probabilities
    
    for start in range(0, max(T_full - window_size + 1, 1), stride):
        end = start + window_size
        if end > T_full:
            # Pad the last window by repeating the last frame
            window = feature_tensor[start:]
            pad = window[-1:].expand(end - T_full, -1, -1)
            window = torch.cat([window, pad], dim=0)
            end = T_full  # only count up to actual sequence length
        else:
            window = feature_tensor[start:end]
        
        window_tensor = window.unsqueeze(0).float().to(device)    # [1, T, P, D]
        logits = model(window_tensor)                              # [1, 4]
        probs = F.softmax(logits, dim=-1).squeeze(0)              # [4]
        
        # Accumulate into frames [start:end]
        per_frame_probs[start:end] += probs.cpu()
        per_frame_counts[start:end] += 1.0
    
    # Average by count (handle potential divide by zero)
    per_frame_counts[per_frame_counts == 0] = 1.0
    per_frame_probs = per_frame_probs / per_frame_counts
    
    return per_frame_probs  # [T_full, 4]


@torch.no_grad()
def infer_human_test(cfg: dict, model: nn.Module, device: str, split_name: str):
    """
    Run per-frame inference on human-labeled test set.
    Evaluates frame-level predictions against human ground truth, gracefully handling
    sequences with label transitions.
    """
    ic = cfg["inference"]
    fc = cfg["features"]
    wc = cfg["window"]
    dc = cfg["data"]
    output_dir = Path(ic.get("output_dir", "results/signal_classifier/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(fc["output_dir"]) / "metadata.json"
    
    # Load human test dataset
    log.info("Loading human-labeled test set for per-frame evaluation...")
    human_ds = HumanTestDataset(
        front_csv      = dc.get("human_test_front_csv"),
        back_csv       = dc.get("human_test_back_csv"),
        label_column   = dc.get("human_label_column", "true_label"),
        metadata_path  = str(meta_path),
        crop_base_dir  = dc["crop_base_dir"],
        min_frames     = dc.get("min_sequence_frames", 10),
    )
    log.info(f"Loaded {len(human_ds)} human-labeled sequences")

    model.eval()
    all_results = []         # flat list of per-frame predictions
    all_pred_labels = []
    all_human_labels = []
    
    window_size = wc["size"]
    stride      = wc["inference_stride"]

    for idx in range(len(human_ds)):
        features, per_frame_labels, seq_id = human_ds[idx]  # [T, P, D], [T], str
        features = features.to(device)
        T_seq = features.shape[0]
        
        # Get per-frame predictions by sliding windows
        per_frame_probs = predict_sequence_framewise(features, model, window_size, stride, device)  # [T, 4]
        
        # Convert to class predictions
        per_frame_preds = per_frame_probs.argmax(dim=1)  # [T]
        
        # Store per-frame results
        for frame_idx in range(T_seq):
            pred_cls = per_frame_preds[frame_idx].item()
            human_cls = per_frame_labels[frame_idx].item()
            confidence = per_frame_probs[frame_idx, pred_cls].item()
            
            all_results.append({
                "sequence_id":   seq_id,
                "frame_index":   frame_idx,
                "predicted":     LABEL_NAMES[pred_cls],
                "human_label":   LABEL_NAMES[human_cls],
                "correct":       int(pred_cls == human_cls),
                "confidence":    round(confidence, 4),
                "none_prob":     round(per_frame_probs[frame_idx, 0].item(), 4),
                "left_prob":     round(per_frame_probs[frame_idx, 1].item(), 4),
                "right_prob":    round(per_frame_probs[frame_idx, 2].item(), 4),
                "hazard_prob":   round(per_frame_probs[frame_idx, 3].item(), 4),
            })
            
            all_pred_labels.append(pred_cls)
            all_human_labels.append(human_cls)

    # ── Evaluate at frame level ────────────────────────────────────────────────
    from .train import compute_metrics
    metrics = compute_metrics(all_pred_labels, all_human_labels)

    log.info(f"\n{'='*80}")
    log.info(f"HUMAN TEST SET — PER-FRAME EVALUATION")
    log.info(f"{'='*80}")
    log.info(f"Total frames evaluated: {len(all_results)}")
    log.info(f"Overall frame-level accuracy : {metrics['accuracy']:.4f}")
    log.info(f"Macro F1                    : {metrics['macro_f1']:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        log.info(f"  {name:7s}  acc={metrics['acc_per_class'][i]:.3f}  f1={metrics['f1_per_class'][i]:.3f}")
    log.info(f"{'='*80}\n")

    # ── Write output ──────────────────────────────────────────────────────────
    fmt = ic.get("output_format", "csv")
    if fmt == "csv":
        import csv
        out_path = output_dir / f"predictions_{split_name}_framewise.csv"
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Per-frame predictions written to {out_path}")
    else:
        out_path = output_dir / f"predictions_{split_name}_framewise.json"
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics, "predictions": all_results}, f, indent=2)
        log.info(f"Per-frame predictions written to {out_path}")


@torch.no_grad()
def predict_sequence(
    feature_tensor: torch.Tensor,   # [T_full, P, D]
    model: nn.Module,
    window_size: int,
    stride: int,
    device: str,
) -> torch.Tensor:
    """
    Slide a window over the full feature tensor and return the mean softmax 
    probability vector [4] across all windows (for sequence-level inference).
    """
    model.eval()
    T_full = feature_tensor.shape[0]
    window_probs: List[torch.Tensor] = []

    for start in range(0, max(T_full - window_size + 1, 1), stride):
        end = start + window_size
        if end > T_full:
            # Pad the last (short) window by repeating the last frame
            window = feature_tensor[start:]
            pad = window[-1:].expand(end - T_full, -1, -1)
            window = torch.cat([window, pad], dim=0)
        else:
            window = feature_tensor[start:end]

        window = window.unsqueeze(0).float().to(device)    # [1, T, P, D]
        logits = model(window)                             # [1, 4]
        probs  = F.softmax(logits, dim=-1).squeeze(0)      # [4]
        window_probs.append(probs.cpu())

    if not window_probs:
        return torch.full((4,), 0.25)

    return torch.stack(window_probs).mean(dim=0)           # [4]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run inference with SignalClassifier.")
    parser.add_argument("--config", default="signal_classifier/config.yaml")
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "human_test"],
        default="test",
        help="Which dataset split to run inference on (default: test).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ic = cfg["inference"]
    fc = cfg["features"]
    wc = cfg["window"]
    mc = cfg["model"]
    dc = cfg["data"]

    device     = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    meta_path  = Path(fc["output_dir"]) / "metadata.json"
    ckpt_path  = Path(ic["checkpoint"])
    output_dir = Path(ic.get("output_dir", "results/signal_classifier/"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model = SignalClassifier(
        T          = wc["size"],
        P          = fc["spatial_tokens"],
        d_dino     = 384,
        d_model    = mc["d_model"],
        d_hidden   = mc["d_hidden"],
        num_layers = mc["num_layers"],
        num_heads  = mc["num_heads"],
        dropout    = mc.get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # ── Route to correct inference path ─────────────────────────────────────
    if args.split == "human_test":
        return infer_human_test(cfg, model, device, args.split)

    # ── Autolabeled splits (test, val, train) proceed here ──────────────────
    # Load metadata
    with open(meta_path) as f:
        meta: Dict = json.load(f)

    # Determine which sequences belong to the requested split
    # (replicate the same recording-level split logic as dataset.py)
    from .dataset import SlidingWindowDataset
    dummy_ds = SlidingWindowDataset(
        metadata_path          = str(meta_path),
        split                  = args.split,
        window_size            = wc["size"],
        stride                 = wc["inference_stride"],
        label_purity_threshold = wc.get("label_purity_threshold", 0.80),
        train_ratio            = dc.get("train_ratio", 0.70),
        val_ratio              = dc.get("val_ratio",   0.15),
        split_seed             = dc.get("split_seed",  42),
    )

    # Collect unique (non-flipped) sequence keys in this split
    split_feature_files = {w.feature_file for w in dummy_ds.windows}

    # Build reverse map: feature_file → metadata entry
    file_to_entry = {
        e["feature_file"]: (k, e)
        for k, e in meta.items()
        if not e.get("is_flipped", False) and e["feature_file"] in split_feature_files
    }

    log.info(f"Running per-frame inference on {len(file_to_entry)} sequences ({args.split} split)...")

    all_results = []        # flat list of per-frame predictions
    all_pred_labels = []
    all_gt_labels = []
    
    window_size = wc["size"]
    stride      = wc["inference_stride"]

    for feature_file, (key, entry) in file_to_entry.items():
        features: torch.Tensor = torch.load(feature_file, weights_only=True)   # [T, P, D]
        seq_id = entry["sequence_id"]
        
        # Get per-frame predictions by sliding windows
        per_frame_probs = predict_sequence_framewise(features, model, window_size, stride, device)  # [T, 4]
        per_frame_preds = per_frame_probs.argmax(dim=1)  # [T]
        
        # Get ground-truth per-frame labels from metadata
        per_frame_gt = entry.get("per_frame_labels", [])
        
        # Store per-frame results
        for frame_idx in range(len(per_frame_gt)):
            if frame_idx >= features.shape[0]:
                break
                
            pred_cls = per_frame_preds[frame_idx].item()
            gt_cls = per_frame_gt[frame_idx]
            confidence = per_frame_probs[frame_idx, pred_cls].item()
            
            all_results.append({
                "sequence_id":   seq_id,
                "camera":        entry.get("camera", ""),
                "recording":     entry.get("recording", ""),
                "frame_index":   frame_idx,
                "predicted":     LABEL_NAMES[pred_cls],
                "ground_truth":  LABEL_NAMES[gt_cls],
                "correct":       int(pred_cls == gt_cls),
                "confidence":    round(confidence, 4),
                "none_prob":     round(per_frame_probs[frame_idx, 0].item(), 4),
                "left_prob":     round(per_frame_probs[frame_idx, 1].item(), 4),
                "right_prob":    round(per_frame_probs[frame_idx, 2].item(), 4),
                "hazard_prob":   round(per_frame_probs[frame_idx, 3].item(), 4),
            })
            
            all_pred_labels.append(pred_cls)
            all_gt_labels.append(gt_cls)

    # ── Evaluate at frame level ────────────────────────────────────────────────
    from .train import compute_metrics
    metrics = compute_metrics(all_pred_labels, all_gt_labels)
    
    log.info(f"\nPER-FRAME EVALUATION ({args.split} split)")
    log.info(f"Total frames evaluated: {len(all_results)}")
    log.info(f"Overall frame-level accuracy : {metrics['accuracy']:.4f}")
    log.info(f"Macro F1                    : {metrics['macro_f1']:.4f}")
    for i, name in enumerate(LABEL_NAMES):
        log.info(f"  {name:7s}  acc={metrics['acc_per_class'][i]:.3f}  f1={metrics['f1_per_class'][i]:.3f}")

    # ── Write output ──────────────────────────────────────────────────────────
    fmt = ic.get("output_format", "csv")
    if fmt == "csv":
        import csv
        out_path = output_dir / f"predictions_{args.split}_framewise.csv"
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Per-frame predictions written to {out_path}")
    else:
        out_path = output_dir / f"predictions_{args.split}_framewise.json"
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics, "predictions": all_results}, f, indent=2)
        log.info(f"Per-frame predictions written to {out_path}")


if __name__ == "__main__":
    main()
