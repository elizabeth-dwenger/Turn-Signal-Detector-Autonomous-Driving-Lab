#!/usr/bin/env python
"""
Reconstruct full dataset with integrated predictions.

Takes original tracking CSV and per-sequence prediction CSVs from a pipeline run, then produces a single CSV with predictions mapped back to every frame (including frames skipped by downsampling, via forward-fill propagation).

Usage:
  # Unlabeled dataset
  python reconstruct_predictions.py \
      --original-csv data/back_unlabeled_default_none_label.csv \
      --results-dir results/cosmos_reason2_video_unlabeled/full_runs/cosmos_reason2_20260301_235955 \
      --mode unlabeled \
      --output reconstructed_unlabeled.csv

  # Labeled dataset
  python reconstruct_predictions.py \
      --original-csv data/tracking_data.csv \
      --results-dir results/cosmos_reason2_video/full_runs/cosmos_reason2_20260226_195914 \
      --mode labeled \
      --output reconstructed_labeled.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(sequence_id: str) -> str:
    """Mirror the pipeline's filename sanitisation (slashes → underscores)."""
    return sequence_id.replace("/", "_").replace("\\", "_")


def _strip_track_suffix(sequence_id: str) -> tuple[str, int | None]:
    """
    Normalize sequence ids that may already contain ``__track_<id>``.
    Returns ``(base_sequence_id, embedded_track_id)``.
    """
    match = re.match(r"^(.*)__track_(\d+)$", str(sequence_id))
    if not match:
        return str(sequence_id), None
    return match.group(1), int(match.group(2))


def _build_sequence_key(sequence_id: str, track_id: int) -> str:
    """Build the key used by the pipeline: ``sequence_id__track_<track_id>``."""
    base_sequence_id, embedded_track_id = _strip_track_suffix(sequence_id)
    resolved_track_id = embedded_track_id if embedded_track_id is not None else int(track_id)
    return f"{base_sequence_id}__track_{resolved_track_id}"


def _safe_sequence_key(sequence_id: str, track_id: int) -> str:
    """Filename-safe version of the sequence key."""
    return _safe_filename(_build_sequence_key(sequence_id, track_id))


# ---------------------------------------------------------------------------
# Load predictions from the results directory
# ---------------------------------------------------------------------------

def load_predictions(results_dir: Path) -> dict:
    """
    Load all per-sequence prediction CSVs from a pipeline run directory.
    """
    unified_frame_csv = results_dir / "frame_predictions.csv"
    if unified_frame_csv.exists():
        df = pd.read_csv(unified_frame_csv)
        required = {"sequence_id", "track_id", "frame_id", "label"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(
                "ERROR: frame_predictions.csv is missing required columns: "
                + ", ".join(sorted(missing))
            )

        predictions = {}
        for (sequence_id, track_id), group in df.groupby(["sequence_id", "track_id"], sort=False):
            normalized_sequence_id, embedded_track_id = _strip_track_suffix(str(sequence_id))
            resolved_track_id = embedded_track_id if embedded_track_id is not None else int(track_id)
            key = _safe_sequence_key(normalized_sequence_id, resolved_track_id)
            predictions[key] = group.sort_values("frame_id").reset_index(drop=True)

        print(
            f"  Loaded unified frame predictions for {len(predictions)} sequences "
            f"from {unified_frame_csv}"
        )
        return predictions

    results_json = results_dir / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            payload = json.load(f)

        predictions = {}
        for item in payload.get("results", []):
            sequence_id = item.get("sequence_id")
            track_id = item.get("track_id")
            if sequence_id is None or track_id is None:
                continue

            normalized_sequence_id, embedded_track_id = _strip_track_suffix(str(sequence_id))
            resolved_track_id = embedded_track_id if embedded_track_id is not None else int(track_id)
            key = _safe_sequence_key(normalized_sequence_id, resolved_track_id)
            predictions[key] = {
                "label": item.get("prediction", {}).get("label", "none"),
                "segments": item.get("prediction", {}).get("segments", []),
            }

        print(
            f"  Loaded sequence-level heuristic predictions for {len(predictions)} sequences "
            f"from {results_json}"
        )
        return predictions

    csv_files = sorted(results_dir.glob("*.csv"))

    # Exclude metadata / summary files produced by the pipeline
    exclude_patterns = [
        "dataset_summary",
        "pipeline_report",
        "evaluation_metrics",
        "evaluation_per_sequence",
        "review_queue",
        "unlabeled_per_sequence",
        "unlabeled_analysis",
        "model_metrics",
    ]

    predictions = {}
    for csv_path in csv_files:
        stem = csv_path.stem
        if any(pat in stem for pat in exclude_patterns):
            continue
        # Only load CSVs that look like prediction files (must have frame_id & label)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  [WARN] Could not read {csv_path.name}: {exc}")
            continue

        if "frame_id" not in df.columns or "label" not in df.columns:
            continue

        predictions[stem] = df

    print(f"  Loaded predictions for {len(predictions)} sequences from {results_dir}")
    return predictions


# ---------------------------------------------------------------------------
# Match predictions to original rows
# ---------------------------------------------------------------------------

def _match_key(row_seq_id: str, row_track_id: int, pred_keys: set) -> str | None:
    """
    Try to find the matching prediction key for a given
    (sequence_id, track_id) pair.

    The pipeline saves files as ``<safe_sequence_key>.csv`` where the key is
    ``sequence_id__track_<track_id>`` with slashes turned into underscores.
    """
    candidate_keys = []

    candidate_keys.append(_safe_sequence_key(row_seq_id, row_track_id))

    normalized_row_seq_id, embedded_track_id = _strip_track_suffix(row_seq_id)
    if normalized_row_seq_id != row_seq_id or embedded_track_id is not None:
        candidate_keys.append(
            _safe_sequence_key(
                normalized_row_seq_id,
                embedded_track_id if embedded_track_id is not None else row_track_id,
            )
        )

    for candidate in candidate_keys:
        if candidate in pred_keys:
            return candidate

    # Fallback: try matching with different track_id formatting
    # (e.g. zero-padded, etc.) — unlikely but defensive
    for key in pred_keys:
        if key.endswith(f"__track_{row_track_id}"):
            prefix = key[: -(len(f"__track_{row_track_id}"))]
            if prefix in {_safe_filename(row_seq_id), _safe_filename(normalized_row_seq_id)}:
                return key

    return None


# ---------------------------------------------------------------------------
# Forward-fill / propagate labels to non-predicted frames
# ---------------------------------------------------------------------------

def propagate_labels(
    original_group: pd.DataFrame,
    pred_df,
) -> pd.Series:
    """
    For a single (sequence_id, track_id) group, produce a predicted label
    for **every** frame_id in the original data—including frames that were
    skipped by downsampling.

    Strategy
    --------
    1. Build a mapping ``frame_id -> label`` from the prediction CSV.
    2. Sort original frames by frame_id.
    3. Assign known labels, then **forward-fill** gaps (``ffill``).
       If the earliest frames have no prediction, **back-fill** those
       (``bfill``) so no NaN remains.
    """
    group_sorted = original_group.sort_values("frame_id").copy()

    if isinstance(pred_df, dict):
        group_sorted["_pred"] = pred_df.get("label", "none")
        segments = pred_df.get("segments", []) or []

        if segments:
            group_sorted["_pred"] = "none"
            labels_by_position = ["none"] * len(group_sorted)
            default_label = pred_df.get("label", "none")

            for seg in segments:
                label = seg.get("label", default_label)
                start = max(0, int(seg.get("start_frame", 0)))
                end = min(len(labels_by_position) - 1, int(seg.get("end_frame", len(labels_by_position) - 1)))
                for idx in range(start, end + 1):
                    labels_by_position[idx] = label

            group_sorted["_pred"] = labels_by_position

        return group_sorted.set_index(group_sorted.index)["_pred"]

    pred_map = dict(zip(pred_df["frame_id"], pred_df["label"]))

    # Map known predictions
    group_sorted["_pred"] = group_sorted["frame_id"].map(pred_map)

    # Forward-fill then back-fill
    group_sorted["_pred"] = group_sorted["_pred"].ffill().bfill()

    # If still any NaN (no predictions at all for this track), default to "none"
    group_sorted["_pred"] = group_sorted["_pred"].fillna("none")

    # Return series aligned to the original index
    return group_sorted.set_index(group_sorted.index)["_pred"]


# ---------------------------------------------------------------------------
# Main reconstruction
# ---------------------------------------------------------------------------

def reconstruct(
    original_csv: Path,
    results_dir: Path,
    mode: str,
    output_path: Path,
):
    """
    Parameters
    ----------
    original_csv : Path
        Path to the original tracking CSV (labeled or unlabeled).
    results_dir : Path
        Path to the pipeline run directory containing per-sequence CSVs.
    mode : str
        ``"labeled"`` or ``"unlabeled"``.
    output_path : Path
        Where to write the reconstructed CSV.
    """
    # ---- 1. Load original data ----
    print(f"\n{'='*70}")
    print("RECONSTRUCT PREDICTIONS")
    print(f"{'='*70}")
    print(f"  Original CSV : {original_csv}")
    print(f"  Results dir  : {results_dir}")
    print(f"  Mode         : {mode}")
    print(f"  Output       : {output_path}")

    orig = pd.read_csv(original_csv)
    print(f"\n  Original dataset: {len(orig)} rows")

    # Detect the label column in the original data
    if mode == "unlabeled":
        if "default_none_label" not in orig.columns:
            print("  [WARN] 'default_none_label' column not found; "
                  "will add 'predicted_label' column instead.")
    else:
        if "true_label" not in orig.columns:
            sys.exit("ERROR: labeled mode selected but 'true_label' column "
                     "not found in original CSV.")

    # ---- 2. Load predictions ----
    pred_dict = load_predictions(results_dir)
    if not pred_dict:
        sys.exit("ERROR: No valid prediction CSVs found in results directory.")

    pred_keys = set(pred_dict.keys())

    # ---- 3. Assign predicted labels per (sequence_id, track_id) group ----
    print("\n  Matching predictions to original rows ...")

    # We will populate this column
    orig["predicted_label"] = pd.Series(dtype="object")

    matched_sequences = 0
    unmatched_sequences = []
    total_predicted_frames = 0
    total_propagated_frames = 0

    grouped = orig.groupby(["sequence_id", "track_id"], sort=False)

    for (seq_id, track_id), group_idx in grouped.groups.items():
        group = orig.loc[group_idx]
        key = _match_key(seq_id, int(track_id), pred_keys)

        if key is None:
            unmatched_sequences.append((seq_id, track_id))
            # Default to "none" for unmatched sequences
            orig.loc[group_idx, "predicted_label"] = "none"
            continue

        pred_df = pred_dict[key]
        matched_sequences += 1

        # Count how many frames have direct predictions
        if isinstance(pred_df, dict):
            segments = pred_df.get("segments", []) or []
            if segments:
                predicted_ids = set()
                group_sorted = group.sort_values("frame_id")
                frame_ids = list(group_sorted["frame_id"])
                for seg in segments:
                    start = max(0, int(seg.get("start_frame", 0)))
                    end = min(len(frame_ids) - 1, int(seg.get("end_frame", len(frame_ids) - 1)))
                    for idx in range(start, end + 1):
                        predicted_ids.add(frame_ids[idx])
                direct_hits = group["frame_id"].isin(predicted_ids).sum()
            else:
                direct_hits = len(group)
        else:
            direct_hits = group["frame_id"].isin(pred_df["frame_id"]).sum()
        total_predicted_frames += direct_hits
        total_propagated_frames += len(group) - direct_hits

        # Propagate labels (handles downsampling gaps)
        labels = propagate_labels(group, pred_df)
        orig.loc[labels.index, "predicted_label"] = labels.values

    print(f"  Matched sequences     : {matched_sequences}")
    print(f"  Unmatched sequences   : {len(unmatched_sequences)}")
    if unmatched_sequences and len(unmatched_sequences) <= 20:
        for seq_id, track_id in unmatched_sequences:
            print(f"    - {seq_id}  track {track_id}")
    elif unmatched_sequences:
        print(f"    (showing first 20 of {len(unmatched_sequences)})")
        for seq_id, track_id in unmatched_sequences[:20]:
            print(f"    - {seq_id}  track {track_id}")
    print(f"  Directly predicted    : {total_predicted_frames} frames")
    print(f"  Gap-filled (propagated): {total_propagated_frames} frames")

    # ---- 4. Finalise output columns ----
    if mode == "unlabeled":
        # Replace the placeholder column with predictions
        if "default_none_label" in orig.columns:
            orig = orig.drop(columns=["default_none_label"])
            # Rename predicted_label to the position where default_none_label was
            # (keeps the column; user can rename later if desired)
            orig = orig.rename(columns={"predicted_label": "predicted_label"})
            print("\n  Replaced 'default_none_label' with 'predicted_label'.")
        else:
            print("\n  Added 'predicted_label' column.")
    else:
        # Labeled: keep true_label, add predicted_label alongside
        print("\n  Added 'predicted_label' column alongside existing 'true_label'.")

    # ---- 5. Sanity checks ----
    null_count = orig["predicted_label"].isna().sum()
    if null_count > 0:
        print(f"  [WARN] {null_count} rows still have no predicted label "
              f"(filling with 'none').")
        orig["predicted_label"] = orig["predicted_label"].fillna("none")

    # ---- 6. Save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    orig.to_csv(output_path, index=False)
    print(f"\n  Saved reconstructed CSV: {output_path}  ({len(orig)} rows)")

    # ---- 7. Quick summary stats ----
    label_counts = orig["predicted_label"].value_counts()
    print(f"\n  Predicted label distribution:")
    for label, count in label_counts.items():
        pct = count / len(orig) * 100
        print(f"    {str(label):.<20} {count:>7} ({pct:>5.1f}%)")

    if mode == "labeled" and "true_label" in orig.columns:
        match_mask = orig["true_label"].str.strip().str.lower() == orig["predicted_label"].str.strip().str.lower()
        accuracy = match_mask.mean()
        print(f"\n  Quick accuracy check (true vs predicted): {accuracy:.4f}")

    print(f"\n{'='*70}")
    print("RECONSTRUCTION COMPLETE")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct full dataset with integrated pipeline predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        required=True,
        help="Path to the original tracking CSV (labeled or unlabeled).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Path to the pipeline run directory with per-sequence CSV predictions.",
    )
    parser.add_argument(
        "--mode",
        choices=["labeled", "unlabeled"],
        required=True,
        help="'labeled' keeps true_label and adds predicted_label; "
             "'unlabeled' replaces default_none_label with predicted_label.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <results-dir>/reconstructed_<mode>.csv",
    )

    args = parser.parse_args()

    if not args.original_csv.exists():
        sys.exit(f"ERROR: Original CSV not found: {args.original_csv}")
    if not args.results_dir.is_dir():
        sys.exit(f"ERROR: Results directory not found: {args.results_dir}")

    output_path = args.output or (args.results_dir / f"reconstructed_{args.mode}.csv")

    reconstruct(
        original_csv=args.original_csv,
        results_dir=args.results_dir,
        mode=args.mode,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
