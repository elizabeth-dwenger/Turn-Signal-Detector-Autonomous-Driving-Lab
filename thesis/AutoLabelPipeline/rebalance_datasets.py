#!/usr/bin/env python
"""
Rebalance labeled / unlabeled datasets by swapping entire sequences.

Moves sequences rich in true 'left' labels OUT of the labeled set into the
unlabeled set, and exchanges them with an equal number of all-'none'-predicted
sequences from the unlabeled set.

Updates four files in-place (backups are written alongside):
  1. results/reconstructed_cosmos_results_labeled.csv
  2. results/reconstructed_cosmos_results_unlabeled.csv
  3. data/tracking_data.csv
  4. data/back_unlabeled_default_none_label.csv

Usage:
  python rebalance_datasets.py                         # uses defaults
  python rebalance_datasets.py --target-left 5000      # explicit target
  python rebalance_datasets.py --dry-run               # preview only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ===================== paths (defaults) ====================================

BASE = Path(__file__).resolve().parent

DEFAULT_LABELED_RECON   = BASE / "results" / "reconstructed_cosmos_results_labeled.csv"
DEFAULT_UNLABELED_RECON = BASE / "results" / "reconstructed_cosmos_results_unlabeled.csv"
DEFAULT_TRACKING        = BASE / "data"    / "tracking_data.csv"
DEFAULT_BACK_UNLABELED  = BASE / "data"    / "back_unlabeled_default_none_label.csv"


# ===================== sequence selection ==================================

def select_sequences_to_move(
    labeled_df: pd.DataFrame,
    target: int = 5000,
    tolerance: int = 500,
    max_attempts: int = 5000,
    rng_seed: int = 42,
) -> list[str]:
    """
    Randomly select sequences from ``labeled_df`` whose total number of
    ``true_label == 'left'`` frames falls within ``target ± tolerance``.

    Uses repeated random sampling: shuffle the candidate pool, accumulate
    until we land inside the window.  Retries with different shuffles.
    """
    lo, hi = target - tolerance, target + tolerance
    rng = np.random.default_rng(rng_seed)

    # Count 'left' frames per sequence
    left_mask = labeled_df["true_label"].str.strip().str.lower() == "left"
    left_counts = (
        labeled_df.loc[left_mask]
        .groupby("sequence")
        .size()
    )
    left_counts = left_counts[left_counts > 0]

    if left_counts.sum() < lo:
        print(f"  [WARN] Total available 'left' frames ({left_counts.sum()}) "
              f"is below the target range [{lo}, {hi}].")
        print("         Will move ALL sequences with 'left' labels.")
        return list(left_counts.index)

    candidates = list(left_counts.items())  # [(seq_name, left_count), ...]

    best_selected = None
    best_total = 0
    best_diff = float("inf")

    for attempt in range(max_attempts):
        order = rng.permutation(len(candidates))
        selected = []
        running = 0
        for idx in order:
            seq, cnt = candidates[idx]
            if running + cnt > hi:
                continue  # skip this one, might overshoot
            selected.append(seq)
            running += cnt
            if running >= lo:
                break

        if lo <= running <= hi:
            diff = abs(running - target)
            if diff < best_diff or (
                diff == best_diff and len(selected) < len(best_selected or selected)
            ):
                best_selected = list(selected)
                best_total = running
                best_diff = diff
            if diff == 0:
                break  # perfect hit

    if best_selected is None:
        # Fallback: greedy largest-first
        print("  [WARN] Random sampling could not hit target; falling back to greedy.")
        left_counts_sorted = left_counts.sort_values(ascending=False)
        selected = []
        running = 0
        for seq, cnt in left_counts_sorted.items():
            if running >= lo:
                break
            selected.append(seq)
            running += cnt
        best_selected = selected
        best_total = running

    print(f"  Random selection: {len(best_selected)} sequences, "
          f"{best_total} 'left' frames (target {lo}-{hi})")
    return best_selected


def select_unlabeled_sequences_to_receive(
    unlabeled_df: pd.DataFrame,
    n: int,
    rng_seed: int = 42,
) -> list[str]:
    """
    Select ``n`` sequences from ``unlabeled_df`` that are entirely
    predicted as 'none' (i.e. all frames have ``predicted_label == 'none'``).
    """
    pred_col = "predicted_label"
    if pred_col not in unlabeled_df.columns:
        sys.exit("ERROR: 'predicted_label' column not found in unlabeled CSV.")

    # Find sequences where EVERY frame has predicted_label == 'none'
    def _all_none(group):
        return (group[pred_col].str.strip().str.lower() == "none").all()

    none_seqs = (
        unlabeled_df.groupby("sequence")
        .filter(_all_none)["sequence"]
        .unique()
    )

    if len(none_seqs) < n:
        print(f"  [WARN] Only {len(none_seqs)} all-none sequences available; "
              f"need {n}. Will use all of them.")
        return list(none_seqs[:n])

    # Random selection from the pool of all-none sequences
    rng = np.random.default_rng(rng_seed)
    chosen = rng.choice(none_seqs, size=n, replace=False)
    return list(chosen)


# ===================== column alignment helpers ============================

# Columns present in each file (based on pipeline data structures)
COMMON_COLS = [
    "sequence", "track_id", "frame_id", "class_id", "score",
    "x1", "y1", "x2", "y2", "crop_path", "img_path",
    "width", "height", "sequence_id", "sampled_frame_id",
    "signal_start_frame",
]


def _align_to_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ``df`` has ``true_label`` column (NaN if unknown)
    and does NOT have ``default_none_label``.
    """
    if "true_label" not in df.columns:
        df = df.copy()
        df["true_label"] = np.nan
    if "default_none_label" in df.columns:
        df = df.drop(columns=["default_none_label"])
    return df


def _align_to_unlabeled_recon(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the reconstructed unlabeled CSV: must have ``predicted_label``,
    must NOT have ``true_label`` or ``default_none_label``.
    """
    df = df.copy()
    if "true_label" in df.columns:
        df = df.drop(columns=["true_label"])
    if "default_none_label" in df.columns:
        df = df.drop(columns=["default_none_label"])
    if "predicted_label" not in df.columns:
        df["predicted_label"] = "none"
    return df


def _align_to_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    For ``tracking_data.csv``: has ``true_label``, no ``predicted_label``,
    no ``default_none_label``.
    """
    df = df.copy()
    for drop_col in ("predicted_label", "default_none_label"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    if "true_label" not in df.columns:
        df["true_label"] = np.nan
    return df


def _align_to_back_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """
    For ``back_unlabeled_default_none_label.csv``:
    has ``default_none_label``, no ``true_label``, no ``predicted_label``.
    """
    df = df.copy()
    for drop_col in ("predicted_label", "true_label"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    if "default_none_label" not in df.columns:
        df["default_none_label"] = "none"
    return df


# ===================== main ================================================

def rebalance(
    labeled_recon_path: Path,
    unlabeled_recon_path: Path,
    tracking_path: Path,
    back_unlabeled_path: Path,
    target_left: int = 5000,
    tolerance: int = 500,
    target_sequences: int = 1010,
    dry_run: bool = False,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print("REBALANCE DATASETS")
    print(sep)

    # ---- 1. Load all four CSVs -------------------------------------------
    print("\n[1/6] Loading CSVs ...")
    recon_lab   = pd.read_csv(labeled_recon_path)
    recon_unlab = pd.read_csv(unlabeled_recon_path)
    tracking    = pd.read_csv(tracking_path)
    back_unlab  = pd.read_csv(back_unlabeled_path)

    print(f"  reconstructed labeled   : {len(recon_lab):>8} rows")
    print(f"  reconstructed unlabeled : {len(recon_unlab):>8} rows")
    print(f"  tracking_data           : {len(tracking):>8} rows")
    print(f"  back_unlabeled          : {len(back_unlab):>8} rows")

    # ---- 1b. Detect & resolve pre-existing overlaps ----------------------
    pre_lab_seqs  = set(recon_lab["sequence"].unique())
    pre_unlab_seqs = set(recon_unlab["sequence"].unique())
    pre_overlap = pre_lab_seqs & pre_unlab_seqs

    if pre_overlap:
        print(f"\n  [INFO] {len(pre_overlap)} sequences already exist in BOTH datasets.")
        print(f"         Deduplicating: keeping duplicates in LABELED, removing from unlabeled.")

        recon_unlab = recon_unlab[~recon_unlab["sequence"].isin(pre_overlap)]

        # Also deduplicate the source CSVs the same way
        track_seqs = set(tracking["sequence"].unique())
        back_seqs  = set(back_unlab["sequence"].unique())
        source_overlap = track_seqs & back_seqs
        if source_overlap:
            back_unlab = back_unlab[~back_unlab["sequence"].isin(source_overlap)]
            print(f"         Also resolved {len(source_overlap)} overlaps in source CSVs.")

        recon_lab = recon_lab.reset_index(drop=True)
        recon_unlab = recon_unlab.reset_index(drop=True)
        tracking = tracking.reset_index(drop=True)
        back_unlab = back_unlab.reset_index(drop=True)

        print(f"  After dedup:")
        print(f"    reconstructed labeled   : {len(recon_lab):>8} rows")
        print(f"    reconstructed unlabeled : {len(recon_unlab):>8} rows")
        print(f"    tracking_data           : {len(tracking):>8} rows")
        print(f"    back_unlabeled          : {len(back_unlab):>8} rows")

    # ---- 2. Select sequences to move labeled → unlabeled -----------------
    print(f"\n[2/6] Selecting labeled sequences with ~{target_left} 'left' frames ...")
    seqs_to_unlabeled = select_sequences_to_move(
        recon_lab, target=target_left, tolerance=tolerance
    )
    n_move = len(seqs_to_unlabeled)

    # Count left frames being moved
    move_mask_lab = recon_lab["sequence"].isin(seqs_to_unlabeled)
    left_frames_moved = (
        recon_lab.loc[move_mask_lab, "true_label"]
        .str.strip().str.lower()
        .eq("left").sum()
    )
    total_frames_moved_from_lab = move_mask_lab.sum()

    print(f"  Sequences selected      : {n_move}")
    print(f"  Total frames moved      : {total_frames_moved_from_lab}")
    print(f"  'left' frames moved     : {left_frames_moved}")
    if seqs_to_unlabeled:
        print(f"  Sequences:")
        for s in seqs_to_unlabeled[:20]:
            seq_left = (
                recon_lab.loc[
                    (recon_lab["sequence"] == s),
                    "true_label"
                ].str.strip().str.lower().eq("left").sum()
            )
            seq_total = (recon_lab["sequence"] == s).sum()
            print(f"    {s}  ({seq_left} left / {seq_total} total)")
        if len(seqs_to_unlabeled) > 20:
            print(f"    ... and {len(seqs_to_unlabeled) - 20} more")

    # ---- 3. Compute how many sequences to receive to hit target ----------
    current_lab_seqs = recon_lab[~move_mask_lab]["sequence"].nunique()
    needed_from_unlab = target_sequences - current_lab_seqs

    print(f"\n[3/6] Balancing to exactly {target_sequences} labeled sequences ...")
    print(f"  Labeled sequences after removing {n_move}: {current_lab_seqs}")

    if needed_from_unlab > 0:
        print(f"  Need {needed_from_unlab} sequences from unlabeled to reach {target_sequences}")
        seqs_to_labeled = select_unlabeled_sequences_to_receive(recon_unlab, needed_from_unlab)
    elif needed_from_unlab < 0:
        # Too many labeled sequences — move excess (all-none-predicted) to unlabeled
        excess = -needed_from_unlab
        print(f"  {excess} excess labeled sequences — moving all-none-predicted ones out")
        remaining_lab = recon_lab[~move_mask_lab]
        # Find labeled sequences where all predicted_label == 'none'
        pred_col = "predicted_label"
        if pred_col in remaining_lab.columns:
            def _all_none_lab(group):
                return (group[pred_col].str.strip().str.lower() == "none").all()
            candidate_seqs = (
                remaining_lab.groupby("sequence")
                .filter(_all_none_lab)["sequence"]
                .unique()
            )
        else:
            candidate_seqs = remaining_lab["sequence"].unique()
        extra_to_move = list(candidate_seqs[:excess])
        seqs_to_unlabeled = seqs_to_unlabeled + extra_to_move
        move_mask_lab = recon_lab["sequence"].isin(seqs_to_unlabeled)
        n_move = len(seqs_to_unlabeled)
        print(f"  Will move {len(extra_to_move)} additional sequences out")
        seqs_to_labeled = []
    else:
        print(f"  Already exactly {target_sequences} — no additional swap needed")
        seqs_to_labeled = []

    n_swap = len(seqs_to_labeled)
    move_mask_unlab = recon_unlab["sequence"].isin(seqs_to_labeled)
    total_frames_moved_from_unlab = move_mask_unlab.sum()

    print(f"  Sequences moving unlabeled → labeled : {n_swap}")
    print(f"  Frames moving unlabeled → labeled    : {total_frames_moved_from_unlab}")
    if seqs_to_labeled:
        for s in list(seqs_to_labeled)[:20]:
            seq_total = (recon_unlab["sequence"] == s).sum()
            print(f"    {s}  ({seq_total} frames)")
        if len(seqs_to_labeled) > 20:
            print(f"    ... and {len(seqs_to_labeled) - 20} more")

    # ---- 4. Verify no overlap between swap sets --------------------------
    overlap = set(seqs_to_unlabeled) & set(seqs_to_labeled)
    if overlap:
        sys.exit(f"ERROR: {len(overlap)} sequences selected for BOTH directions: {overlap}")

    # Also ensure sequences being moved don't already exist in the target
    already_in_unlab = set(seqs_to_unlabeled) & set(recon_unlab["sequence"].unique())
    if already_in_unlab:
        print(f"  [WARN] {len(already_in_unlab)} sequences to move already exist in "
              f"unlabeled set — will be deduplicated.")
        recon_unlab = recon_unlab[~recon_unlab["sequence"].isin(already_in_unlab)].reset_index(drop=True)
        back_unlab = back_unlab[~back_unlab["sequence"].isin(already_in_unlab)].reset_index(drop=True)

    already_in_lab = set(seqs_to_labeled) & set(recon_lab["sequence"].unique())
    if already_in_lab:
        print(f"  [WARN] {len(already_in_lab)} sequences to receive already exist in "
              f"labeled set — will be deduplicated.")
        recon_lab = recon_lab[~recon_lab["sequence"].isin(already_in_lab)].reset_index(drop=True)
        tracking = tracking[~tracking["sequence"].isin(already_in_lab)].reset_index(drop=True)

    # Recompute masks after potential dedup
    move_mask_lab = recon_lab["sequence"].isin(seqs_to_unlabeled)
    move_mask_unlab = recon_unlab["sequence"].isin(seqs_to_labeled)

    # ---- 5. Perform the swap ---------------------------------------------
    print(f"\n[4/6] Building new DataFrames ...")

    # Rows moving labeled → unlabeled
    rows_lab_to_unlab = recon_lab[move_mask_lab].copy()
    # Rows moving unlabeled → labeled
    rows_unlab_to_lab = recon_unlab[move_mask_unlab].copy()

    print(f"  Rows labeled → unlabeled : {len(rows_lab_to_unlab)}")
    print(f"  Rows unlabeled → labeled : {len(rows_unlab_to_lab)}")

    # --- Reconstructed labeled: remove moved-out, add moved-in ---
    new_recon_lab = pd.concat([
        recon_lab[~move_mask_lab],
        _align_to_labeled(rows_unlab_to_lab),
    ], ignore_index=True)
    # Ensure predicted_label column exists
    if "predicted_label" not in new_recon_lab.columns:
        new_recon_lab["predicted_label"] = "none"

    # --- Reconstructed unlabeled: remove moved-out, add moved-in ---
    new_recon_unlab = pd.concat([
        recon_unlab[~move_mask_unlab],
        _align_to_unlabeled_recon(rows_lab_to_unlab),
    ], ignore_index=True)

    # --- tracking_data.csv: same swap, different column set ---
    tracking_move_mask = tracking["sequence"].isin(seqs_to_unlabeled)
    rows_tracking_to_unlab = tracking[tracking_move_mask].copy()

    # Rows coming FROM unlabeled (use back_unlabeled as source for column parity)
    back_move_mask_in = back_unlab["sequence"].isin(seqs_to_labeled)
    rows_back_to_tracking = back_unlab[back_move_mask_in].copy()

    new_tracking = pd.concat([
        tracking[~tracking_move_mask],
        _align_to_tracking(rows_back_to_tracking),
    ], ignore_index=True)

    # --- back_unlabeled: remove moved-out, add moved-in ---
    back_move_mask_out = back_unlab["sequence"].isin(seqs_to_labeled)
    new_back_unlab = pd.concat([
        back_unlab[~back_move_mask_out],
        _align_to_back_unlabeled(rows_tracking_to_unlab),
    ], ignore_index=True)

    # ---- 6. Summary & sanity checks -------------------------------------
    print(f"\n[5/6] Sanity checks ...")

    # No sequence in both labeled and unlabeled
    lab_seqs = set(new_recon_lab["sequence"].unique())
    unlab_seqs = set(new_recon_unlab["sequence"].unique())
    dups = lab_seqs & unlab_seqs
    if dups:
        print(f"  [ERROR] {len(dups)} sequences appear in BOTH datasets!")
        for d in list(dups)[:10]:
            print(f"    - {d}")
        sys.exit(1)
    else:
        print("  No sequences duplicated across labeled/unlabeled. OK")

    # Sequence count check
    n_lab_seqs = new_recon_lab["sequence"].nunique()
    print(f"  Labeled sequences  : {n_lab_seqs}  (target: {target_sequences})")
    if n_lab_seqs != target_sequences:
        print(f"  [WARN] Labeled sequence count is {n_lab_seqs}, not {target_sequences}!")
    else:
        print(f"  Labeled sequence count matches target. OK")

    # Row counts
    orig_total = len(recon_lab) + len(recon_unlab)
    new_total  = len(new_recon_lab) + len(new_recon_unlab)
    print(f"  Total rows (before): {orig_total}")
    print(f"  Total rows (after) : {new_total}")
    if orig_total != new_total:
        print(f"  [WARN] Row count changed by {new_total - orig_total}")

    # Left-frame counts in labeled set before/after
    def _count_left(df):
        if "true_label" not in df.columns:
            return 0
        return df["true_label"].str.strip().str.lower().eq("left").sum()

    left_before = _count_left(recon_lab)
    left_after  = _count_left(new_recon_lab)
    print(f"  'left' frames in labeled (before): {left_before}")
    print(f"  'left' frames in labeled (after) : {left_after}")
    print(f"  'left' frames moved              : {left_frames_moved}")

    print(f"\n  New file sizes:")
    print(f"    reconstructed labeled   : {len(new_recon_lab):>8} rows")
    print(f"    reconstructed unlabeled : {len(new_recon_unlab):>8} rows")
    print(f"    tracking_data           : {len(new_tracking):>8} rows")
    print(f"    back_unlabeled          : {len(new_back_unlab):>8} rows")

    if dry_run:
        print(f"\n  ** DRY RUN — no files written **")
        print(f"\n{sep}\nDONE (dry run)\n{sep}\n")
        return

    # ---- 7. Back up originals & write ------------------------------------
    print(f"\n[6/6] Saving updated CSVs ...")

    for path in (labeled_recon_path, unlabeled_recon_path, tracking_path, back_unlabeled_path):
        bak = path.with_suffix(".csv.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  Backed up: {bak}")

    new_recon_lab.to_csv(labeled_recon_path, index=False)
    print(f"  Saved: {labeled_recon_path}")

    new_recon_unlab.to_csv(unlabeled_recon_path, index=False)
    print(f"  Saved: {unlabeled_recon_path}")

    new_tracking.to_csv(tracking_path, index=False)
    print(f"  Saved: {tracking_path}")

    new_back_unlab.to_csv(back_unlabeled_path, index=False)
    print(f"  Saved: {back_unlabeled_path}")

    print(f"\n{sep}")
    print("REBALANCE COMPLETE")
    print(sep + "\n")


# ===================== CLI =================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rebalance labeled/unlabeled datasets by swapping sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--labeled-recon", type=Path, default=DEFAULT_LABELED_RECON,
        help="Path to reconstructed labeled CSV.",
    )
    parser.add_argument(
        "--unlabeled-recon", type=Path, default=DEFAULT_UNLABELED_RECON,
        help="Path to reconstructed unlabeled CSV.",
    )
    parser.add_argument(
        "--tracking", type=Path, default=DEFAULT_TRACKING,
        help="Path to tracking_data.csv.",
    )
    parser.add_argument(
        "--back-unlabeled", type=Path, default=DEFAULT_BACK_UNLABELED,
        help="Path to back_unlabeled_default_none_label.csv.",
    )
    parser.add_argument(
        "--target-left", type=int, default=5000,
        help="Target number of 'left' frames to move (default: 5000).",
    )
    parser.add_argument(
        "--tolerance", type=int, default=500,
        help="Acceptable tolerance around target (default: 500).",
    )
    parser.add_argument(
        "--target-sequences", type=int, default=1010,
        help="Exact number of sequences in the labeled datasets (default: 1010).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without writing files.",
    )

    args = parser.parse_args()

    for path, label in [
        (args.labeled_recon, "reconstructed labeled"),
        (args.unlabeled_recon, "reconstructed unlabeled"),
        (args.tracking, "tracking_data"),
        (args.back_unlabeled, "back_unlabeled"),
    ]:
        if not path.exists():
            sys.exit(f"ERROR: {label} CSV not found: {path}")

    rebalance(
        labeled_recon_path=args.labeled_recon,
        unlabeled_recon_path=args.unlabeled_recon,
        tracking_path=args.tracking,
        back_unlabeled_path=args.back_unlabeled,
        target_left=args.target_left,
        tolerance=args.tolerance,
        target_sequences=args.target_sequences,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
