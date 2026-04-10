#!/usr/bin/env python
"""
Run heuristic inference on front/back unlabeled datasets and reconstruct
classifier-ready CSVs with predicted_label columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from heuristic_method.test_heuristic import test_heuristic
from reconstruct_predictions import reconstruct


def main():
    parser = argparse.ArgumentParser(
        description="Prepare heuristic pseudo-labeled CSVs for classifier training."
    )
    parser.add_argument(
        "--front-config",
        default="configs/front_heuristic_unlabeled.yaml",
        help="Heuristic config for front unlabeled data.",
    )
    parser.add_argument(
        "--back-config",
        default="configs/back_heuristic_unlabeled.yaml",
        help="Heuristic config for back unlabeled data.",
    )
    parser.add_argument(
        "--front-original-csv",
        default="data/front_unlabeled_default_none_label.csv",
        help="Original front unlabeled CSV used for reconstruction.",
    )
    parser.add_argument(
        "--back-original-csv",
        default="data/back_unlabeled_default_none_label.csv",
        help="Original back unlabeled CSV used for reconstruction.",
    )
    parser.add_argument(
        "--front-output-dir",
        default="results/heuristic_front_unlabeled",
        help="Output directory for front heuristic runs.",
    )
    parser.add_argument(
        "--back-output-dir",
        default="results/heuristic_back_unlabeled",
        help="Output directory for back heuristic runs.",
    )
    parser.add_argument(
        "--front-reconstructed-csv",
        default="results/reconstructed_heuristic_front_unlabeled.csv",
        help="Classifier-ready reconstructed front CSV.",
    )
    parser.add_argument(
        "--back-reconstructed-csv",
        default="results/reconstructed_heuristic_back_unlabeled.csv",
        help="Classifier-ready reconstructed back CSV.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for heuristic detection.")
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=1000.0,
        help="Activity threshold for heuristic detection.",
    )
    parser.add_argument(
        "--hazard-ratio",
        type=float,
        default=0.7,
        help="Hazard ratio threshold for heuristic detection.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    front_result = test_heuristic(
        config_path=args.front_config,
        test_sequences_file=None,
        output_dir=args.front_output_dir,
        verbose=args.verbose,
        fps=args.fps,
        activity_threshold=args.activity_threshold,
        hazard_ratio_threshold=args.hazard_ratio,
        export_frame_csv=True,
    )
    front_run_dir = Path(args.front_output_dir) / f"{front_result['model']}_{front_result['timestamp']}"
    reconstruct(
        original_csv=Path(args.front_original_csv),
        results_dir=front_run_dir,
        mode="unlabeled",
        output_path=Path(args.front_reconstructed_csv),
    )

    back_result = test_heuristic(
        config_path=args.back_config,
        test_sequences_file=None,
        output_dir=args.back_output_dir,
        verbose=args.verbose,
        fps=args.fps,
        activity_threshold=args.activity_threshold,
        hazard_ratio_threshold=args.hazard_ratio,
        export_frame_csv=True,
    )
    back_run_dir = Path(args.back_output_dir) / f"{back_result['model']}_{back_result['timestamp']}"
    reconstruct(
        original_csv=Path(args.back_original_csv),
        results_dir=back_run_dir,
        mode="unlabeled",
        output_path=Path(args.back_reconstructed_csv),
    )


if __name__ == "__main__":
    main()
