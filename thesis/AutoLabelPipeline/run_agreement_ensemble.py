#!/usr/bin/env python
"""
CLI for agreement-based filtering between two models.

Takes predictions from a VLM model (cosmos_reason2_video) and the heuristic
detector, compares per-frame labels, and keeps only frames where both agree.

Example usage
-------------
# Basic
python run_agreement_ensemble.py \
    --vlm-results results/cosmos_reason2_video \
    --heuristic-results prompt_comparison/heuristic_20260215_120000/results.json \
    --output-dir agreement_experiments/

# With ground truth
python run_agreement_ensemble.py \
    --vlm-results results/cosmos_reason2_video \
    --heuristic-results prompt_comparison/heuristic_20260215_120000/results.json \
    --ground-truth data/tracking_data.csv \
    --output-dir agreement_experiments/ \
    --verbose
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ensemble.agreement_filter import (
    AgreementFilter,
    HeuristicResultLoader,
    VLMResultLoader,
)
from ensemble.loader import EnsembleLoader, FramePredictionDataset


def setup_logging(log_file=None, verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Agreement-based ensemble: keep only frames where "
            "cosmos_reason2_video and heuristic_model agree."
        )
    )

    # --- Input sources --------------------------------------------------
    parser.add_argument(
        "--vlm-results",
        type=Path,
        default=Path("results/cosmos_reason2_video"),
        help=(
            "Path to VLM model results directory "
            "(e.g. results/cosmos_reason2_video). "
            "Accepts the model root or the direct run directory."
        ),
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="cosmos_reason2_video",
        help="Name for the VLM model (default: cosmos_reason2_video)",
    )
    parser.add_argument(
        "--heuristic-results",
        type=Path,
        required=True,
        help=(
            "Path to heuristic results.json file, OR the directory "
            "containing it (e.g. prompt_comparison/heuristic_20260215_120000)."
        ),
    )
    parser.add_argument(
        "--heuristic-model-name",
        type=str,
        default="heuristic_model",
        help="Name for the heuristic model (default: heuristic_model)",
    )

    # --- Ground truth ---------------------------------------------------
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to ground truth CSV (must have sequence_id, frame_id, label/true_label)",
    )

    # --- Output ---------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agreement_experiments"),
        help="Directory to save results (default: agreement_experiments/)",
    )

    # --- Options --------------------------------------------------------
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (overrides CLI flags)",
    )
    parser.add_argument(
        "--no-transform-ids",
        action="store_true",
        help="Disable automatic sequence_id transformation for VLM predictions",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # --- Config override ------------------------------------------------
    if args.config and args.config.exists():
        cfg = load_config(args.config)
        args.vlm_results = Path(cfg.get("vlm_results", str(args.vlm_results)))
        args.vlm_model_name = cfg.get("vlm_model_name", args.vlm_model_name)
        args.heuristic_results = Path(cfg.get("heuristic_results", str(args.heuristic_results)))
        args.heuristic_model_name = cfg.get("heuristic_model_name", args.heuristic_model_name)
        args.ground_truth = Path(cfg["ground_truth"]) if cfg.get("ground_truth") else args.ground_truth
        args.output_dir = Path(cfg.get("output_dir", str(args.output_dir)))
        args.no_transform_ids = cfg.get("no_transform_ids", args.no_transform_ids)
        args.verbose = cfg.get("verbose", args.verbose)

    # --- Logging --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / f"agreement_{timestamp}.log"
    setup_logging(log_file, args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("AGREEMENT-BASED ENSEMBLE FILTERING")
    logger.info("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load VLM predictions
    # ---------------------------------------------------------------
    logger.info(f"Loading VLM predictions from {args.vlm_results}")
    vlm_loader = VLMResultLoader(
        model_name=args.vlm_model_name,
        transform_sequence_ids=not args.no_transform_ids,
    )
    try:
        dataset_vlm = vlm_loader.load(args.vlm_results)
    except FileNotFoundError as e:
        logger.error(f"VLM results not found: {e}")
        return 1

    logger.info(
        f"Loaded {len(dataset_vlm.predictions)} VLM frame predictions "
        f"across {len(set(k[0] for k in dataset_vlm.predictions))} sequences"
    )

    # ---------------------------------------------------------------
    # 2. Load heuristic predictions
    # ---------------------------------------------------------------
    logger.info(f"Loading heuristic predictions from {args.heuristic_results}")
    heuristic_loader = HeuristicResultLoader(model_name=args.heuristic_model_name)
    try:
        dataset_heuristic = heuristic_loader.load(args.heuristic_results)
    except FileNotFoundError as e:
        logger.error(f"Heuristic results not found: {e}")
        return 1

    logger.info(
        f"Loaded {len(dataset_heuristic.predictions)} heuristic frame predictions "
        f"across {len(set(k[0] for k in dataset_heuristic.predictions))} sequences"
    )

    # ---------------------------------------------------------------
    # 3. Load ground truth (optional)
    # ---------------------------------------------------------------
    ground_truth = {}
    if args.ground_truth and args.ground_truth.exists():
        logger.info(f"Loading ground truth from {args.ground_truth}")
        gt_loader = EnsembleLoader(transform_sequence_ids=False)
        gt_dataset = FramePredictionDataset()
        gt_loader.load_ground_truth(args.ground_truth, gt_dataset)
        ground_truth = gt_dataset.ground_truth
        logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    elif args.ground_truth:
        logger.warning(f"Ground truth file not found: {args.ground_truth}")

    # ---------------------------------------------------------------
    # 4. Run agreement filter
    # ---------------------------------------------------------------
    exp_dir = args.output_dir / f"agreement_{timestamp}"
    filter_obj = AgreementFilter(
        model_a_name=args.vlm_model_name,
        model_b_name=args.heuristic_model_name,
    )

    result = filter_obj.run(
        dataset_a=dataset_vlm,
        dataset_b=dataset_heuristic,
        output_dir=exp_dir,
        ground_truth=ground_truth if ground_truth else None,
    )

    # ---------------------------------------------------------------
    # 5. Final summary
    # ---------------------------------------------------------------
    report = result["report"]
    metrics = result.get("metrics", {})

    logger.info("")
    logger.info("RESULTS SAVED TO: %s", exp_dir)
    for name, path in result["output_paths"].items():
        logger.info("  %s -> %s", name, path)

    if metrics:
        logger.info("")
        logger.info("EVALUATION (agreed frames only):")
        logger.info("  Macro F1 : %.3f", metrics.get("frame_macro_f1", 0))
        logger.info("  Accuracy : %.3f", metrics.get("accuracy", 0))
        for label in ["left", "right", "hazard", "none"]:
            logger.info(
                "  F1 %-7s: %.3f", label, metrics.get(f"f1_{label}", 0)
            )

    logger.info("")
    logger.info("Agreement-based filtering complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
