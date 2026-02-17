#!/usr/bin/env python
"""
CLI for ensemble prediction experiments.
Runs end-to-end ensemble workflows from saved model predictions.
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ensemble.loader import EnsembleLoader
from ensemble.aggregator import (
    MajorityVoter,
    EnsembleAggregator,
)
from ensemble.experiment_runner import EnsembleExperimentRunner


def setup_logging(log_file=None, verbose=False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble experiments on saved model predictions"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to ensemble config YAML file"
    )
    
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing model results (default: results/)"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Path to ground truth CSV file (sequence_id, frame_id, label)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ensemble_experiments"),
        help="Directory to save results (default: ensemble_experiments/)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to ensemble (space-separated list with F1 scores, e.g., 'qwen3_vl_single:0.574')"
    )
    
    parser.add_argument(
        "--n-top-models",
        type=int,
        default=3,
        help="Number of top models to select (used if --models not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output_dir / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file, args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting ensemble experiment runner")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize runner
    runner = EnsembleExperimentRunner(args.output_dir, verbose=args.verbose)
    
    # Load ground truth if provided
    ground_truth = {}
    if args.ground_truth and args.ground_truth.exists():
        loader = EnsembleLoader()
        from ensemble.loader import FramePredictionDataset
        dummy_dataset = FramePredictionDataset()
        loader.load_ground_truth(args.ground_truth, dummy_dataset)
        ground_truth = dummy_dataset.ground_truth
        logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Load model predictions
    loader = EnsembleLoader()
    
    # Determine which models to load
    if args.models:
        # Parse model specifications: "model_name:f1_score"
        model_specs = {}
        for spec in args.models:
            parts = spec.split(":")
            if len(parts) == 2:
                model_name, f1_str = parts
                model_specs[model_name] = float(f1_str)
            else:
                model_name = spec
                model_specs[model_name] = 0.5  # Default F1
        
        selected_models = runner.select_top_models(
            args.results_root,
            n_top=len(model_specs),
            model_f1_scores=model_specs,
            diversity_mode="manual"
        )
    else:
        # Use built-in F1 scores from the problem statement
        default_f1_scores = {
            "qwen3_vl_single/exp_5": 0.574,
            "cosmos_reason2_video/exp_2": 0.465,
            "cosmos_reason1_video/exp_5": 0.538,
            "qwen25_vl_video/exp_5": 0.527,
            "qwen3_vl_video/exp_1": 0.405,
        }
        
        selected_models = runner.select_top_models(
            args.results_root,
            n_top=args.n_top_models,
            model_f1_scores=default_f1_scores,
            diversity_mode="auto"
        )
    
    logger.info(f"Selected {len(selected_models)} models for ensemble")
    
    # Load predictions from selected models
    from ensemble.loader import FramePredictionDataset
    all_datasets = []
    
    for result_dir, model_name, f1_score in selected_models:
        if not result_dir.exists():
            logger.warning(f"Results directory not found: {result_dir}")
            continue
        
        logger.info(f"Loading predictions from {model_name} (F1={f1_score:.3f})")
        
        dataset = loader.load_from_results_dir(
            result_dir,
            model_name=model_name,
            model_config={"mode": "single" if "single" in model_name else "video"}
        )
        
        all_datasets.append(dataset)
    
    if not all_datasets:
        logger.error("No model predictions loaded. Exiting.")
        return 1
    
    # Merge datasets
    merged_dataset = loader.merge_datasets(*all_datasets)
    
    # Load ground truth into dataset
    if ground_truth:
        merged_dataset.ground_truth = ground_truth
    
    # Validate dataset
    validation = loader.validate_dataset(merged_dataset)
    logger.info(f"Dataset validation: {validation['num_frames']} frames, {validation['num_models']} models")
    
    # Run experiments
    all_results = []
    
    logger.info("Running: Majority Vote (MV)")
    voter = MajorityVoter(tie_break_method="confidence")
    results = runner.run_single_experiment(
        merged_dataset, voter, "exp_mv", ground_truth=ground_truth
    )
    all_results.append(results)
    
    # Generate summary report
    summary = runner.generate_summary(all_results)
    print("\n" + summary)
    
    # Save summary
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    
    logger.info(f"Ensemble experiments complete. Results saved to {args.output_dir}")
    logger.info(f"Summary saved to {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
