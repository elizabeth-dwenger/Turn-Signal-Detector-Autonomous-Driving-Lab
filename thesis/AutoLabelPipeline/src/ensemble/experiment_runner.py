"""
Ensemble experiment runner.
Orchestrates end-to-end ensemble workflows: loading, aggregation, evaluation, disagreement analysis.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ensemble.loader import EnsembleLoader, FramePredictionDataset
from ensemble.aggregator import (
    MajorityVoter,
    EnsembleAggregator,
)
from ensemble.evaluator import EnsembleEvaluator
from ensemble.disagreement import DisagreementAnalyzer

logger = logging.getLogger(__name__)


class EnsembleExperimentRunner:
    """Run ensemble experiments end-to-end"""
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        """
        Args:
            output_dir: Directory to save results
            verbose: Enable verbose logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = EnsembleLoader(
            label_harmonization=True,
            calibrate_confidence=False,
            handle_missing_frames="exclude"
        )
        
        self.evaluator = EnsembleEvaluator()
        self.disagreement_analyzer = DisagreementAnalyzer()
        
        self.verbose = verbose
    
    def select_top_models(self, results_root: Path, n_top: int = 3,
                         model_f1_scores: Dict[str, float] = None,
                         diversity_mode: str = "auto") -> List[Tuple[Path, str, float]]:
        """
        Select top N models by F1 score.
        
        Args:
            results_root: Root directory containing model results
            n_top: Number of top models to select
            model_f1_scores: Optional dict mapping model_name -> f1_score (e.g., from user input)
            diversity_mode: "auto" (top N by F1) or "manual" (user-provided)
        
        Returns:
            List of (results_dir, model_name, f1_score)
        """
        if diversity_mode == "manual" and model_f1_scores:
            # Manual selection from provided scores
            selected = sorted(
                model_f1_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_top]
            
            results = []
            for model_name, f1 in selected:
                # Find the actual result directory
                result_dir = self._find_result_directory(results_root, model_name)
                if result_dir:
                    results.append((result_dir, model_name, f1))
                else:
                    logger.warning(f"Could not find results directory for {model_name}")
            
            return results
        
        # Auto selection: scan results_root and use provided F1 scores
        if not model_f1_scores:
            logger.warning(
                "No F1 scores provided; scanning for results directories. "
                "Provide model_f1_scores for better selection."
            )
            model_f1_scores = {}
        
        # Create list of (dir, model_name, f1_score)
        candidates = []
        for model_name, f1 in model_f1_scores.items():
            # Find the actual result directory
            result_dir = self._find_result_directory(results_root, model_name)
            if result_dir:
                candidates.append((result_dir, model_name, f1))
            else:
                logger.warning(f"Could not find results directory for {model_name}")
        
        # Sort by F1 and select top N
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = candidates[:n_top]
        
        logger.info(f"Selected top {len(selected)} models:")
        for result_dir, model_name, f1 in selected:
            logger.info(f"  - {model_name}: F1={f1:.3f}")
        
        return selected
    
    def _find_result_directory(self, results_root: Path, model_name: str) -> Optional[Path]:
        """
        Find the actual result directory for a model.
        
        Handles different structures:
        - results/model_name/test_runs/timestamp/  (standard pipeline structure)
        - results/model_name/experiment_name/      (alternative structure)
        - results/model_name/                      (flat structure)
        
        Args:
            results_root: Root directory containing model results
            model_name: Model name like "qwen3_vl_single/exp_5"
        
        Returns:
            Path to the directory containing CSV/JSON prediction files, or None if not found
        """
        # Parse model_name: "qwen3_vl_single/exp_5" -> base="qwen3_vl_single", exp="exp_5"
        parts = model_name.split("/")
        if len(parts) == 2:
            model_base, exp_id = parts
        else:
            model_base = model_name
            exp_id = None
        
        # Try standard pipeline structure: results/model_base/test_runs/
        test_runs_dir = results_root / model_base / "test_runs"
        if test_runs_dir.exists():
            # Find the most recent timestamped directory
            timestamp_dirs = sorted([d for d in test_runs_dir.iterdir() if d.is_dir()], 
                                   key=lambda p: p.name, reverse=True)
            if timestamp_dirs:
                logger.debug(f"Found result directory: {timestamp_dirs[0]}")
                return timestamp_dirs[0]
        
        # Try alternative: results/model_base/exp_id/
        if exp_id:
            exp_dir = results_root / model_base / exp_id
            if exp_dir.exists() and self._has_prediction_files(exp_dir):
                logger.debug(f"Found result directory: {exp_dir}")
                return exp_dir
        
        # Try flat structure: results/model_base/
        model_dir = results_root / model_base
        if model_dir.exists() and self._has_prediction_files(model_dir):
            logger.debug(f"Found result directory: {model_dir}")
            return model_dir
        
        # Try with underscore: results/model_base_exp_id/
        if exp_id:
            combined_dir = results_root / f"{model_base}_{exp_id}"
            if combined_dir.exists() and self._has_prediction_files(combined_dir):
                logger.debug(f"Found result directory: {combined_dir}")
                return combined_dir
        
        return None
    
    def _has_prediction_files(self, directory: Path) -> bool:
        """Check if directory contains prediction CSV or JSON files"""
        csv_files = list(directory.glob("*.csv"))
        json_files = list(directory.glob("*.json"))
        # Exclude metadata files
        prediction_files = [f for f in csv_files + json_files 
                          if not any(x in f.name for x in ['dataset_summary', 'pipeline_report', 
                                                            'evaluation_metrics', 'evaluation_per_sequence',
                                                            'review_queue'])]
        return len(prediction_files) > 0
    
    def run_single_experiment(self, dataset: FramePredictionDataset,
                              voter, exp_name: str,
                              ground_truth: Optional[Dict] = None) -> Dict:
        """
        Run a single ensemble experiment (load, aggregate, evaluate).
        
        Args:
            dataset: FramePredictionDataset
            voter: Voter instance
            exp_name: Name for this experiment
            ground_truth: Optional ground truth for evaluation
        
        Returns:
            Dict with results
        """
        logger.info(f"Running experiment: {exp_name}")
        
        # Aggregate predictions
        aggregator = EnsembleAggregator(voter)
        ensemble_df = aggregator.aggregate(dataset.predictions, verbose=self.verbose)
        
        # Save ensemble predictions
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        ensemble_csv = exp_dir / "ensemble_predictions.csv"
        ensemble_df.to_csv(ensemble_csv, index=False)
        logger.info(f"Saved ensemble predictions to {ensemble_csv}")
        
        # Evaluate
        metrics = {}
        if ground_truth and dataset.ground_truth:
            # Use provided or loaded ground truth
            gt = ground_truth or dataset.ground_truth
            
            # Debug: Show sample keys
            ensemble_keys = [(row["sequence_id"], row["frame_id"]) for _, row in ensemble_df.head(3).iterrows()]
            gt_keys_sample = list(gt.keys())[:3]
            logger.debug(f"Sample ensemble keys: {ensemble_keys}")
            logger.debug(f"Sample ground truth keys: {gt_keys_sample}")
            logger.debug(f"Total ensemble frames: {len(ensemble_df)}, Total ground truth frames: {len(gt)}")
            
            # Prepare for evaluation
            y_true = []
            y_pred = []
            
            for _, row in ensemble_df.iterrows():
                key = (row["sequence_id"], row["frame_id"])
                if key in gt:
                    y_true.append(gt[key])
                    y_pred.append(row["ensemble_label"])
            
            if y_true:
                metrics = self.evaluator.compute_frame_metrics(
                    y_true, y_pred, per_label=True
                )
                logger.info(f"Metrics: frame_macro_f1={metrics['frame_macro_f1']:.3f}")
                logger.info(f"Evaluated {len(y_true)} frames with ground truth")
            else:
                logger.warning(
                    f"No matching frames between ensemble predictions and ground truth. "
                    f"Check that sequence_id and frame_id formats match."
                )
        
        else:
            logger.warning("No ground truth provided; skipping evaluation")
        
        # Disagreement analysis
        disagreement_metrics = self.disagreement_analyzer.compute_disagreement_metrics(
            dataset.predictions, ensemble_df
        )
        
        # Save disagreement log
        disagreements = self.disagreement_analyzer.identify_disagreement_frames(
            dataset.predictions, entropy_threshold=0.5
        )
        
        disagreement_log = exp_dir / "disagreement_log.json"
        self.disagreement_analyzer.export_disagreement_log(disagreements, disagreement_log)
        
        # Combine results
        results = {
            "experiment_name": exp_name,
            "timestamp": datetime.now().isoformat(),
            "ensemble_predictions_csv": str(ensemble_csv),
            "metrics": metrics,
            "disagreement_metrics": disagreement_metrics,
            "disagreement_log_json": str(disagreement_log),
        }
        
        # Save results metadata
        metadata_json = exp_dir / "metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def generate_comparison_report(self, individual_datasets: Dict[str, FramePredictionDataset],
                                    ensemble_results: Dict,
                                    ground_truth: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate comparison table: individual models vs. ensemble.
        
        Args:
            individual_datasets: Dict[model_name -> FramePredictionDataset]
            ensemble_results: Results dict from run_single_experiment
            ground_truth: Ground truth labels
        
        Returns:
            Comparison DataFrame
        """
        gt = ground_truth
        if not gt:
            # Try to use ground truth from any dataset
            for dataset in individual_datasets.values():
                if dataset.ground_truth:
                    gt = dataset.ground_truth
                    break
        
        if not gt:
            logger.warning("No ground truth found; cannot generate comparison")
            return pd.DataFrame()
        
        # Prepare individual predictions dataframes
        individual_dfs = {}
        for model_name, dataset in individual_datasets.items():
            # Convert to prediction dataframe
            rows = []
            for (seq_id, frame_id), preds in dataset.predictions.items():
                for pred in preds:
                    rows.append({
                        "sequence_id": seq_id,
                        "frame_id": frame_id,
                        "label": pred.label,
                        "confidence": pred.confidence,
                    })
            individual_dfs[model_name] = pd.DataFrame(rows)
        
        # Load ensemble predictions if available
        ensemble_csv = ensemble_results.get("ensemble_predictions_csv")
        ensemble_df = None
        if ensemble_csv and Path(ensemble_csv).exists():
            ensemble_df = pd.read_csv(ensemble_csv)
        
        # Compare
        comparison = self.evaluator.compare_models(
            individual_dfs, ensemble_df, gt
        )
        
        # Save comparison
        comparison_csv = self.output_dir / "model_comparison.csv"
        comparison.to_csv(comparison_csv, index=False)
        logger.info(f"Saved comparison to {comparison_csv}")
        
        return comparison
    
    def generate_summary(self, all_experiments: List[Dict]) -> str:
        """
        Generate text summary of all experiments.
        
        Args:
            all_experiments: List of experiment result dicts
        
        Returns:
            Summary text
        """
        summary = []
        summary.append("=" * 80)
        summary.append("ENSEMBLE EXPERIMENT SUMMARY")
        summary.append(f"Generated: {datetime.now().isoformat()}")
        summary.append("=" * 80)
        summary.append("")
        
        for exp in all_experiments:
            summary.append(f"Experiment: {exp['experiment_name']}")
            summary.append("-" * 40)
            
            # Show metrics if available
            if exp.get("metrics"):
                metrics = exp["metrics"]
                summary.append(f"  Frame Macro F1: {metrics.get('frame_macro_f1', 0.0):.3f}")
                summary.append(f"  Accuracy:       {metrics.get('accuracy', 0.0):.3f}")
                summary.append("")
                summary.append("  Per-Label F1:")
                summary.append(f"    left:    {metrics.get('f1_left', 0.0):.3f}")
                summary.append(f"    right:   {metrics.get('f1_right', 0.0):.3f}")
                summary.append(f"    hazard:  {metrics.get('f1_hazard', 0.0):.3f}")
                summary.append(f"    none:    {metrics.get('f1_none', 0.0):.3f}")
            else:
                summary.append("  ⚠ No evaluation metrics computed")
                summary.append("  Reason: No matching frames between predictions and ground truth")
                summary.append("  Check that sequence_id and frame_id formats match between:")
                summary.append("    - Ensemble predictions (saved to CSV)")
                summary.append("    - Ground truth CSV file")
            
            # Show disagreement metrics
            if exp.get("disagreement_metrics"):
                summary.append("")
                dmets = exp["disagreement_metrics"]
                summary.append("  Disagreement Analysis:")
                summary.append(f"    Total frames:             {dmets.get('total_frames', 0)}")
                summary.append(f"    Unanimous frames:         {dmets.get('unanimous_frames', 0)}")
                summary.append(f"    High-disagreement frames: {dmets.get('high_disagreement_frames', 0)}")
                if 'mean_entropy' in dmets:
                    summary.append(f"    Mean entropy:             {dmets.get('mean_entropy', 0.0):.3f}")
            
            summary.append("")
        
        summary.append("=" * 80)
        return "\n".join(summary)
