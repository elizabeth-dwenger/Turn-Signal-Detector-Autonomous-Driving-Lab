"""
Analysis of model disagreements and error patterns.
Identifies which frames models disagree on and attempts to categorize why.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging
from collections import Counter, defaultdict
import json
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class DisagreementAnalyzer:
    """Analyze disagreements between model predictions"""
    
    def __init__(self):
        pass
    
    def compute_disagreement_metrics(self, frame_predictions: Dict,
                                      ensemble_df: pd.DataFrame = None) -> Dict:
        """
        Compute disagreement metrics across all frames.
        
        Args:
            frame_predictions: Dict[(seq_id, frame_id) -> list of FramePrediction]
            ensemble_df: Optional ensemble_df with voting distributions
        
        Returns:
            Dict with disagreement statistics
        """
        metrics = {
            "total_frames": len(frame_predictions),
            "high_disagreement_frames": 0,
            "unanimous_frames": 0,
            "entropy_distribution": [],
            "pattern_frequency": defaultdict(int),
        }
        
        for (seq_id, frame_id), predictions in frame_predictions.items():
            if not predictions:
                continue
            
            # Compute voting distribution
            label_counts = Counter([p.label for p in predictions])
            
            # Entropy of distribution
            counts = np.array(list(label_counts.values()))
            probs = counts / counts.sum()
            frame_entropy = entropy(probs)
            
            metrics["entropy_distribution"].append(frame_entropy)
            
            # Pattern representation
            pattern = tuple(sorted(label_counts.items()))
            metrics["pattern_frequency"][str(pattern)] += 1
            
            # High disagreement = entropy > 0.5
            if frame_entropy > 0.5:
                metrics["high_disagreement_frames"] += 1
            
            # Unanimous
            if len(label_counts) == 1:
                metrics["unanimous_frames"] += 1
        
        # Summary statistics
        if metrics["entropy_distribution"]:
            metrics["mean_entropy"] = float(np.mean(metrics["entropy_distribution"]))
            metrics["std_entropy"] = float(np.std(metrics["entropy_distribution"]))
            metrics["max_entropy"] = float(np.max(metrics["entropy_distribution"]))
            metrics["min_entropy"] = float(np.min(metrics["entropy_distribution"]))
        
        # Convert defaultdict to regular dict
        metrics["pattern_frequency"] = dict(metrics["pattern_frequency"])
        
        return metrics
    
    def identify_disagreement_frames(self, frame_predictions: Dict,
                                      entropy_threshold: float = 0.5,
                                      ground_truth: Dict[Tuple, str] = None) -> List[Dict]:
        """
        Identify frames where models disagree above threshold.
        
        Args:
            frame_predictions: Dict[(seq_id, frame_id) -> list of FramePrediction]
            entropy_threshold: Disagreement threshold
            ground_truth: Optional ground truth for error classification
        
        Returns:
            List of disagreement records
        """
        disagreements = []
        
        for (seq_id, frame_id), predictions in frame_predictions.items():
            if not predictions:
                continue
            
            # Voting distribution
            label_counts = Counter([p.label for p in predictions])
            counts = np.array(list(label_counts.values()))
            probs = counts / counts.sum()
            frame_entropy = entropy(probs)
            
            # Filter by threshold
            if frame_entropy < entropy_threshold:
                continue
            
            # Construct prediction dict
            pred_dict = {}
            for pred in predictions:
                pred_dict[pred.model_name] = {
                    "label": pred.label,
                    "confidence": pred.confidence,
                }
            
            record = {
                "sequence_id": seq_id,
                "frame_id": frame_id,
                "entropy": float(frame_entropy),
                "voting_distribution": dict(label_counts),
                "predictions": pred_dict,
            }
            
            # Add ground truth if available
            key = (seq_id, frame_id)
            if ground_truth and key in ground_truth:
                record["ground_truth"] = ground_truth[key]
            
            disagreements.append(record)
        
        logger.info(f"Found {len(disagreements)} high-disagreement frames")
        return disagreements
    
    def categorize_patterns(self, disagreement_records: List[Dict]) -> Dict:
        """
        Categorize disagreement patterns.
        
        Args:
            disagreement_records: List of disagreement records from identify_disagreement_frames
        
        Returns:
            Dict mapping pattern -> list of frame records
        """
        pattern_groups = defaultdict(list)
        
        for record in disagreement_records:
            # Create pattern signature
            dist = record["voting_distribution"]
            pattern_sig = tuple(sorted(dist.items()))
            
            # Add ground truth to pattern if available
            if "ground_truth" in record:
                pattern_key = (pattern_sig, record["ground_truth"])
            else:
                pattern_key = (pattern_sig, "unknown")
            
            pattern_groups[pattern_key].append(record)
        
        # Compute summaries per pattern
        pattern_summary = {}
        for pattern_key, records in pattern_groups.items():
            pattern_sig, gt_label = pattern_key
            
            pattern_summary[str(pattern_sig)] = {
                "pattern": dict(pattern_sig),
                "ground_truth": gt_label,
                "count": len(records),
                "example_frames": [
                    f"{r['sequence_id']}:{r['frame_id']}" for r in records[:3]
                ],
                "mean_entropy": float(np.mean([r["entropy"] for r in records])),
            }
        
        return pattern_summary
    
    def analyze_model_errors(self, frame_predictions: Dict,
                              ground_truth: Dict[Tuple, str]) -> Dict[str, Dict]:
        """
        Analyze per-model errors.
        
        Args:
            frame_predictions: Dict[(seq_id, frame_id) -> list of FramePrediction]
            ground_truth: Dict[(seq_id, frame_id) -> label]
        
        Returns:
            Dict[model_name -> error_profile]
        """
        model_profiles = {}
        
        # Collect all unique models
        all_models = set()
        for predictions in frame_predictions.values():
            for pred in predictions:
                all_models.add(pred.model_name)
        
        for model_name in all_models:
            # Collect predictions for this model
            model_preds = {}
            for (seq_id, frame_id), predictions in frame_predictions.items():
                for pred in predictions:
                    if pred.model_name == model_name:
                        model_preds[(seq_id, frame_id)] = pred
            
            # Compute error profile
            errors = {
                "false_positives": [],
                "false_negatives": [],
                "correct": [],
                "confusion": defaultdict(int),
            }
            
            for key, pred in model_preds.items():
                if key not in ground_truth:
                    continue
                
                gt_label = ground_truth[key]
                pred_label = pred.label
                
                if pred_label == gt_label:
                    errors["correct"].append(key)
                elif pred_label == "none" and gt_label != "none":
                    errors["false_negatives"].append(key)
                elif pred_label != "none" and gt_label == "none":
                    errors["false_positives"].append(key)
                else:
                    errors["confusion"][f"{gt_label}->{pred_label}"] += 1
            
            # Compute metrics
            num_predictions = len(model_preds)
            num_correct = len(errors["correct"])
            accuracy = num_correct / num_predictions if num_predictions > 0 else 0
            
            model_profiles[model_name] = {
                "total_predictions": num_predictions,
                "correct": len(errors["correct"]),
                "accuracy": float(accuracy),
                "false_positives": len(errors["false_positives"]),
                "false_negatives": len(errors["false_negatives"]),
                "confusion_errors": dict(errors["confusion"]),
                "example_fp_frames": [str(k) for k in errors["false_positives"][:3]],
                "example_fn_frames": [str(k) for k in errors["false_negatives"][:3]],
            }
        
        return model_profiles
    
    def analyze_error_correlation(self, frame_predictions: Dict,
                                   ground_truth: Dict[Tuple, str]) -> Dict:
        """
        Analyze correlations in model errors (are they making same mistakes?).
        
        Args:
            frame_predictions: Dict of predictions
            ground_truth: Ground truth labels
        
        Returns:
            Dict with correlation analysis
        """
        model_profiles = self.analyze_model_errors(frame_predictions, ground_truth)
        
        # Collect error frames per model
        all_models = list(model_profiles.keys())
        error_sets = {}
        
        for model_name in all_models:
            profile = model_profiles[model_name]
            errors = set()
            
            for frame_str in profile.get("example_fp_frames", []):
                errors.add(frame_str)
            for frame_str in profile.get("example_fn_frames", []):
                errors.add(frame_str)
            
            error_sets[model_name] = errors
        
        # Pairwise error correlation
        correlations = {}
        for i, model_a in enumerate(all_models):
            for model_b in all_models[i+1:]:
                overlap = len(error_sets[model_a] & error_sets[model_b])
                union = len(error_sets[model_a] | error_sets[model_b])
                
                # Jaccard similarity
                similarity = overlap / union if union > 0 else 0
                
                correlations[f"{model_a}-{model_b}"] = {
                    "shared_errors": overlap,
                    "jaccard_similarity": float(similarity),
                }
        
        return {
            "model_profiles": model_profiles,
            "error_correlations": correlations,
        }
    
    def export_disagreement_log(self, disagreement_records: List[Dict],
                                 output_path: str) -> None:
        """
        Export disagreement log to JSON file.
        
        Args:
            disagreement_records: List of disagreement records
            output_path: Path to save JSON file
        """
        # Convert numpy/defaultdict types for JSON serialization
        records = []
        for record in disagreement_records:
            clean_record = {
                "sequence_id": record["sequence_id"],
                "frame_id": record["frame_id"],
                "entropy": float(record["entropy"]),
                "voting_distribution": dict(record["voting_distribution"]),
                "predictions": record["predictions"],
            }
            
            if "ground_truth" in record:
                clean_record["ground_truth"] = record["ground_truth"]
            
            records.append(clean_record)
        
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)
        
        logger.info(f"Exported {len(records)} disagreement records to {output_path}")
