"""
Evaluation of ensemble predictions.
Computes F1, per-label metrics, confusion matrix, and parameter sweeps.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Evaluate ensemble predictions against ground truth"""
    
    VALID_LABELS = ["left", "right", "hazard", "none"]
    
    def __init__(self, label_order: List[str] = None):
        """
        Args:
            label_order: Order of labels for metrics. Defaults to standard order.
        """
        self.label_order = label_order or self.VALID_LABELS
    
    def compute_frame_metrics(self, y_true: List[str], y_pred: List[str],
                              per_label: bool = True) -> Dict:
        """
        Compute frame-level metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            per_label: If True, return per-label metrics
        
        Returns:
            Dict with frame_macro_f1, per_label scores, etc.
        """
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute macro F1 (unweighted average across classes)
        frame_macro_f1 = f1_score(
            y_true, y_pred, labels=self.label_order, average="macro", zero_division=0
        )
        
        metrics = {
            "frame_macro_f1": float(frame_macro_f1),
        }
        
        # Per-label metrics
        if per_label:
            per_label_metrics = {}
            
            for label in self.label_order:
                # Binary classification: this label vs. rest
                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)
                
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                support = np.sum(y_true_binary)
                
                per_label_metrics[f"f1_{label}"] = float(f1)
                per_label_metrics[f"precision_{label}"] = float(precision)
                per_label_metrics[f"recall_{label}"] = float(recall)
                per_label_metrics[f"support_{label}"] = int(support)
            
            metrics.update(per_label_metrics)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.label_order)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)
        metrics["accuracy"] = float(accuracy)
        
        return metrics
    
    def compare_models(self, individual_results: Dict[str, pd.DataFrame],
                       ensemble_result: pd.DataFrame,
                       ground_truth: Dict[Tuple, str]) -> pd.DataFrame:
        """
        Compare individual models against ensemble.
        
        Args:
            individual_results: Dict[model_name -> DataFrame with predictions]
            ensemble_result: DataFrame with ensemble predictions
            ground_truth: Dict[(sequence_id, frame_id) -> label]
        
        Returns:
            Comparison DataFrame
        """
        comparison_rows = []
        
        # Evaluate individual models
        for model_name, df in individual_results.items():
            y_true = []
            y_pred = []
            
            for _, row in df.iterrows():
                key = (row["sequence_id"], row["frame_id"])
                if key in ground_truth:
                    y_true.append(ground_truth[key])
                    y_pred.append(row["label"])
            
            if y_true:
                metrics = self.compute_frame_metrics(y_true, y_pred, per_label=True)
                
                comparison_rows.append({
                    "model": model_name,
                    "type": "individual",
                    "num_frames": len(y_true),
                    **metrics,
                })
        
        # Evaluate ensemble
        y_true = []
        y_pred = []
        
        for _, row in ensemble_result.iterrows():
            key = (row["sequence_id"], row["frame_id"])
            if key in ground_truth:
                y_true.append(ground_truth[key])
                y_pred.append(row["ensemble_label"])
        
        if y_true:
            metrics = self.compute_frame_metrics(y_true, y_pred, per_label=True)
            
            comparison_rows.append({
                "model": "ensemble",
                "type": "ensemble",
                "num_frames": len(y_true),
                **metrics,
            })
        
        return pd.DataFrame(comparison_rows)
