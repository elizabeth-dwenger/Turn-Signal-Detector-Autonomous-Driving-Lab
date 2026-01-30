"""
Metrics report generation for model performance analysis.
Generates comprehensive statistics and quality metrics.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter
import numpy as np
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class MetricsReporter:
    """
    Generates comprehensive metrics reports from predictions and processing data.
    """
    
    def __init__(self, model_name: str, dataset_name: str):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.start_time = time.time()
    
    def generate_report(self,
                       predictions_by_sequence: Dict[str, List[Dict]],
                       quality_reports: Optional[Dict[str, Dict]] = None,
                       model_metrics: Optional[Dict] = None,
                       processing_stats: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.
        """
        report = {
            'metadata': {
                'model': self.model_name,
                'dataset': self.dataset_name,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': time.time() - self.start_time,
            },
            'dataset_stats': self._compute_dataset_stats(predictions_by_sequence),
            'label_distribution': self._compute_label_distribution(predictions_by_sequence),
            'confidence_stats': self._compute_confidence_stats(predictions_by_sequence),
            'quality_metrics': self._compute_quality_metrics(predictions_by_sequence, quality_reports),
            'temporal_metrics': self._compute_temporal_metrics(predictions_by_sequence),
        }
        
        # Add model-specific metrics if provided
        if model_metrics:
            report['model_specific'] = model_metrics
        
        # Add processing stats if provided
        if processing_stats:
            report['processing_stats'] = processing_stats
        
        return report
    
    def _compute_dataset_stats(self, predictions_by_sequence: Dict[str, List[Dict]]) -> Dict:
        """Compute basic dataset statistics"""
        total_sequences = len(predictions_by_sequence)
        total_frames = sum(len(preds) for preds in predictions_by_sequence.values())
        
        sequence_lengths = [len(preds) for preds in predictions_by_sequence.values()]
        
        processing_time = time.time() - self.start_time
        fps_throughput = total_frames / processing_time if processing_time > 0 else 0
        
        return {
            'total_sequences': total_sequences,
            'total_frames': total_frames,
            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
            'min_sequence_length': min(sequence_lengths) if sequence_lengths else 0,
            'max_sequence_length': max(sequence_lengths) if sequence_lengths else 0,
            'fps_throughput': fps_throughput,
        }
    
    def _compute_label_distribution(self, predictions_by_sequence: Dict[str, List[Dict]]) -> Dict:
        """Compute label distribution statistics"""
        all_labels = []
        for predictions in predictions_by_sequence.values():
            all_labels.extend([pred['label'] for pred in predictions])
        
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        distribution = {}
        for label in ['none', 'left', 'right', 'both']:
            count = label_counts.get(label, 0)
            distribution[label] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
        
        return distribution
    
    def _compute_confidence_stats(self, predictions_by_sequence: Dict[str, List[Dict]]) -> Dict:
        """Compute confidence score statistics"""
        all_confidences = []
        for predictions in predictions_by_sequence.values():
            all_confidences.extend([pred.get('confidence', 0.0) for pred in predictions])
        
        if not all_confidences:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'percentiles': {}
            }
        
        confidences = np.array(all_confidences)
        
        return {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'percentiles': {
                '25': float(np.percentile(confidences, 25)),
                '50': float(np.percentile(confidences, 50)),
                '75': float(np.percentile(confidences, 75)),
                '90': float(np.percentile(confidences, 90)),
                '95': float(np.percentile(confidences, 95)),
            }
        }
    
    def _compute_quality_metrics(self,
                                 predictions_by_sequence: Dict[str, List[Dict]],
                                 quality_reports: Optional[Dict[str, Dict]]) -> Dict:
        """Compute quality control metrics"""
        total_frames = sum(len(preds) for preds in predictions_by_sequence.values())
        
        # Count high confidence predictions
        high_conf_count = 0
        low_conf_count = 0
        
        for predictions in predictions_by_sequence.values():
            for pred in predictions:
                conf = pred.get('confidence', 0.0)
                if conf >= 0.7:
                    high_conf_count += 1
                elif conf < 0.5:
                    low_conf_count += 1
        
        # Count flagged frames from quality reports
        total_flagged = 0
        flag_reason_counts = Counter()
        
        if quality_reports:
            for report in quality_reports.values():
                if report and 'flagged_frames' in report:
                    total_flagged += len(report['flagged_frames'])
                    
                    for flagged in report['flagged_frames']:
                        for flag in flagged.get('flags', []):
                            flag_reason_counts[flag] += 1
        
        return {
            'high_confidence_rate': high_conf_count / total_frames if total_frames > 0 else 0,
            'low_confidence_rate': low_conf_count / total_frames if total_frames > 0 else 0,
            'flagged_for_review_count': total_flagged,
            'flagged_for_review_rate': total_flagged / total_frames if total_frames > 0 else 0,
            'flag_reasons': dict(flag_reason_counts),
        }
    
    def _compute_temporal_metrics(self, predictions_by_sequence: Dict[str, List[Dict]]) -> Dict:
        """Compute temporal analysis metrics"""
        all_episodes = []
        total_smoothed = 0
        total_reconstructed = 0
        total_constraint_enforced = 0
        total_frames = 0
        
        for predictions in predictions_by_sequence.values():
            # Count post-processing modifications
            for pred in predictions:
                total_frames += 1
                if pred.get('smoothed'):
                    total_smoothed += 1
                if pred.get('reconstructed'):
                    total_reconstructed += 1
                if pred.get('constraint_enforced'):
                    total_constraint_enforced += 1
            
            # Detect signal episodes
            episodes = self._detect_episodes(predictions)
            all_episodes.extend(episodes)
        
        # Compute episode statistics
        if all_episodes:
            episode_durations = [ep['duration'] for ep in all_episodes]
            avg_duration = np.mean(episode_durations)
            
            # Count episodes by label
            episode_counts = Counter([ep['label'] for ep in all_episodes])
        else:
            avg_duration = 0
            episode_counts = {}
        
        return {
            'signal_episodes_detected': len(all_episodes),
            'avg_signal_duration_frames': avg_duration,
            'episode_counts_by_label': dict(episode_counts),
            'smoothing_correction_rate': total_smoothed / total_frames if total_frames > 0 else 0,
            'reconstruction_rate': total_reconstructed / total_frames if total_frames > 0 else 0,
            'constraint_enforcement_rate': total_constraint_enforced / total_frames if total_frames > 0 else 0,
        }
    
    def _detect_episodes(self, predictions: List[Dict]) -> List[Dict]:
        """
        Detect continuous signal episodes in predictions.
        """
        episodes = []
        
        if not predictions:
            return episodes
        
        current_label = None
        episode_start = 0
        
        for i, pred in enumerate(predictions):
            label = pred['label']
            
            if label == 'none':
                # End current episode if exists
                if current_label is not None and current_label != 'none':
                    episodes.append({
                        'label': current_label,
                        'start_frame': episode_start,
                        'end_frame': i - 1,
                        'duration': i - episode_start
                    })
                    current_label = None
            else:
                # Signal active
                if label != current_label:
                    # End previous episode
                    if current_label is not None and current_label != 'none':
                        episodes.append({
                            'label': current_label,
                            'start_frame': episode_start,
                            'end_frame': i - 1,
                            'duration': i - episode_start
                        })
                    
                    # Start new episode
                    current_label = label
                    episode_start = i
        
        # Close final episode
        if current_label is not None and current_label != 'none':
            episodes.append({
                'label': current_label,
                'start_frame': episode_start,
                'end_frame': len(predictions) - 1,
                'duration': len(predictions) - episode_start
            })
        
        return episodes
    
    def save_report(self, report: Dict, output_path: str) -> str:
        """
        Save metrics report to JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved metrics report to {output_path}")
        return str(output_path)
    
    def print_summary(self, report: Dict):
        """
        Print a summary of the report.
        """
        print("\n" + "="*70)
        print(f"METRICS REPORT: {self.model_name}")
        print("="*70)
        
        # Dataset stats
        ds = report['dataset_stats']
        print(f"\nDataset: {self.dataset_name}")
        print(f"  Sequences: {ds['total_sequences']}")
        print(f"  Total frames: {ds['total_frames']}")
        print(f"  Avg sequence length: {ds['avg_sequence_length']:.1f} frames")
        print(f"  Processing throughput: {ds['fps_throughput']:.2f} FPS")
        
        # Label distribution
        print(f"\nLabel Distribution:")
        for label, stats in report['label_distribution'].items():
            print(f"  {label:8s}: {stats['count']:6d} ({stats['percentage']:5.1f}%)")
        
        # Confidence stats
        cs = report['confidence_stats']
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {cs['mean']:.3f}")
        print(f"  Median: {cs['median']:.3f}")
        print(f"  Std Dev: {cs['std']:.3f}")
        print(f"  Range: [{cs['min']:.3f}, {cs['max']:.3f}]")
        
        # Quality metrics
        qm = report['quality_metrics']
        print(f"\nQuality Metrics:")
        print(f"  High confidence rate: {qm['high_confidence_rate']:.1%}")
        print(f"  Flagged for review: {qm['flagged_for_review_count']} ({qm['flagged_for_review_rate']:.1%})")
        
        # Temporal metrics
        tm = report['temporal_metrics']
        print(f"\nTemporal Metrics:")
        print(f"  Signal episodes detected: {tm['signal_episodes_detected']}")
        print(f"  Avg signal duration: {tm['avg_signal_duration_frames']:.1f} frames")
        print(f"  Smoothing correction rate: {tm['smoothing_correction_rate']:.1%}")
        
        print("\n" + "="*70)


def generate_and_save_report(predictions_by_sequence: Dict[str, List[Dict]],
                            output_path: str,
                            model_name: str,
                            dataset_name: str,
                            quality_reports: Optional[Dict[str, Dict]] = None,
                            model_metrics: Optional[Dict] = None,
                            print_summary: bool = True) -> str:
    """
    Convenience function to generate and save metrics report.
    
    Args:
        predictions_by_sequence: Dict mapping sequence_id to predictions
        output_path: Output JSON file path
        model_name: Name of model used
        dataset_name: Name of dataset
        quality_reports: Optional quality control reports
        model_metrics: Optional model-specific metrics
        print_summary: Whether to print summary to console
    
    Returns:
        Path to saved report file
    """
    reporter = MetricsReporter(model_name, dataset_name)
    report = reporter.generate_report(
        predictions_by_sequence,
        quality_reports=quality_reports,
        model_metrics=model_metrics
    )
    
    saved_path = reporter.save_report(report, output_path)
    
    if print_summary:
        reporter.print_summary(report)
    
    return saved_path

