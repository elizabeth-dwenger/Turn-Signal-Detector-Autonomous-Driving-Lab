"""
Quality control for predictions.
Flags low-confidence predictions and anomalies for manual review.
"""
import numpy as np
from typing import List, Dict, Set
import logging
from collections import Counter


logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Identifies predictions that need manual review.
    Flags low confidence, anomalies, and inconsistencies.
    """
    
    def __init__(self, quality_control_config):
        """
        Args:
            quality_control_config: QualityControlConfig from configuration
        """
        # Handle None case with defaults
        if quality_control_config is None:
            # Use default values
            self.low_conf_threshold = 0.4
            self.flag_low_confidence = True
            self.flag_both_signals = True
            self.flag_rapid_changes = True
            self.rapid_change_threshold = 3
            self.sample_rate = 0.05
            self.stratified = True
        else:
            self.config = quality_control_config
            self.low_conf_threshold = quality_control_config.low_confidence_threshold
            self.flag_low_confidence = quality_control_config.flag_low_confidence
            self.flag_both_signals = quality_control_config.flag_both_signals
            self.flag_rapid_changes = quality_control_config.flag_rapid_changes
            self.rapid_change_threshold = quality_control_config.rapid_change_threshold
            self.sample_rate = quality_control_config.random_sample_rate
            self.stratified = quality_control_config.stratified_sampling
    
    def check_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Run quality checks on predictions.
        
        Args:
            predictions: List of prediction dicts
        
        Returns:
            Quality report dict with flagged frames
        """
        report = {
            'total_frames': len(predictions),
            'flagged_frames': [],
            'flag_reasons': Counter(),
            'confidence_stats': {},
            'label_distribution': Counter(),
        }
        
        if not predictions:
            return report
        
        # Compute statistics
        confidences = [p['confidence'] for p in predictions]
        labels = [p['label'] for p in predictions]
        
        report['confidence_stats'] = {
            'mean': np.mean(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
        }
        
        report['label_distribution'] = Counter(labels)
        
        # Run checks
        for i, pred in enumerate(predictions):
            flags = []
            
            # Check 1: Low confidence
            if self.flag_low_confidence and pred['confidence'] < self.low_conf_threshold:
                flags.append('low_confidence')
            
            # Check 2: Both signals (rare/anomalous)
            if self.flag_both_signals and pred['label'] == 'both':
                flags.append('both_signals')
            
            # Check 3: Rapid changes
            if self.flag_rapid_changes and i > 0:
                # Count changes in recent history
                start = max(0, i - self.rapid_change_threshold)
                recent_labels = [predictions[j]['label'] for j in range(start, i + 1)]
                unique_labels = len(set(recent_labels))
                
                if unique_labels >= 3:  # 3+ different labels in small window
                    flags.append('rapid_changes')
            
            # Check 4: Smoothing changed label significantly
            if pred.get('smoothed') or pred.get('reconstructed'):
                if pred.get('original_label') != pred['label']:
                    # Only flag if confidence was high but changed anyway
                    if pred['confidence'] > 0.7:
                        flags.append('high_conf_but_changed')
            
            # Add to flagged list
            if flags:
                report['flagged_frames'].append({
                    'frame_index': i,
                    'label': pred['label'],
                    'confidence': pred['confidence'],
                    'flags': flags
                })
                
                for flag in flags:
                    report['flag_reasons'][flag] += 1
        
        # Random sampling for review
        sample_indices = self._sample_for_review(predictions)
        
        for idx in sample_indices:
            # Check if already flagged
            if not any(f['frame_index'] == idx for f in report['flagged_frames']):
                report['flagged_frames'].append({
                    'frame_index': idx,
                    'label': predictions[idx]['label'],
                    'confidence': predictions[idx]['confidence'],
                    'flags': ['random_sample']
                })
                report['flag_reasons']['random_sample'] += 1
        
        # Summary stats
        report['total_flagged'] = len(report['flagged_frames'])
        report['flag_rate'] = report['total_flagged'] / report['total_frames'] if report['total_frames'] > 0 else 0
        
        logger.info(f"Quality check: {report['total_flagged']}/{report['total_frames']} frames flagged ({report['flag_rate']:.1%})")
        logger.info(f"  Flag reasons: {dict(report['flag_reasons'])}")
        
        return report
    
    def _sample_for_review(self, predictions: List[Dict]) -> Set[int]:
        """
        Select random sample of frames for manual review.
        """
        if self.sample_rate <= 0:
            return set()
        
        n_samples = int(len(predictions) * self.sample_rate)
        
        if not self.stratified:
            # Simple random sampling
            indices = np.random.choice(len(predictions), size=n_samples, replace=False)
            return set(indices.tolist())
        
        # Stratified sampling (equal samples per label)
        label_to_indices = {}
        for i, pred in enumerate(predictions):
            label = pred['label']
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        # Sample equally from each label
        n_labels = len(label_to_indices)
        samples_per_label = max(1, n_samples // n_labels)
        
        sampled_indices = set()
        for label, indices in label_to_indices.items():
            n_available = len(indices)
            n_to_sample = min(samples_per_label, n_available)
            
            sampled = np.random.choice(indices, size=n_to_sample, replace=False)
            sampled_indices.update(sampled.tolist())
        
        return sampled_indices
    
    def filter_by_confidence(self, predictions: List[Dict]) -> List[Dict]:
        """
        Filter predictions by confidence threshold.
        Low-confidence predictions set to 'none' or marked as uncertain.
        
        Args:
            predictions: List of prediction dicts
        
        Returns:
            Filtered predictions
        """
        filtered = []
        changed = 0
        
        for pred in predictions:
            filtered_pred = pred.copy()
            
            if pred['confidence'] < self.low_conf_threshold and pred['label'] != 'none':
                filtered_pred['original_label'] = pred['label']
                filtered_pred['label'] = 'none'
                filtered_pred['filtered'] = True
                changed += 1
            
            filtered.append(filtered_pred)
        
        if changed > 0:
            logger.info(f"Confidence filter: {changed}/{len(predictions)} predictions changed to 'none'")
        
        return filtered


class ConstraintEnforcer:
    """
    Enforces physical constraints on turn signal predictions.
    """
    
    def __init__(self, postprocessing_config):
        self.config = postprocessing_config
        self.min_duration = postprocessing_config.min_signal_duration_frames
        self.max_duration = postprocessing_config.max_signal_duration_frames
        self.allow_both = postprocessing_config.allow_both_signals
    
    def enforce_constraints(self, predictions: List[Dict]) -> List[Dict]:
        """
        Enforce physical constraints on predictions.
        """
        constrained = predictions.copy()
        
        # Constraint 1: Minimum duration
        constrained = self._enforce_min_duration(constrained)
        
        # Constraint 2: Maximum duration (if set)
        if self.max_duration:
            constrained = self._enforce_max_duration(constrained)
        
        # Constraint 3: Both signals (if not allowed)
        if not self.allow_both:
            constrained = self._remove_both_signals(constrained)
        
        return constrained
    
    def _enforce_min_duration(self, predictions: List[Dict]) -> List[Dict]:
        """Remove signal episodes shorter than minimum duration"""
        result = predictions.copy()
        
        i = 0
        while i < len(result):
            if result[i]['label'] != 'none':
                # Found signal start
                signal_label = result[i]['label']
                signal_start = i
                
                # Find end
                j = i
                while j < len(result) and result[j]['label'] == signal_label:
                    j += 1
                
                signal_duration = j - i
                
                # If too short, remove
                if signal_duration < self.min_duration:
                    for k in range(signal_start, j):
                        if 'original_label' not in result[k]:
                            result[k]['original_label'] = result[k]['label']
                        result[k]['label'] = 'none'
                        result[k]['constraint_enforced'] = True
                
                i = j
            else:
                i += 1
        
        return result
    
    def _enforce_max_duration(self, predictions: List[Dict]) -> List[Dict]:
        """Split or truncate episodes longer than maximum duration"""
        result = predictions.copy()
        
        i = 0
        while i < len(result):
            if result[i]['label'] != 'none':
                signal_label = result[i]['label']
                signal_start = i
                
                # Find end
                j = i
                while j < len(result) and result[j]['label'] == signal_label:
                    j += 1
                
                signal_duration = j - i
                
                # If too long, truncate end
                if signal_duration > self.max_duration:
                    truncate_start = signal_start + self.max_duration
                    for k in range(truncate_start, j):
                        if 'original_label' not in result[k]:
                            result[k]['original_label'] = result[k]['label']
                        result[k]['label'] = 'none'
                        result[k]['constraint_enforced'] = True
                
                i = j
            else:
                i += 1
        
        return result
    
    def _remove_both_signals(self, predictions: List[Dict]) -> List[Dict]:
        """Convert 'both' signals to 'none' if not allowed"""
        result = []
        changed = 0
        
        for pred in predictions:
            result_pred = pred.copy()
            
            if pred['label'] == 'both':
                result_pred['original_label'] = 'both'
                result_pred['label'] = 'none'
                result_pred['constraint_enforced'] = True
                changed += 1
            
            result.append(result_pred)
        
        if changed > 0:
            logger.info(f"Removed {changed} 'both' signals (not allowed)")
        
        return result
