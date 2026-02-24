"""
Quality control for predictions.
Flags anomalies and inconsistencies for manual review.
"""
from typing import List, Dict, Set
import logging
from collections import Counter
import numpy as np


logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Identifies predictions that need manual review.
    Flags anomalies and inconsistencies.
    """
    def __init__(self, quality_control_config):
        # Handle None case with defaults
        if quality_control_config is None:
            # Use default values
            self.flag_both_signals = True
            self.flag_rapid_changes = True
            self.rapid_change_threshold = 3
            self.sample_rate = 0.05
            self.stratified = True
        else:
            self.config = quality_control_config
            self.flag_both_signals = quality_control_config.flag_both_signals
            self.flag_rapid_changes = quality_control_config.flag_rapid_changes
            self.rapid_change_threshold = quality_control_config.rapid_change_threshold
            self.sample_rate = quality_control_config.random_sample_rate
            self.stratified = quality_control_config.stratified_sampling
    
    def check_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Run quality checks on predictions.
        """
        report = {
            'total_frames': len(predictions),
            'flagged_frames': [],
            'flag_reasons': Counter(),
            'label_distribution': Counter(),
        }
        
        if not predictions:
            return report
        
        # Compute statistics
        labels = [p['label'] for p in predictions]
        report['label_distribution'] = Counter(labels)
        
        # Run checks
        for i, pred in enumerate(predictions):
            flags = []
            
            # Check 1: Both signals (rare/anomalous)
            if self.flag_both_signals and pred['label'] in {'both', 'hazard'}:
                flags.append('both_signals')
            
            # Check 2: Rapid changes
            if self.flag_rapid_changes and i > 0:
                # Count changes in recent history
                start = max(0, i - self.rapid_change_threshold)
                recent_labels = [predictions[j]['label'] for j in range(start, i + 1)]
                unique_labels = len(set(recent_labels))
                
                if unique_labels >= 3:  # 3+ different labels in small window
                    flags.append('rapid_changes')
            
            # Add to flagged list
            if flags:
                report['flagged_frames'].append({
                    'frame_id': pred.get('frame_id', i),
                    'label': pred['label'],
                    'flags': flags
                })
                
                for flag in flags:
                    report['flag_reasons'][flag] += 1
        
        # Random sampling for review
        sample_indices = self._sample_for_review(predictions)
        
        for idx in sample_indices:
            # Check if already flagged
            if not any(f.get('frame_id') == predictions[idx].get('frame_id', idx) for f in report['flagged_frames']):
                report['flagged_frames'].append({
                    'frame_id': predictions[idx].get('frame_id', idx),
                    'label': predictions[idx]['label'],
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
    


class ConstraintEnforcer:
    """
    Enforces physical constraints on turn signal predictions.
    """
    def __init__(self, postprocessing_config):
        self.config = postprocessing_config
        self.min_duration = postprocessing_config.min_signal_duration_frames
        self.max_duration = postprocessing_config.max_signal_duration_frames
    
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
    
