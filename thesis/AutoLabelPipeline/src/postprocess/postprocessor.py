"""
Post-processing orchestrator.
Coordinates temporal smoothing, quality control, and constraint enforcement.
"""
import logging
from typing import List, Dict, Tuple

from .temporal_smoother import TemporalSmoother, EpisodeReconstructor
from .quality_checker import QualityChecker, ConstraintEnforcer
from utils.enums import InferenceMode


logger = logging.getLogger(__name__)


class Postprocessor:
    """
    Orchestrates all post-processing steps.
    Handles both video and single-image modes.
    """
    
    def __init__(self, postprocessing_config, model_config, quality_control_config=None):
        self.config = postprocessing_config
        self.inference_mode = model_config.inference_mode
        
        # Initialize components
        self.temporal_smoother = TemporalSmoother(postprocessing_config)
        
        self.quality_checker = QualityChecker(quality_control_config)
        self.constraint_enforcer = ConstraintEnforcer(postprocessing_config)
        
        # Single-image mode components
        if self.inference_mode == InferenceMode.SINGLE_IMAGE and postprocessing_config.single_image:
            self.episode_reconstructor = EpisodeReconstructor(postprocessing_config.single_image)
        else:
            self.episode_reconstructor = None

    def _apply_temporal_smoothing(self, predictions: List[Dict], actual_num_frames: int) -> List[Dict]:
        """Apply temporal smoothing to predictions"""
        return self.temporal_smoother.smooth_sequence(predictions)
    
    def _apply_quality_control(self, predictions: List[Dict], actual_num_frames: int) -> Tuple[List[Dict], List[Dict]]:
        """Apply quality control checks"""
        # Run quality checker
        quality_report = self.quality_checker.check_predictions(predictions)
        
        # Apply constraint enforcement
        predictions = self.constraint_enforcer.enforce_constraints(predictions)
        
        # Mark flagged frames in predictions
        flagged_ids = {f.get('frame_id') for f in quality_report['flagged_frames']}
        for i, pred in enumerate(predictions):
            pred_frame_id = pred.get('frame_id', i)
            pred['flagged'] = pred_frame_id in flagged_ids
        
        return predictions, quality_report['flagged_frames']

    def _count_flag_reasons(self, flagged: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each flag reason"""
        from collections import Counter
        reasons = Counter()
        for f in flagged:
            for reason in f.get('flags', []):
                reasons[reason] += 1
        return dict(reasons)


    def _compute_label_distribution(self, predictions: List[Dict], actual_num_frames: int) -> Dict[str, int]:
        """Compute distribution of labels across all frames"""
        from collections import Counter
        
        # For segment-based predictions, count frames per label
        label_counts = Counter()
        
        for pred in predictions:
            if 'segments' in pred:
                # Multi-segment format
                for seg in pred['segments']:
                    label = seg['label']
                    start = seg.get('start_frame', 0)
                    end = seg.get('end_frame', 0)
                    frame_count = end - start + 1
                    label_counts[label] += frame_count
            else:
                # Single prediction format
                label_counts[pred['label']] += 1
        
        return dict(label_counts)
    
    def process_sequence(self, predictions: List[Dict], 
                     actual_num_frames: int = None,
                     apply_quality_control: bool = True) -> Dict:
        """
        Post-process predictions for a sequence.
        """
        if not predictions:
            return {'predictions': [], 'quality_report': {}, 'stats': {}}
        
        # Use actual_num_frames if provided
        if actual_num_frames is None:
            # Fallback: infer from predictions (may be wrong)
            actual_num_frames = max(
                (p.get('end_frame', 0) for p in predictions),
                default=len(predictions)
            ) + 1
        
        # Apply smoothing
        if self.config.temporal_smoothing_enabled:
            predictions = self._apply_temporal_smoothing(predictions, actual_num_frames)
        
        # Apply quality control
        flagged = []
        if apply_quality_control:
            predictions, flagged = self._apply_quality_control(predictions, actual_num_frames)
        
        # Generate quality report
        quality_report = {
            'total_frames': actual_num_frames, 
            'flagged_frames': flagged,
            'flag_reasons': self._count_flag_reasons(flagged),
            'label_distribution': self._compute_label_distribution(predictions, actual_num_frames),
            'total_flagged': len(flagged),
            'flag_rate': len(flagged) / actual_num_frames if actual_num_frames > 0 else 0,
        }
        
        return {
            'predictions': predictions,
            'quality_report': quality_report,
            'stats': {
                'input_predictions': len(predictions),
                'steps_applied': ['constraint_enforcement', 'quality_control'] if apply_quality_control else [],
                'flagged_frames': len(flagged),
                'output_predictions': len(predictions),
                'label_distribution': quality_report['label_distribution'],
            }
        }
    
    def process_dataset(self, sequences_with_predictions: Dict[str, List[Dict]],
                       apply_quality_control: bool = True) -> Dict:
        """
        Process predictions for entire dataset.
        """
        logger.info(f"Post-processing dataset: {len(sequences_with_predictions)} sequences")
        
        results = {}
        dataset_stats = {
            'total_sequences': len(sequences_with_predictions),
            'total_frames': 0,
            'total_flagged': 0,
            'label_distribution': {},
        }
        
        for sequence_id, predictions in sequences_with_predictions.items():
            result = self.process_sequence(predictions, apply_quality_control)
            results[sequence_id] = result
            
            # Aggregate stats
            dataset_stats['total_frames'] += len(predictions)
            if result['quality_report']:
                dataset_stats['total_flagged'] += result['quality_report']['total_flagged']
            
            # Aggregate label distribution
            for label, count in result['stats']['label_distribution'].items():
                dataset_stats['label_distribution'][label] = dataset_stats['label_distribution'].get(label, 0) + count
        
        logger.info(f"Dataset post-processing complete:")
        logger.info(f"  Total frames: {dataset_stats['total_frames']}")
        logger.info(f"  Total flagged: {dataset_stats['total_flagged']}")
        logger.info(f"  Label distribution: {dataset_stats['label_distribution']}")
        
        return {
            'sequences': results,
            'dataset_stats': dataset_stats
        }
    
    def _count_labels(self, predictions: List[Dict]) -> Dict[str, int]:
        """Count occurrences of each label"""
        from collections import Counter
        labels = [p['label'] for p in predictions]
        return dict(Counter(labels))


def create_postprocessor(config):
    """
    Factory function to create postprocessor.
    """
    return Postprocessor(config.postprocessing, config.model, config.quality_control)
