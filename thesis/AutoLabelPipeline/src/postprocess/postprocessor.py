"""
Post-processing orchestrator.
Coordinates temporal smoothing, quality control, and constraint enforcement.
"""
from typing import List, Dict
import logging
import sys
from pathlib import Path

# Add src to path if needed
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.postprocess.temporal_smoother import TemporalSmoother, EpisodeReconstructor
from src.postprocess.quality_checker import QualityChecker, ConstraintEnforcer
from src.utils.enums import InferenceMode


logger = logging.getLogger(__name__)


class Postprocessor:
    """
    Orchestrates all post-processing steps.
    Handles both video and single-image modes.
    """
    
    def __init__(self, postprocessing_config, model_config):
        self.config = postprocessing_config
        self.inference_mode = model_config.inference_mode
        
        # Initialize components
        self.temporal_smoother = TemporalSmoother(postprocessing_config)
        
        # Quality checker - try to get quality_control from postprocessing_config or config
        quality_control_config = None
        if hasattr(postprocessing_config, 'quality_control'):
            quality_control_config = postprocessing_config.quality_control
        
        self.quality_checker = QualityChecker(quality_control_config)
        self.constraint_enforcer = ConstraintEnforcer(postprocessing_config)
        
        # Single-image mode components
        if self.inference_mode == InferenceMode.SINGLE_IMAGE and postprocessing_config.single_image:
            self.episode_reconstructor = EpisodeReconstructor(postprocessing_config.single_image)
        else:
            self.episode_reconstructor = None
    
    def process_sequence(self, predictions: List[Dict], apply_quality_control: bool = True) -> Dict:
        """
        Apply all post-processing steps to sequence predictions.
        """
        logger.debug(f"Post-processing {len(predictions)} predictions ({self.inference_mode.value} mode)")
        
        stats = {
            'input_predictions': len(predictions),
            'steps_applied': [],
        }
        
        # Step 1: Episode reconstruction (single-image mode only)
        if self.episode_reconstructor:
            logger.debug("Step 1: Episode reconstruction")
            predictions = self.episode_reconstructor.reconstruct_episodes(predictions)
            stats['steps_applied'].append('episode_reconstruction')
        
        # Step 2: Temporal smoothing
        if self.temporal_smoother.enabled:
            logger.debug("Step 2: Temporal smoothing")
            original_labels = [p['label'] for p in predictions]
            predictions = self.temporal_smoother.smooth_sequence(predictions)
            smoothed_count = sum(1 for i, p in enumerate(predictions) if p['label'] != original_labels[i])
            stats['smoothed_frames'] = smoothed_count
            stats['steps_applied'].append('temporal_smoothing')
        
        # Step 3: Constraint enforcement
        logger.debug("Step 3: Constraint enforcement")
        predictions = self.constraint_enforcer.enforce_constraints(predictions)
        stats['steps_applied'].append('constraint_enforcement')
        
        # Step 4: Quality control
        quality_report = None
        if apply_quality_control:
            logger.debug("Step 4: Quality control")
            quality_report = self.quality_checker.check_predictions(predictions)
            stats['flagged_frames'] = quality_report['total_flagged']
            stats['steps_applied'].append('quality_control')
        
        # Final statistics
        stats['output_predictions'] = len(predictions)
        stats['label_distribution'] = self._count_labels(predictions)
        
        logger.info(f"Post-processing complete: {len(stats['steps_applied'])} steps applied")
        if quality_report:
            logger.info(f"  {quality_report['total_flagged']} frames flagged for review")
        
        return {
            'predictions': predictions,
            'quality_report': quality_report,
            'stats': stats
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
    return Postprocessor(config.postprocessing, config.model)
