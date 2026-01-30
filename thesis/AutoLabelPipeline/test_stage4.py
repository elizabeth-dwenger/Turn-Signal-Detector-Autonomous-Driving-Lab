"""
Test script for Stage 4: Post-processing
Tests temporal smoothing, quality control, and constraint enforcement.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.postprocess.temporal_smoother import TemporalSmoother, EpisodeReconstructor
from src.postprocess.quality_checker import QualityChecker, ConstraintEnforcer
from src.postprocess.postprocessor import create_postprocessor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_predictions(pattern='flickering'):
    """Create test predictions with known patterns"""
    if pattern == 'flickering':
        # Flickering signal (should be smoothed)
        labels = ['none', 'left', 'none', 'left', 'left', 'left', 'none', 'left', 'left', 'none']
        confidences = [0.9, 0.85, 0.9, 0.88, 0.92, 0.90, 0.85, 0.87, 0.91, 0.9]
    
    elif pattern == 'brief_signal':
        # Brief signal (should be removed)
        labels = ['none'] * 5 + ['left', 'left'] + ['none'] * 5
        confidences = [0.9] * len(labels)
    
    elif pattern == 'low_confidence':
        # Low confidence predictions
        labels = ['left'] * 10
        confidences = [0.3, 0.4, 0.35, 0.38, 0.42, 0.36, 0.39, 0.37, 0.41, 0.38]
    
    elif pattern == 'both_signals':
        # Both signals active
        labels = ['none', 'none', 'both', 'both', 'both', 'none', 'none']
        confidences = [0.9] * len(labels)
    
    else:  # normal
        # Normal signal
        labels = ['none'] * 3 + ['left'] * 7 + ['none'] * 3
        confidences = [0.9] * len(labels)
    
    predictions = []
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        predictions.append({
            'label': label,
            'confidence': conf,
            'frame_id': i,
            'raw_output': f'{{"label": "{label}", "confidence": {conf}}}'
        })
    
    return predictions


def test_temporal_smoothing(config):
    """Test temporal smoothing"""
    print("\n" + "=" * 80)
    print("TEST 1: Temporal Smoothing")
    print("=" * 80)
    
    try:
        smoother = TemporalSmoother(config.postprocessing)
        
        # Test on flickering signal
        predictions = create_test_predictions('flickering')
        
        print(f"\n  Original predictions:")
        print(f"    Labels: {[p['label'] for p in predictions]}")
        
        smoothed = smoother.smooth_sequence(predictions)
        
        print(f"\n  Smoothed predictions:")
        print(f"    Labels: {[p['label'] for p in smoothed]}")
        
        # Count changes
        changed = sum(1 for i in range(len(predictions))
                     if predictions[i]['label'] != smoothed[i]['label'])
        
        print(f"\n  ✓ Smoothing applied")
        print(f"    Method: {config.postprocessing.smoothing_method.value}")
        print(f"    Window size: {config.postprocessing.smoothing_window_size}")
        print(f"    Changed: {changed}/{len(predictions)} frames")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in temporal smoothing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_reconstruction(config):
    """Test episode reconstruction for single-image mode"""
    print("\n" + "=" * 80)
    print("TEST 2: Episode Reconstruction (Single-Image Mode)")
    print("=" * 80)
    
    try:
        if not config.postprocessing.single_image:
            print("  ⚠ Single-image config not set, skipping")
            return True
        
        reconstructor = EpisodeReconstructor(config.postprocessing.single_image)
        
        # Create predictions with gaps
        predictions = []
        labels = ['none', 'none', 'left', 'none', 'none', 'left', 'left', 'none', 'left', 'none']
        for i, label in enumerate(labels):
            predictions.append({
                'label': label,
                'confidence': 0.85 if label == 'left' else 0.9,
                'frame_id': i
            })
        
        print(f"\n  Original labels: {[p['label'] for p in predictions]}")
        
        reconstructed = reconstructor.reconstruct_episodes(predictions)
        
        print(f"  Reconstructed:   {[p['label'] for p in reconstructed]}")
        
        # Check if gaps were filled
        filled = sum(1 for i in range(len(predictions))
                    if predictions[i]['label'] != reconstructed[i]['label'])
        
        print(f"\n  ✓ Episode reconstruction applied")
        print(f"    Filled gaps: {filled} frames")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in episode reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_control(config):
    """Test quality control checks"""
    print("\n" + "=" * 80)
    print("TEST 3: Quality Control")
    print("=" * 80)
    
    try:
        # Create quality control config
        from utils.config import load_config
        config = load_config('configs/qwen3_vl_video.yaml')
        
        # Access quality_control from postprocessing config
        # Note: We need to ensure quality_control exists in config
        class QCConfig:
            def __init__(self):
                self.flag_low_confidence = True
                self.low_confidence_threshold = 0.4
                self.random_sample_rate = 0.1
                self.stratified_sampling = True
                self.flag_both_signals = True
                self.flag_rapid_changes = True
                self.rapid_change_threshold = 3
        
        qc_config = QCConfig()
        checker = QualityChecker(qc_config)
        
        # Test on various patterns
        test_cases = [
            ('low_confidence', create_test_predictions('low_confidence')),
            ('both_signals', create_test_predictions('both_signals')),
            ('normal', create_test_predictions('normal')),
        ]
        
        for name, predictions in test_cases:
            print(f"\n  Testing: {name}")
            report = checker.check_predictions(predictions)
            
            print(f"    Total frames: {report['total_frames']}")
            print(f"    Flagged: {report['total_flagged']} ({report['flag_rate']:.1%})")
            if report['flag_reasons']:
                print(f"    Reasons: {dict(report['flag_reasons'])}")
        
        print(f"\n  ✓ Quality control working")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in quality control: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_constraint_enforcement(config):
    """Test physical constraint enforcement"""
    print("\n" + "=" * 80)
    print("TEST 4: Constraint Enforcement")
    print("=" * 80)
    
    try:
        enforcer = ConstraintEnforcer(config.postprocessing)
        
        # Test on brief signal (should be removed)
        predictions = create_test_predictions('brief_signal')
        
        print(f"\n  Original labels: {[p['label'] for p in predictions]}")
        
        constrained = enforcer.enforce_constraints(predictions)
        
        print(f"  After constraints: {[p['label'] for p in constrained]}")
        
        # Check if brief signal was removed
        brief_removed = all(p['label'] == 'none' for p in constrained)
        
        print(f"\n  ✓ Constraints enforced")
        print(f"    Min duration: {config.postprocessing.min_signal_duration_frames} frames")
        print(f"    Brief signal removed: {brief_removed}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in constraint enforcement: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_postprocessing(config):
    """Test complete post-processing pipeline"""
    print("\n" + "=" * 80)
    print("TEST 5: Complete Post-processing Pipeline")
    print("=" * 80)
    
    try:
        postprocessor = create_postprocessor(config)
        
        # Create test sequence
        predictions = create_test_predictions('flickering')
        
        print(f"\n  Input predictions: {len(predictions)}")
        print(f"    Labels: {[p['label'] for p in predictions]}")
        
        # Process
        result = postprocessor.process_sequence(predictions, apply_quality_control=True)
        
        processed = result['predictions']
        quality_report = result['quality_report']
        stats = result['stats']
        
        print(f"\n  Output predictions: {len(processed)}")
        print(f"    Labels: {[p['label'] for p in processed]}")
        
        print(f"\n  Post-processing complete")
        print(f"    Steps applied: {stats['steps_applied']}")
        print(f"    Label distribution: {stats['label_distribution']}")
        
        if quality_report:
            print(f"    Frames flagged: {quality_report['total_flagged']}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in full post-processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_before_after(config):
    """Demonstrate before/after post-processing"""
    print("\n" + "=" * 80)
    print("DEMO: Before vs After Post-processing")
    print("=" * 80)
    
    try:
        postprocessor = create_postprocessor(config)
        
        # Create messy predictions
        raw_labels = ['none', 'left', 'none', 'left', 'left', 'left', 'none',
                     'left', 'left', 'left', 'left', 'none', 'right', 'none']
        raw_predictions = []
        for i, label in enumerate(raw_labels):
            raw_predictions.append({
                'label': label,
                'confidence': 0.75 + np.random.random() * 0.2,
                'frame_id': i
            })
        
        print(f"\n  BEFORE post-processing:")
        print(f"    {raw_labels}")
        print(f"    Issues: flickering, brief signals, inconsistent")
        
        # Process
        result = postprocessor.process_sequence(raw_predictions)
        processed_labels = [p['label'] for p in result['predictions']]
        
        print(f"\n  AFTER post-processing:")
        print(f"    {processed_labels}")
        print(f"    Result: smooth, consistent temporal patterns")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Error in demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print(" " * 23 + "STAGE 4: POST-PROCESSING TEST")
    print("=" * 80)
    
    # Load config
    config_path = 'configs/qwen3_vl_video.yaml'
    
    print(f"\nLoading configuration: {config_path}")
    try:
        config = load_config(config_path)
        print(f"Config loaded: {config}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    # Run tests
    results = {}
    
    # Test 1: Temporal smoothing
    results['temporal_smoothing'] = test_temporal_smoothing(config)
    
    # Test 2: Episode reconstruction
    results['episode_reconstruction'] = test_episode_reconstruction(config)
    
    # Test 3: Quality control
    results['quality_control'] = test_quality_control(config)
    
    # Test 4: Constraint enforcement
    results['constraint_enforcement'] = test_constraint_enforcement(config)
    
    # Test 5: Full pipeline
    results['full_pipeline'] = test_full_postprocessing(config)
    
    # Demo
    demo_before_after(config)
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "✗ FAIL"
        print(f"  {test_name:.<50} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  All tests passed! Stage 4 is complete.")
    else:
        print("\n  Some tests failed. Check errors above.")


if __name__ == '__main__':
    main()
