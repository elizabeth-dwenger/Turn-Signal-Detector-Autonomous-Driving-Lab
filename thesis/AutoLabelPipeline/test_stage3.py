"""
Test script for Stage 3: Model Inference
Enhanced version that can test multiple models.
"""
import sys
from pathlib import Path
import numpy as np
from typing import List

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from utils.config import load_config
from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)
from models import create_model, load_model
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Test a single config
SINGLE_CONFIG = 'configs/qwen3_vl_video.yaml'

# Or test multiple configs
MULTIPLE_CONFIGS = [
    'configs/qwen3_vl_video.yaml',
    'configs/qwen3_vl_single.yaml',
    'configs/qwen25_vl_video.yaml',
    'configs/qwen25_vl_single.yaml',
    'configs/cosmos_reason1_video.yaml',
    'configs/cosmos_reason2_video.yaml',
]

# Set to True to test all configs, False to test only SINGLE_CONFIG
TEST_ALL = True  # <-- Change this to True to test all models


# ============================================================================
# Test Functions
# ============================================================================

def test_model_creation(config):
    """Test model factory"""
    print("\n" + "=" * 80)
    print("TEST 1: Model Creation")
    print("=" * 80)
    
    try:
        print(f"\n  Creating model: {config.model.type.value}")
        print(f"    Path: {config.model.model_name_or_path}")
        print(f"    Mode: {config.model.inference_mode.value}")
        
        model = create_model(config.model)
        
        print(f"\n    Model created: {model}")
        print(f"    Class: {model.__class__.__name__}")
        print(f"    Device: {model.device}")
        
        return model
    
    except Exception as e:
        print(f"\n  Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_loading(config):
    """Test model loading"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Loading")
    print("=" * 80)
    
    try:
        print(f"\n  Loading model: {config.model.model_name_or_path}")
        print(f"  This may take 30-60 seconds...")
        
        model = load_model(config.model, warmup=True)
        
        print(f"\n    Model loaded successfully!")
        print(f"    Model type: {type(model.model).__name__ if model.model else 'N/A'}")
        print(f"    Tokenizer/Processor: {type(model.tokenizer).__name__ if model.tokenizer else type(model.processor).__name__ if model.processor else 'N/A'}")
        
        return model
    
    except Exception as e:
        print(f"\n  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_image_inference(model, config):
    """Test inference on single image"""
    print("\n" + "=" * 80)
    print("TEST 3: Single Image Inference")
    print("=" * 80)
    
    # Skip if not in single image mode
    if config.model.inference_mode.value != "single":
        print("\n  ⊘ Skipping (not in single image mode)")
        return None
    
    try:
        print("\n  Loading sample data...")
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=1)
        
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, show_progress=False)
        
        preprocessor = SequencePreprocessor(config.preprocessing)
        sequence = dataset.sequences[0]
        samples = preprocessor.preprocess_for_single_images(sequence)
        
        if not samples:
            print("  ⚠ No samples available")
            return False
        
        image, frame_id = samples[0]
        
        print(f"\n  Testing on frame {frame_id}")
        print(f"    Image shape: {image.shape}")
        print(f"    Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        print(f"\n  Running inference...")
        prediction = model.predict_single(image)
        
        print(f"\n    Prediction successful!")
        print(f"    Label: {prediction['label']}")
        print(f"    Confidence: {prediction['confidence']:.3f}")
        print(f"    Latency: {prediction['latency_ms']:.1f} ms")
        
        if prediction.get('reasoning'):
            print(f"    Reasoning: {prediction['reasoning'][:100]}...")
        
        print(f"\n  Raw output:")
        print(f"    {prediction['raw_output'][:200]}...")
        
        return True
    
    except Exception as e:
        print(f"\n  Error in single image inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_inference(model, config):
    """Test inference on video sequence"""
    print("\n" + "=" * 80)
    print("TEST 4: Video Sequence Inference")
    print("=" * 80)
    
    # Skip if not in video mode
    if config.model.inference_mode.value != "video":
        print("\n  ⊘ Skipping (not in video mode)")
        return None
    
    try:
        print("\n  Loading sample sequence...")
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=1)
        
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, show_progress=False)
        
        preprocessor = SequencePreprocessor(config.preprocessing)
        sequence = dataset.sequences[0]
        video = preprocessor.preprocess_for_video(sequence)
        
        print(f"\n  Sequence: {sequence.sequence_id[:50]}...")
        print(f"    Video shape: {video.shape}")
        print(f"    Frames: {video.shape[0]}")
        print(f"    Resolution: {video.shape[2]}x{video.shape[1]}")
        
        print(f"\n  Running video inference...")
        prediction = model.predict_video(video)
        
        print(f"\n    Prediction successful!")
        print(f"    Label: {prediction['label']}")
        print(f"    Confidence: {prediction['confidence']:.3f}")
        print(f"    Latency: {prediction['latency_ms']:.1f} ms")
        
        if prediction.get('reasoning'):
            print(f"    Reasoning: {prediction['reasoning'][:100]}...")
        
        if sequence.has_ground_truth:
            gt_label = sequence.ground_truth_label
            match = "  MATCH" if prediction['label'] == gt_label else "  MISMATCH"
            print(f"\n  Ground truth: {gt_label}")
            print(f"  Prediction: {prediction['label']}")
            print(f"  {match}")
        
        return True
    
    except Exception as e:
        print(f"\n  Error in video inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_metrics(model):
    """Test metrics tracking"""
    print("\n" + "=" * 80)
    print("TEST 5: Model Metrics")
    print("=" * 80)
    
    try:
        metrics = model.get_metrics()
        
        print(f"\n  Metrics:")
        print(f"    Total inferences: {metrics['total_inferences']}")
        print(f"    Successful parses: {metrics['successful_parses']}")
        print(f"    Failed parses: {metrics['failed_parses']}")
        print(f"    Parse success rate: {metrics['parse_success_rate']:.2%}")
        print(f"    Average latency: {metrics['avg_latency_ms']:.1f} ms")
        
        return True
    
    except Exception as e:
        print(f"\n  Error getting metrics: {e}")
        return False


def test_single_config(config_path: str):
    """Test a single configuration"""
    print("\n" + "=" * 80)
    print(f" TESTING: {config_path}")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading configuration: {config_path}")
    try:
        config = load_config(config_path)
        print(f"  Config loaded: {config}")
    except Exception as e:
        print(f"  Failed to load config: {e}")
        return {'config_load': False}
    
    # Run tests
    results = {}
    model = None
    
    # Test 1: Model creation
    model = test_model_creation(config)
    results['model_creation'] = model is not None
    
    if not model:
        print("\n⚠ Cannot continue without model")
        return results
    
    # Test 2: Model loading
    model = test_model_loading(config)
    results['model_loading'] = model is not None
    
    if not model or not model.model:
        print("\n⚠ Model failed to load, cannot test inference")
        return results
    
    # Test 3 & 4: Inference (mode-dependent)
    if config.model.inference_mode.value == "single":
        results['single_image'] = test_single_image_inference(model, config)
    elif config.model.inference_mode.value == "video":
        results['video_inference'] = test_video_inference(model, config)
    
    # Test 5: Metrics
    results['metrics'] = test_model_metrics(model)
    
    return results


def print_summary(all_results: dict):
    """Print summary of all tests"""
    print("\n" + "=" * 80)
    print(" " * 30 + "OVERALL SUMMARY")
    print("=" * 80)
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        for test_name, passed in results.items():
            if passed is None:
                status = "⊘ SKIP"
            elif passed:
                status = "  PASS"
            else:
                status = "  FAIL"
            print(f"  {test_name:.<50} {status}")
    
    # Overall stats
    total_tests = sum(len([r for r in results.values() if r is not None])
                     for results in all_results.values())
    total_passed = sum(sum(1 for r in results.values() if r is True)
                      for results in all_results.values())
    
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  🎉 All tests passed! Stage 3 is complete.")
        print("  Ready for Stage 4: Post-processing")
    else:
        print("\n  ⚠ Some tests failed. Check errors above.")


def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STAGE 3: MODEL INFERENCE TEST")
    print("=" * 80)
    
    # Determine which configs to test
    if TEST_ALL:
        configs_to_test = MULTIPLE_CONFIGS
        print(f"\nTesting {len(configs_to_test)} configurations...")
    else:
        configs_to_test = [SINGLE_CONFIG]
        print(f"\nTesting single configuration: {SINGLE_CONFIG}")
    
    # Test each config
    all_results = {}
    for config_path in configs_to_test:
        # Check if config file exists
        if not Path(config_path).exists():
            print(f"\n⚠ Config file not found: {config_path}, skipping...")
            continue
        
        try:
            results = test_single_config(config_path)
            all_results[config_path] = results
        except Exception as e:
            print(f"\n  Unexpected error testing {config_path}: {e}")
            import traceback
            traceback.print_exc()
            all_results[config_path] = {'unexpected_error': False}
    
    # Print overall summary
    if len(all_results) > 1:
        print_summary(all_results)
    else:
        # Single config - summary already printed
        config_name = list(all_results.keys())[0]
        results = all_results[config_name]
        
        print("\n" + "=" * 80)
        print(" " * 30 + "TEST SUMMARY")
        print("=" * 80)
        
        for test_name, passed in results.items():
            if passed is None:
                status = " SKIP"
            elif passed:
                status = "  PASS"
            else:
                status = "  FAIL"
            print(f"  {test_name:.<50} {status}")
        
        total_passed = sum(1 for r in results.values() if r is True)
        total_tests = len([r for r in results.values() if r is not None])
        
        print(f"\n  Total: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("\n  All tests passed! Stage 3 is complete.")
            print("  Ready for Stage 4: Post-processing")
        else:
            print("\n  Some tests failed. Check errors above.")


if __name__ == '__main__':
    main()
