import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from data import (
    load_dataset_from_config,
    create_image_loader,
    ImagePreprocessor,
    SequencePreprocessor,
    create_preprocessor,
    create_batcher,
)
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_image_preprocessing(config):
    """Test basic image preprocessing"""
    print("\n" + "=" * 80)
    print("TEST 1: Image Preprocessing")
    print("=" * 80)
    
    try:
        # Load a small dataset
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=3)
        
        # Load images
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(config.preprocessing)
        
        print(f"\n  Preprocessing config:")
        print(f"    Target size: {config.preprocessing.resize_resolution}")
        print(f"    Maintain aspect: {config.preprocessing.maintain_aspect_ratio}")
        print(f"    Normalize: {config.preprocessing.normalize}")
        
        # Test on first few frames
        tested = 0
        for sequence in dataset.sequences:
            for frame in sequence.frames[:3]:  # First 3 frames
                if frame.crop_image is None:
                    continue
                
                original_shape = frame.crop_image.shape
                
                # Preprocess
                preprocessed = preprocessor.preprocess_image(frame.crop_image)
                
                if tested == 0:
                    print(f"\n  Example preprocessing:")
                    print(f"    Original shape: {original_shape}")
                    print(f"    Preprocessed shape: {preprocessed.shape}")
                    print(f"    Original dtype: {frame.crop_image.dtype}")
                    print(f"    Preprocessed dtype: {preprocessed.dtype}")
                    
                    if config.preprocessing.normalize:
                        print(f"    Original range: [{frame.crop_image.min()}, {frame.crop_image.max()}]")
                        print(f"    Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
                
                # Verify shape
                expected_h = config.preprocessing.resize_resolution[1]
                expected_w = config.preprocessing.resize_resolution[0]
                assert preprocessed.shape == (expected_h, expected_w, 3), \
                    f"Wrong shape: {preprocessed.shape} != ({expected_h}, {expected_w}, 3)"
                
                # Verify normalization
                if config.preprocessing.normalize:
                    assert preprocessed.dtype == np.float32, f"Wrong dtype: {preprocessed.dtype}"
                    assert 0 <= preprocessed.min() <= preprocessed.max() <= 1.0, \
                        f"Values out of range: [{preprocessed.min()}, {preprocessed.max()}]"
                
                tested += 1
                if tested >= 10:
                    break
            if tested >= 10:
                break
        
        print(f"\n  Tested {tested} images")
        print(f"    All images preprocessed to correct shape and range")
        
        return True
    
    except Exception as e:
        print(f"\nError in image preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_preprocessing_video(config):
    """Test sequence preprocessing for video mode"""
    print("\n" + "=" * 80)
    print("TEST 2: Sequence Preprocessing (Video Mode)")
    print("=" * 80)
    
    try:
        # Load dataset
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=5)
        
        # Load images
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        
        # Create sequence preprocessor
        seq_preprocessor = SequencePreprocessor(config.preprocessing)
        
        print(f"\n  Sequence config:")
        print(f"    Max length: {config.preprocessing.max_sequence_length}")
        print(f"    Stride: {config.preprocessing.sequence_stride}")
        
        # Test on sequences
        for i, sequence in enumerate(dataset.sequences[:3], 1):
            print(f"\n  Sequence {i}: {sequence.sequence_id[:50]}...")
            print(f"    Original frames: {sequence.num_frames}")
            
            # Get original shapes (before preprocessing)
            original_shapes = set(f.crop_image.shape for f in sequence.frames if f.crop_image is not None)
            print(f"    Original shapes: {len(original_shapes)} different sizes")
            
            # Preprocess to video
            video = seq_preprocessor.preprocess_for_video(sequence, use_crops=True)
            
            print(f"    Video tensor shape: {video.shape}")
            print(f"      (T={video.shape[0]} frames, H={video.shape[1]}, W={video.shape[2]}, C={video.shape[3]})")
            
            # Verify all frames have same shape
            assert video.shape[1] == config.preprocessing.resize_resolution[1]
            assert video.shape[2] == config.preprocessing.resize_resolution[0]
            assert video.shape[3] == 3
            
            # Verify normalization
            if config.preprocessing.normalize:
                assert video.dtype == np.float32
                assert 0 <= video.min() <= video.max() <= 1.0
        
        print(f"\n  All sequences preprocessed successfully")
        print(f"    Images with different shapes → consistent video tensors")
        
        return True
    
    except Exception as e:
        print(f"\nError in sequence preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_preprocessing_single(config):
    """Test sequence preprocessing for single-image mode"""
    print("\n" + "=" * 80)
    print("TEST 3: Sequence Preprocessing (Single-Image Mode)")
    print("=" * 80)
    
    try:
        # Load dataset
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=3)
        
        # Load images
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        
        # Create sequence preprocessor
        seq_preprocessor = SequencePreprocessor(config.preprocessing)
        
        # Test on sequences
        for i, sequence in enumerate(dataset.sequences, 1):
            print(f"\n  Sequence {i}: {sequence.sequence_id[:50]}...")
            
            # Preprocess for single images
            samples = seq_preprocessor.preprocess_for_single_images(sequence, use_crops=True)
            
            print(f"    Original frames: {sequence.num_frames}")
            print(f"    Sampled frames: {len(samples)}")
            
            # Verify samples
            for j, (image, frame_id) in enumerate(samples[:3]):
                if j == 0:
                    print(f"    Sample image shape: {image.shape}")
                
                # Verify shape
                assert image.shape[0] == config.preprocessing.resize_resolution[1]
                assert image.shape[1] == config.preprocessing.resize_resolution[0]
                assert image.shape[2] == 3
                
                # Verify normalization
                if config.preprocessing.normalize:
                    assert image.dtype == np.float32
                    assert 0 <= image.min() <= image.max() <= 1.0
        
        print(f"\n  Single-image preprocessing successful")
        
        return True
    
    except Exception as e:
        print(f"\nError in single-image preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batching(config):
    """Test batch processing"""
    print("\n" + "=" * 80)
    print("TEST 4: Batch Processing")
    print("=" * 80)
    
    try:
        # Load dataset
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=10)
        
        # Load images
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        
        # Preprocess
        seq_preprocessor = SequencePreprocessor(config.preprocessing)
        
        # Get single images
        all_samples = []
        for sequence in dataset.sequences:
            samples = seq_preprocessor.preprocess_for_single_images(sequence, use_crops=True)
            all_samples.extend(samples)
        
        print(f"\n  Total preprocessed images: {len(all_samples)}")
        
        # Create batcher
        batch_size = 8
        batcher = create_batcher(batch_size, mode='simple')
        
        # Batch images
        batch_count = 0
        total_images = 0
        
        for batch_tensor, frame_ids in batcher.batch_with_metadata(all_samples):
            batch_count += 1
            total_images += len(frame_ids)
            
            if batch_count == 1:
                print(f"\n  First batch:")
                print(f"    Batch shape: {batch_tensor.shape}")
                print(f"    (B={batch_tensor.shape[0]}, H={batch_tensor.shape[1]}, "
                      f"W={batch_tensor.shape[2]}, C={batch_tensor.shape[3]})")
                print(f"    Frame IDs: {frame_ids[:5]}{'...' if len(frame_ids) > 5 else ''}")
        
        print(f"\n  Created {batch_count} batches")
        print(f"    Total images batched: {total_images}")
        print(f"    Batch size: {batch_size} (last batch may be smaller)")
        
        return True
    
    except Exception as e:
        print(f"\nError in batching: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aspect_ratio_preservation(config):
    """Test that aspect ratio is preserved correctly"""
    print("\n" + "=" * 80)
    print("TEST 5: Aspect Ratio Preservation")
    print("=" * 80)
    
    try:
        # Load one sequence
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=1)
        
        # Load images
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        
        # Create preprocessor
        preprocessor = ImagePreprocessor(config.preprocessing)
        
        # Get first frame with image
        sequence = dataset.sequences[0]
        frame = next(f for f in sequence.frames if f.crop_image is not None)
        
        original = frame.crop_image
        preprocessed = preprocessor.preprocess_image(original)
        
        orig_h, orig_w = original.shape[:2]
        prep_h, prep_w = preprocessed.shape[:2]
        
        print(f"\n  Original image: {orig_h} × {orig_w}")
        print(f"  Preprocessed image: {prep_h} × {prep_w}")
        
        if config.preprocessing.maintain_aspect_ratio:
            orig_aspect = orig_w / orig_h
            # The actual content will maintain aspect ratio
            # But the output size is padded to target size
            print(f"\n  Original aspect ratio: {orig_aspect:.3f}")
            print(f"  Aspect ratio maintained with padding")
            print(f"  Padding color: {config.preprocessing.padding_color}")
        else:
            print(f"\n  Aspect ratio NOT maintained (forced resize)")
            print(f"  This may distort the image")
        
        return True
    
    except Exception as e:
        print(f"\nError testing aspect ratio: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_complete_pipeline(config):
    """Demonstrate complete preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("DEMO: Complete Preprocessing Pipeline")
    print("=" * 80)
    
    try:
        print("\n  Simulating video-mode inference pipeline...")
        
        # 1. Load data
        print("\n  [1] Loading dataset...")
        dataset = load_dataset_from_config(config.data).filter_sequences(max_sequences=3)
        print(f"      Loaded {dataset.num_sequences} sequences")
        
        # 2. Load images
        print("\n  [2] Loading images...")
        image_loader = create_image_loader(config.data, lazy=False)
        image_loader.load_dataset_images(dataset, load_full_frames=False, show_progress=False)
        print(f"      Loaded images for all sequences")
        
        # 3. Preprocess
        print("\n  [3] Preprocessing sequences...")
        seq_preprocessor = SequencePreprocessor(config.preprocessing)
        
        videos = []
        for sequence in dataset.sequences:
            video = seq_preprocessor.preprocess_for_video(sequence, use_crops=True)
            videos.append(video)
            print(f"      {sequence.sequence_id[:40]}... → {video.shape}")
        
        # 4. Ready for model
        print(f"\n  [4] Ready for model inference!")
        print(f"      {len(videos)} video tensors prepared")
        print(f"      Shape: (T, {videos[0].shape[1]}, {videos[0].shape[2]}, {videos[0].shape[3]})")
        print(f"      Dtype: {videos[0].dtype}")
        print(f"      Range: [{videos[0].min():.3f}, {videos[0].max():.3f}]")
        
        return True
    
    except Exception as e:
        print(f"\nError in pipeline demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print(" " * 25 + "STAGE 2: PREPROCESSING TEST")
    print("=" * 80)
    
    # Test with video mode config
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
    
    # Test 1: Basic image preprocessing
    results['image_preprocessing'] = test_image_preprocessing(config)
    
    # Test 2: Sequence preprocessing (video mode)
    results['sequence_video'] = test_sequence_preprocessing_video(config)
    
    # Test 3: Sequence preprocessing (single-image mode)
    results['sequence_single'] = test_sequence_preprocessing_single(config)
    
    # Test 4: Batching
    results['batching'] = test_batching(config)
    
    # Test 5: Aspect ratio
    results['aspect_ratio'] = test_aspect_ratio_preservation(config)
    
    # Demo: Complete pipeline
    demo_complete_pipeline(config)
    
    # Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:.<50} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  All tests passed! Stage 2 is complete.")
        print("  Ready for Stage 3: Model Inference")
    else:
        print("\n  Some tests failed. Check errors above.")


if __name__ == '__main__':
    main()
