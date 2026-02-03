#!/usr/bin/env python
"""
Testing script for turn signal detection.
Processes a small subset of data for quick iteration and prompt testing.
"""
import sys
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)
from models import load_model
from postprocess import create_postprocessor
from output import create_output_generator


def setup_logging(verbose=False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def test_pipeline(config_path: str, num_sequences: int = 10,
                  sequence_ids: list = None, verbose: bool = False):
    """
    Run pipeline on small test set.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("TURN SIGNAL DETECTION - TESTING MODE")
    print("="*80)
    
    # Load configuration
    config = load_config(config_path)
    print(f"\nConfiguration: {config_path}")
    print(f"  Experiment: {config.experiment.name}")
    print(f"  Model: {config.model.type.value}")
    print(f"  Mode: {config.model.inference_mode.value}")
    
    # Override to use small subset
    if sequence_ids:
        config.data.sequence_filter = sequence_ids
        config.data.max_sequences = None
        print(f"  Testing on specific sequences: {sequence_ids}")
    else:
        config.data.max_sequences = num_sequences
        print(f"  Testing on first {num_sequences} sequences")
    
    # Create timestamped test output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.type.value
    test_output_dir = Path(config.experiment.output_dir) / "test_runs" / f"{model_name}_{timestamp}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    config.experiment.output_dir = str(test_output_dir)
    
    # Always enable visualizations for testing
    config.output.save_visualizations = True
    config.output.visualization_sample_rate = 1.0  # Visualize all test sequences
    
    print(f"  Output: {config.experiment.output_dir}")
    print(f"  Timestamp: {timestamp}")
    
    # Stage 1: Load dataset
    print("\n" + "-"*80)
    print("STAGE 1: Data Loading")
    print("-"*80)
    
    dataset = load_dataset_from_config(config.data)
    print(f"  Loaded {dataset.num_sequences} sequences ({dataset.total_frames} frames)")
    
    if dataset.num_sequences == 0:
        print("\n No sequences loaded! Check your sequence_filter or data path.")
        return
    
    # Stage 2: Load images (LAZY LOADING to save memory)
    print("\n" + "-"*80)
    print("STAGE 2: Image Loading Setup")
    print("-"*80)
    print("  Setting up image loader (will load per-sequence)...")
    
    # Use regular loader but we'll call it per-sequence to save memory
    image_loader = create_image_loader(config.data, lazy=False)
    
    # Stage 3: Load model
    print("\n" + "-"*80)
    print("STAGE 3: Model Loading")
    print("-"*80)
    
    model = load_model(config.model, warmup=True)
    print(f"  Model loaded and ready")
    
    # Stage 4: Preprocessing setup
    print("\n" + "-"*80)
    print("STAGE 4: Preprocessing Setup")
    print("-"*80)
    
    preprocessor = SequencePreprocessor(config.preprocessing)
    print(f"  Target size: {config.preprocessing.resize_resolution}")
    print(f"  Normalize: {config.preprocessing.normalize}")
    
    # Stage 5: Inference (process each sequence individually)
    print("\n" + "-"*80)
    print("STAGE 5: Inference (Memory-Efficient)")
    print("-"*80)
    
    all_predictions = {}
    
    for i, sequence in enumerate(tqdm(dataset.sequences, desc="Processing sequences")):
        print(f"\n  Sequence {i+1}/{dataset.num_sequences}: {sequence.sequence_id}")
        print(f"    Frames: {sequence.num_frames}")
        
        # Load images for this sequence only
        image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
        
        # Check if images loaded
        loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
        print(f"    Loaded: {loaded}/{sequence.num_frames} images")
        
        if loaded == 0:
            print(f"    No images loaded, skipping sequence")
            continue
        
        # Preprocess and predict
        try:
            if config.model.inference_mode.value == 'video':
                video = preprocessor.preprocess_for_video(sequence)
                print(f"    Video shape: {video.shape}")
                
                prediction = model.predict_video(video)
                predictions = [prediction]  # Single prediction for whole sequence
            else:
                # Single-image mode
                samples = preprocessor.preprocess_for_single_images(sequence)
                images = [s[0] for s in samples]
                print(f"    Processing {len(images)} frames...")
                
                predictions = model.predict_batch(images)
            
            all_predictions[sequence.sequence_id] = predictions
            
            # Show prediction summary
            if config.model.inference_mode.value == 'video':
                pred = predictions[0]
                print(f"    Prediction: {pred['label']} (conf: {pred['confidence']:.2f})")
                print(f"    Latency: {pred['latency_ms']:.1f} ms")
                if pred.get('reasoning'):
                    print(f"    Reasoning: {pred['reasoning'][:100]}...")
            else:
                labels = [p['label'] for p in predictions]
                from collections import Counter
                label_counts = Counter(labels)
                print(f"    Label distribution: {dict(label_counts)}")
                avg_conf = np.mean([p['confidence'] for p in predictions])
                print(f"    Avg confidence: {avg_conf:.2f}")
        
        except Exception as e:
            print(f"     Error processing sequence: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            # Clear images to free memory
            for frame in sequence.frames:
                frame.crop_image = None
                frame.full_image = None
    
    if not all_predictions:
        print("\n No predictions generated!")
        return
    
    print(f"\n  Processed {len(all_predictions)} sequences")
    
    # Get model metrics
    model_metrics = model.get_metrics()
    print(f"\n  Model Metrics:")
    print(f"    Total inferences: {model_metrics['total_inferences']}")
    print(f"    Avg latency: {model_metrics['avg_latency_ms']:.1f} ms")
    print(f"    Parse success: {model_metrics['parse_success_rate']:.1%}")
    
    # Stage 6: Post-processing
    print("\n" + "-"*80)
    print("STAGE 6: Post-processing")
    print("-"*80)
    
    postprocessor = create_postprocessor(config)
    
    processed_results = {}
    for sequence_id, predictions in tqdm(all_predictions.items(), desc="Post-processing"):
        result = postprocessor.process_sequence(predictions, apply_quality_control=True)
        processed_results[sequence_id] = result
    
    print(f"  Post-processing complete")
    
    # Stage 7: Output generation
    print("\n" + "-"*80)
    print("STAGE 7: Output Generation")
    print("-"*80)
    
    output_generator = create_output_generator(config)
    
    # Save predictions
    summary = output_generator.save_dataset_predictions(processed_results)
    print(f"  Saved predictions")
    print(f"  Output directory: {config.experiment.output_dir}")
    
    # Also save individual sequence JSON files for easy notebook loading
    sequences_dir = Path(config.experiment.output_dir) / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    
    for sequence_id, result in processed_results.items():
        safe_id = sequence_id.replace('/', '_').replace('\\', '_')
        seq_file = sequences_dir / f"{safe_id}.json"
        
        # Find ground truth from dataset
        seq = next((s for s in dataset.sequences if s.sequence_id == sequence_id), None)
        gt_labels = [f.true_label for f in seq.frames] if seq and seq.has_ground_truth else None
        
        with open(seq_file, 'w') as f:
            json.dump({
                'sequence_id': sequence_id,
                'num_frames': len(result['predictions']),
                'predictions': result['predictions'],
                'ground_truth_labels': gt_labels,
                'ground_truth_sequence': seq.ground_truth_label if seq and seq.has_ground_truth else None,
                'quality_report': result.get('quality_report'),
                'stats': result.get('stats')
            }, f, indent=2, default=str)
    
    print(f"  Saved {len(processed_results)} individual sequence files to {sequences_dir}")
    
    # Create visualizations
    print(f"\n  Creating visualizations...")
    
    # We need to reload images for visualization
    viz_data = {}
    for sequence_id, result in processed_results.items():
        # Find corresponding sequence
        seq = next((s for s in dataset.sequences if s.sequence_id == sequence_id), None)
        if seq:
            # Reload images for this sequence
            image_loader.load_sequence_images(seq, load_full_frames=False, show_progress=False)
            
            # Collect images
            images = [f.crop_image for f in seq.frames if f.crop_image is not None]
            
            viz_data[sequence_id] = {
                'images': images,
                'predictions': result['predictions'],
                'ground_truth': [f.true_label for f in seq.frames] if seq.has_ground_truth else None
            }
            
            # Clear after visualization
            for frame in seq.frames:
                frame.crop_image = None
                frame.full_image = None
    
    output_generator.create_visualizations(viz_data)
    print(f"  Visualizations created")
    
    # Generate report
    dataset_stats = {
        'total_sequences': len(processed_results),
        'total_frames': sum(len(r['predictions']) for r in processed_results.values()),
        'total_flagged': sum(r.get('quality_report', {}).get('total_flagged', 0)
                            for r in processed_results.values()),
        'label_distribution': {}
    }
    
    for result in processed_results.values():
        for label, count in result['stats']['label_distribution'].items():
            dataset_stats['label_distribution'][label] = \
                dataset_stats['label_distribution'].get(label, 0) + count
    
    report_path = output_generator.generate_report(dataset_stats, model_metrics)
    print(f"  Report: {report_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  Sequences: {dataset_stats['total_sequences']}")
    print(f"  Frames: {dataset_stats['total_frames']}")
    print(f"  Flagged: {dataset_stats['total_flagged']}")
    print(f"\nOutput Location:")
    print(f"  {config.experiment.output_dir}")
    print(f"\nVisualizations:")
    viz_dir = Path(config.experiment.output_dir) / 'visualizations'
    if viz_dir.exists():
        viz_files = list(viz_dir.glob('*.png')) + list(viz_dir.glob('*.mp4'))
        print(f"  {len(viz_files)} visualization files created")
        print(f"  Location: {viz_dir}")
    
    print("\nTesting pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Test turn signal detection pipeline on small dataset'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--num-sequences', '-n', type=int, default=10,
                       help='Number of sequences to test (default: 10)')
    parser.add_argument('--sequences', '-s', nargs='+', type=str,
                       help='Specific sequence IDs to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        test_pipeline(
            args.config,
            num_sequences=args.num_sequences,
            sequence_ids=args.sequences,
            verbose=args.verbose
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
