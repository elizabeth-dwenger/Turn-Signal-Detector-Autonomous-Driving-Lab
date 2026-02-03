#!/usr/bin/env python
"""
Runs all stages: data loading, preprocessing, inference, post-processing, and output.
"""
import sys
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

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


def setup_logging(log_file=None, verbose=False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_pipeline(config_path: str, verbose: bool = False):
    """
    Run complete pipeline with memory-efficient processing.
    """
    # Load configuration
    print("\n" + "="*80)
    print("TURN SIGNAL DETECTION PIPELINE")
    print("="*80)
    
    config = load_config(config_path)
    print(f"\nConfiguration: {config_path}")
    print(f"  Experiment: {config.experiment.name}")
    print(f"  Model: {config.model.type.value}")
    print(f"  Mode: {config.model.inference_mode.value}")
    print(f"  Output: {config.experiment.output_dir}")
    
    # Setup logging
    setup_logging(config.logging.log_file, verbose)
    logger = logging.getLogger(__name__)
    
    # Stage 1: Load dataset
    print("\n" + "-"*80)
    print("STAGE 1: Data Loading")
    print("-"*80)
    
    dataset = load_dataset_from_config(config.data)
    print(f"  Loaded {dataset.num_sequences} sequences ({dataset.total_frames} frames)")
    
    # Stage 2: Image Loading Setup (LAZY)
    print("\n" + "-"*80)
    print("STAGE 2: Image Loading Setup")
    print("-"*80)
    
    # Use regular loader but call it per-sequence to save memory
    image_loader = create_image_loader(config.data, lazy=False)
    print(f"  Using per-sequence image loading (memory-efficient)")
    
    # Stage 3: Preprocessing setup
    print("\n" + "-"*80)
    print("STAGE 3: Preprocessing Setup")
    print("-"*80)
    
    preprocessor = SequencePreprocessor(config.preprocessing)
    print(f"  Preprocessor ready")
    print(f"  Target size: {config.preprocessing.resize_resolution}")
    print(f"  Normalize: {config.preprocessing.normalize}")
    
    # Stage 4: Load model
    print("\n" + "-"*80)
    print("STAGE 4: Model Loading")
    print("-"*80)
    
    model = load_model(config.model, warmup=True)
    print(f"  Model loaded and ready")
    
    # Stage 5: Inference (process sequences one at a time)
    print("\n" + "-"*80)
    print("STAGE 5: Inference (Memory-Efficient)")
    print("-"*80)
    
    all_predictions = {}
    
    for sequence in tqdm(dataset.sequences, desc="Running inference"):
        # Load images for this sequence only
        image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
        
        # Check if images loaded successfully
        loaded_count = sum(1 for f in sequence.frames if f.crop_image is not None)
        if loaded_count == 0:
            logger.warning(f"No images loaded for sequence {sequence.sequence_id}, skipping")
            continue
        
        try:
            # Preprocess and predict
            if config.model.inference_mode.value == 'video':
                if (config.preprocessing.enable_chunking and 
                    loaded > config.preprocessing.chunk_size):
                    print(f"    Sequence is long ({loaded} frames), using chunked inference...")
                    chunks = preprocessor.preprocess_for_video_chunked(
                        sequence,
                        chunk_size=config.preprocessing.chunk_size
                    )
                    print(f"    Split into {len(chunks)} chunks")
                    prediction = model.predict_video(chunks=chunks)  # FIX: Use kwarg
                    predictions = [prediction]
                else:
                    video = preprocessor.preprocess_for_video(sequence)
                    print(f"    Video shape: {video.shape}")
                    prediction = model.predict_video(video=video)  # FIX: Use kwarg
                    predictions = [prediction]
            
            all_predictions[sequence.sequence_id] = predictions
        
        except Exception as e:
            logger.error(f"Error processing sequence {sequence.sequence_id}: {e}")
            continue
        
        finally:
            # Clear images from memory after processing
            for frame in sequence.frames:
                frame.crop_image = None
                frame.full_image = None
    
    print(f"\n  Inference complete: {len(all_predictions)}/{dataset.num_sequences} sequences")
    
    # Get model metrics
    model_metrics = model.get_metrics()
    print(f"  Total inferences: {model_metrics['total_inferences']}")
    print(f"  Avg latency: {model_metrics['avg_latency_ms']:.1f} ms")
    print(f"  Parse success: {model_metrics['parse_success_rate']:.1%}")
    
    # Stage 6: Post-processing
    print("\n" + "-"*80)
    print("STAGE 6: Post-processing")
    print("-"*80)
    
    postprocessor = create_postprocessor(config)
    
    processed_results = {}
    for sequence_id, predictions in tqdm(all_predictions.items(), desc="Post-processing"):
        result = postprocessor.process_sequence(predictions, apply_quality_control=True)
        processed_results[sequence_id] = result
    
    print(f"\n  Post-processing complete")
    
    # Aggregate stats
    total_flagged = sum(
        r.get('quality_report', {}).get('total_flagged', 0)
        for r in processed_results.values()
    )
    print(f"  Total frames flagged: {total_flagged}")
    
    # Stage 7: Output generation
    print("\n" + "-"*80)
    print("STAGE 7: Output Generation")
    print("-"*80)
    
    output_generator = create_output_generator(config)
    
    # Save predictions with correct frame counts
    for sequence_id, result in processed_results.items():
        seq = next((s for s in dataset.sequences if s.sequence_id == sequence_id), None)
        actual_num_frames = seq.num_frames if seq else 1
        
        # Attach to result for output generator to use
        result['actual_num_frames'] = actual_num_frames
    
    summary = output_generator.save_dataset_predictions(processed_results)
    print(f"  Saved predictions in {len(config.output.formats)} format(s)")
    print(f"  Output directory: {config.experiment.output_dir}")
    
    # Create visualizations (if enabled)
    if config.output.save_visualizations:
        print(f"\nCreating visualizations...")
        
        # For visualizations, we need to reload a sample of images
        import random
        sequences_to_viz = random.sample(
            list(processed_results.keys()),
            min(len(processed_results), max(1, int(len(processed_results) * config.output.visualization_sample_rate)))
        )
        
        viz_data = {}
        for sequence_id in tqdm(sequences_to_viz, desc="Preparing visualizations"):
            result = processed_results[sequence_id]
            
            # Find corresponding sequence
            seq = next((s for s in dataset.sequences if s.sequence_id == sequence_id), None)
            if seq:
                # Reload images for this sequence
                image_loader.load_sequence_images(seq, load_full_frames=False, show_progress=False)
                
                images = [f.crop_image for f in seq.frames if f.crop_image is not None]
                
                viz_data[sequence_id] = {
                    'images': images,
                    'predictions': result['predictions'],
                    'ground_truth': [f.true_label for f in seq.frames] if seq.has_ground_truth else None
                }
                
                # Clear after collecting
                for frame in seq.frames:
                    frame.crop_image = None
                    frame.full_image = None
        
        output_generator.create_visualizations(viz_data)
        print(f"  Visualizations created")
    
    # Generate report
    dataset_stats = {
        'total_sequences': len(processed_results),
        'total_frames': sum(len(r['predictions']) for r in processed_results.values()),
        'total_flagged': total_flagged,
        'label_distribution': {}
    }
    
    # Aggregate label distribution
    for result in processed_results.values():
        for label, count in result['stats']['label_distribution'].items():
            dataset_stats['label_distribution'][label] = dataset_stats['label_distribution'].get(label, 0) + count
    
    report_path = output_generator.generate_report(dataset_stats, model_metrics)
    print(f"  Generated report: {report_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {config.experiment.output_dir}")
    print(f"  Sequences processed: {dataset_stats['total_sequences']}")
    print(f"  Total frames: {dataset_stats['total_frames']}")
    print(f"  Frames flagged for review: {total_flagged}")
    print(f"\nLabel distribution:")
    for label, count in sorted(dataset_stats['label_distribution'].items()):
        pct = count / dataset_stats['total_frames'] * 100 if dataset_stats['total_frames'] > 0 else 0
        print(f"  {label:.<20} {count:>6} ({pct:>5.1f}%)")
    
    print("\n Pipeline execution successful!")


def main():
    parser = argparse.ArgumentParser(description='Run turn signal detection pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.config, args.verbose)
        sys.exit(0)
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
