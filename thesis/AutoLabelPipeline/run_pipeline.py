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

from utils.config import load_config, set_random_seeds
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


def _get_frame_ids_for_video(sequence, preprocessor, use_crops: bool = True,
                             apply_max_length: bool = True,
                             source_fps: float = None,
                             target_fps: float = None):
    """Get frame_ids used after stride/max_length filtering."""
    if use_crops:
        frames = [f for f in sequence.frames if f.crop_image is not None]
    else:
        frames = [f for f in sequence.frames if f.full_image is not None]

    # Optional FPS resampling (match preprocessor behavior)
    frames = preprocessor._maybe_resample_frames(frames, source_fps, target_fps)
    
    if preprocessor.stride > 1:
        frames = frames[::preprocessor.stride]
    
    if apply_max_length and preprocessor.max_length and len(frames) > preprocessor.max_length:
        import numpy as np
        indices = np.linspace(0, len(frames) - 1, preprocessor.max_length, dtype=int)
        frames = [frames[i] for i in indices]
    
    return [f.frame_id for f in frames]


def _segments_to_frames(segment_prediction, frame_ids, fps: float):
    """
    Convert segment-based prediction to per-frame predictions aligned to frame_ids.
    """
    num_frames = len(frame_ids)
    predictions = []
    segments = segment_prediction.get('segments', [])
    
    if not segments:
        for frame_id in frame_ids:
            predictions.append({
                'frame_id': frame_id,
                'label': segment_prediction.get('label', 'none'),
                'confidence': segment_prediction.get('confidence', 0.0),
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
        return predictions
    
    for seg in segments:
        label = seg['label']
        start = seg.get('start_frame', 0)
        end = seg.get('end_frame', num_frames - 1)
        confidence = seg.get('confidence', 0.5)
        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))
        start_frame_id = frame_ids[start]
        end_frame_id = frame_ids[end]
        
        for idx in range(start, end + 1):
            predictions.append({
                'frame_id': frame_ids[idx],
                'label': label,
                'confidence': confidence,
                'start_frame': start_frame_id,
                'end_frame': end_frame_id,
                'start_time_seconds': round(start_frame_id / fps, 2),
                'end_time_seconds': round(end_frame_id / fps, 2),
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
    
    predictions.sort(key=lambda x: x['frame_id'])
    seen = set()
    unique_predictions = []
    for pred in predictions:
        if pred['frame_id'] not in seen:
            unique_predictions.append(pred)
            seen.add(pred['frame_id'])
    
    if len(unique_predictions) < num_frames:
        pred_dict = {p['frame_id']: p for p in unique_predictions}
        complete_predictions = []
        for frame_id in frame_ids:
            if frame_id in pred_dict:
                complete_predictions.append(pred_dict[frame_id])
            else:
                complete_predictions.append({
                    'frame_id': frame_id,
                    'label': 'none',
                    'confidence': 0.5,
                    'reasoning': 'Gap filled'
                })
        return complete_predictions
    
    return unique_predictions


def run_pipeline(config_path: str, verbose: bool = False):
    """
    Run complete pipeline with memory-efficient processing.
    """
    # Load configuration
    print("\n" + "="*80)
    print("TURN SIGNAL DETECTION PIPELINE")
    print("="*80)
    
    config = load_config(config_path)
    set_random_seeds(config.experiment.random_seed)
    target_fps = config.model.model_kwargs.get('target_video_fps')
    config.model.model_kwargs['video_fps'] = target_fps or config.data.video_fps
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
            sequence_key = f"{sequence.sequence_id}__track_{sequence.track_id}"
            target_fps = config.model.model_kwargs.get('target_video_fps')
            source_fps = config.data.video_fps
            frame_ids = _get_frame_ids_for_video(
                sequence,
                preprocessor,
                use_crops=True,
                source_fps=source_fps,
                target_fps=target_fps
            )
            fps = target_fps or source_fps
            
            if config.model.inference_mode.value == 'video':
                if (config.preprocessing.enable_chunking and 
                    loaded_count > config.preprocessing.chunk_size):
                    print(f"    Sequence is long ({loaded_count} frames), using chunked inference...")
                    frame_ids = _get_frame_ids_for_video(
                        sequence,
                        preprocessor,
                        use_crops=True,
                        apply_max_length=False,
                        source_fps=source_fps,
                        target_fps=target_fps
                    )
                    chunks = preprocessor.preprocess_for_video_chunked(
                        sequence,
                        chunk_size=config.preprocessing.chunk_size,
                        source_fps=source_fps,
                        target_fps=target_fps
                    )
                    print(f"    Split into {len(chunks)} chunks")
                    prediction = model.predict_video(chunks=chunks)
                    predictions = _segments_to_frames(prediction, frame_ids, fps)
                else:
                    video, frame_ids = preprocessor.preprocess_for_video_with_ids(
                        sequence,
                        source_fps=source_fps,
                        target_fps=target_fps
                    )
                    print(f"    Video shape: {video.shape}")
                    prediction = model.predict_video(video=video)
                    predictions = _segments_to_frames(prediction, frame_ids, fps)
            else:
                samples = preprocessor.preprocess_for_single_images(sequence)
                images = [s[0] for s in samples]
                frame_ids = [s[1] for s in samples]
                predictions = []
                batch_size = max(1, config.model.batch_size)
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    batch_frame_ids = frame_ids[i:i + batch_size]
                    batch_preds = model.predict_batch(batch_images)
                    for frame_id, pred in zip(batch_frame_ids, batch_preds):
                        predictions.append({
                            'frame_id': frame_id,
                            'label': pred.get('label', 'none'),
                            'confidence': pred.get('confidence', 0.0),
                            'raw_output': pred.get('raw_output', ''),
                            'reasoning': pred.get('reasoning', '')
                        })
            
            all_predictions[sequence_key] = {
                'predictions': predictions,
                'sequence_id': sequence.sequence_id,
                'track_id': sequence.track_id,
                'actual_num_frames': sequence.num_frames
            }
        
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
    for sequence_key, data in tqdm(all_predictions.items(), desc="Post-processing"):
        result = postprocessor.process_sequence(
            data['predictions'],
            actual_num_frames=data['actual_num_frames'],
            apply_quality_control=True
        )
        result['sequence_id'] = data['sequence_id']
        result['track_id'] = data['track_id']
        result['actual_num_frames'] = data['actual_num_frames']
        processed_results[sequence_key] = result
    
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
    for sequence_key, result in processed_results.items():
        result['actual_num_frames'] = result.get('actual_num_frames', 1)
    
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
        for sequence_key in tqdm(sequences_to_viz, desc="Preparing visualizations"):
            result = processed_results[sequence_key]
            
            # Find corresponding sequence
            seq = next((s for s in dataset.sequences if f"{s.sequence_id}__track_{s.track_id}" == sequence_key), None)
            if seq:
                # Reload images for this sequence
                image_loader.load_sequence_images(seq, load_full_frames=False, show_progress=False)
                
                images = [f.crop_image for f in seq.frames if f.crop_image is not None]
                
                viz_data[sequence_key] = {
                    'images': images,
                    'predictions': result['predictions'],
                    'ground_truth': [f.true_label for f in seq.frames] if seq.has_ground_truth else None
                }
                
                # Clear after collecting
                for frame in seq.frames:
                    frame.crop_image = None
                    frame.full_image = None
        
        output_generator.create_visualizations(viz_data, sample_rate=1.0)
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
