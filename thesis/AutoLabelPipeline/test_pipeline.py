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
from typing import List, Tuple, Dict

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


def setup_logging(verbose=False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def _normalize_label(label: str) -> str:
    if label is None:
        return "none"
    label = str(label).strip().lower()
    return label if label in {"left", "right", "both", "none"} else "none"

def _compute_frame_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = ["left", "right", "both", "none"]
    results = {"per_class": {}, "macro_f1": 0.0, "accuracy": 0.0, "support": 0}
    if not y_true:
        return results
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    results["accuracy"] = correct / total if total else 0.0
    results["support"] = total
    
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label)
        }
        f1s.append(f1)
    
    results["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return results

def _extract_events(labels):
    events = []
    if not labels:
        return events
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            events.append({"label": current, "start": start, "end": i - 1})
            current = labels[i]
            start = i
    events.append({"label": current, "start": start, "end": len(labels) - 1})
    return [e for e in events if e["label"] != "none"]

def _event_iou(a, b):
    overlap_start = max(a["start"], b["start"])
    overlap_end = min(a["end"], b["end"])
    overlap = max(0, overlap_end - overlap_start + 1)
    union = (a["end"] - a["start"] + 1) + (b["end"] - b["start"] + 1) - overlap
    return overlap / union if union > 0 else 0.0

def _compute_event_metrics(y_true_labels, y_pred_labels, fps: float,
                           iou_threshold: float = 0.5, tolerance_seconds: float = 0.5):
    tol_frames = max(1, int(round(tolerance_seconds * fps)))
    gt_events = _extract_events(y_true_labels)
    pred_events = _extract_events(y_pred_labels)
    
    matched_gt = set()
    tp = 0
    for pred in pred_events:
        best_idx = None
        for i, gt in enumerate(gt_events):
            if i in matched_gt or gt["label"] != pred["label"]:
                continue
            iou = _event_iou(pred, gt)
            close = (abs(pred["start"] - gt["start"]) <= tol_frames and
                     abs(pred["end"] - gt["end"]) <= tol_frames)
            if iou >= iou_threshold or close:
                best_idx = i
                break
        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def segments_to_frames(segment_prediction: Dict, frame_ids: List[int], fps: float) -> List[Dict]:
    """
    Convert segment-based prediction to per-frame predictions.
    
    Args:
        segment_prediction: Prediction with 'segments' array from model
        num_frames: Total number of frames in sequence
    
    Returns:
        List of per-frame prediction dicts
    """
    frame_predictions = []
    num_frames = len(frame_ids)
    segments = segment_prediction.get('segments', [])
    
    if not segments:
        for frame_id in frame_ids:
            frame_predictions.append({
                'frame_id': frame_id,
                'label': _normalize_label(segment_prediction.get('label', 'none')),
                'confidence': segment_prediction.get('confidence', 0.0),
                'raw_output': segment_prediction.get('raw_output', '')
            })
        return frame_predictions
    
    # Build frame-level predictions from segments
    for seg in segments:
        label = _normalize_label(seg.get('label', 'none'))
        start = seg.get('start_frame', 0)
        end = seg.get('end_frame', num_frames - 1)
        confidence = seg.get('confidence', 0.5)
        
        # Ensure we don't exceed num_frames
        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))
        start_frame_id = frame_ids[start]
        end_frame_id = frame_ids[end]
        
        for frame_idx in range(start, end + 1):
            frame_predictions.append({
                'frame_id': frame_ids[frame_idx],
                'label': label,
                'confidence': confidence,
                'start_frame': start_frame_id,
                'end_frame': end_frame_id,
                'start_time_seconds': round(start_frame_id / fps, 2),
                'end_time_seconds': round(end_frame_id / fps, 2),
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
    
    # Sort by frame_id and remove duplicates (keep first occurrence)
    frame_predictions.sort(key=lambda x: x['frame_id'])
    
    # Remove duplicate frame_ids (shouldn't happen with valid segments, but be safe)
    seen_frames = set()
    unique_predictions = []
    for pred in frame_predictions:
        if pred['frame_id'] not in seen_frames:
            unique_predictions.append(pred)
            seen_frames.add(pred['frame_id'])
    
    # Fill any gaps with 'none'
    if len(unique_predictions) < num_frames:
        # Create a complete frame list
        complete_predictions = []
        pred_dict = {p['frame_id']: p for p in unique_predictions}
        
        for frame_id in frame_ids:
            if frame_id in pred_dict:
                complete_predictions.append(pred_dict[frame_id])
            else:
                # Gap - fill with 'none'
                complete_predictions.append({
                    'frame_id': frame_id,
                    'label': 'none',
                    'confidence': 0.5,
                    'raw_output': segment_prediction.get('raw_output', ''),
                    'reasoning': 'Gap filled'
                })
        
        return complete_predictions
    
    return unique_predictions


def _get_frame_ids_for_video(sequence, preprocessor, use_crops: bool = True,
                             apply_max_length: bool = True,
                             source_fps: float = None,
                             target_fps: float = None):
    if use_crops:
        frames = [f for f in sequence.frames if f.crop_image is not None]
    else:
        frames = [f for f in sequence.frames if f.full_image is not None]
    
    frames = preprocessor._maybe_resample_frames(frames, source_fps, target_fps)
    
    if preprocessor.stride > 1:
        frames = frames[::preprocessor.stride]
    
    if apply_max_length and preprocessor.max_length and len(frames) > preprocessor.max_length:
        import numpy as np
        indices = np.linspace(0, len(frames) - 1, preprocessor.max_length, dtype=int)
        frames = [frames[i] for i in indices]
    
    return [f.frame_id for f in frames]


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
    print("Step 1: Starting test pipeline...")
    
    # Load configuration
    print("Step 2: Loading configuration...")
    try:
        config = load_config(config_path)
        set_random_seeds(config.experiment.random_seed)
        target_fps = config.model.model_kwargs.get('target_video_fps')
        config.model.model_kwargs['video_fps'] = target_fps or config.data.video_fps
        print("Step 3: Configuration loaded successfully!")
    except Exception as e:
        print(f" Failed to load configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nConfiguration: {config_path}")
    print(f"  Experiment: {config.experiment.name}")
    print(f"  Model: {config.model.type.value}")
    print(f"  Mode: {config.model.inference_mode.value}")
    
    # Verify critical paths
    print("\nStep 4: Verifying paths...")
    print(f"  Model path: {config.model.model_name_or_path}")
    print(f"  Model exists: {Path(config.model.model_name_or_path).exists()}")
    print(f"  CSV path: {config.data.input_csv}")
    print(f"  CSV exists: {Path(config.data.input_csv).exists()}")
    print(f"  Crop dir: {config.data.crop_base_dir}")
    print(f"  Crop dir exists: {Path(config.data.crop_base_dir).exists()}")
    
    # Check GPU
    print("\nStep 5: Checking GPU availability...")
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Override to use small subset
    if sequence_ids:
        config.data.sequence_filter = sequence_ids
        config.data.max_sequences = None
        print(f"\n  Testing on specific sequences: {sequence_ids}")
    else:
        config.data.max_sequences = num_sequences
        print(f"\n  Testing on first {num_sequences} sequences")
    
    # Create timestamped test output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.type.value
    test_output_dir = Path(config.experiment.output_dir) / "test_runs" / f"{model_name}_{timestamp}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    config.experiment.output_dir = str(test_output_dir)
    # Ensure visualizations go under this run
    config.output.visualization_output_dir = str(test_output_dir / "visualizations")
    
    # Always enable visualizations for testing
    config.output.save_visualizations = True
    config.output.visualization_sample_rate = 1.0  # Visualize all test sequences
    
    print(f"  Output: {config.experiment.output_dir}")
    print(f"  Timestamp: {timestamp}")
    
    # Stage 1: Load dataset
    print("\n" + "-"*80)
    print("STAGE 1: Data Loading")
    print("-"*80)
    
    try:
        dataset = load_dataset_from_config(config.data)
        print(f"  Loaded {dataset.num_sequences} sequences ({dataset.total_frames} frames)")
    except Exception as e:
        print(f" Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if dataset.num_sequences == 0:
        print("\n No sequences loaded! Check your sequence_filter or data path.")
        return
    
    # Stage 2: Load images (LAZY LOADING to save memory)
    print("\n" + "-"*80)
    print("STAGE 2: Image Loading Setup")
    print("-"*80)
    print("  Setting up image loader (will load per-sequence)...")
    
    try:
        # Use regular loader but we'll call it per-sequence to save memory
        image_loader = create_image_loader(config.data, lazy=False)
        print("  Image loader ready")
    except Exception as e:
        print(f" Failed to create image loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Stage 3: Load model
    print("\n" + "-"*80)
    print("STAGE 3: Model Loading")
    print("-"*80)
    
    try:
        model = load_model(config.model, warmup=True)
        print(f"  Model loaded and ready")
    except Exception as e:
        print(f" Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Stage 4: Preprocessing setup
    print("\n" + "-"*80)
    print("STAGE 4: Preprocessing Setup")
    print("-"*80)
    
    try:
        preprocessor = SequencePreprocessor(config.preprocessing)
        print(f"  Target size: {config.preprocessing.resize_resolution}")
        print(f"  Normalize: {config.preprocessing.normalize}")
        print(f"  Chunking enabled: {config.preprocessing.enable_chunking}")
        if config.preprocessing.enable_chunking:
            print(f"  Chunk size: {config.preprocessing.chunk_size} frames")
    except Exception as e:
        print(f" Failed to create preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Stage 5: Inference (process each sequence individually)
    print("\n" + "-"*80)
    print("STAGE 5: Inference (Memory-Efficient)")
    print("-"*80)
    
    all_predictions = {}
    sequence_lookup = {}
    
    for i, sequence in enumerate(tqdm(dataset.sequences, desc="Processing sequences")):
        print(f"\n  Sequence {i+1}/{dataset.num_sequences}: {sequence.sequence_id}")
        print(f"    Frames: {sequence.num_frames}")
        sequence_key = f"{sequence.sequence_id}__track_{sequence.track_id}"
        sequence_lookup[sequence_key] = sequence
        
        # Load images for this sequence only
        try:
            image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
            
            loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
            print(f"    Loaded: {loaded}/{sequence.num_frames} images")
            
            if loaded == 0:
                print(f"     No images loaded, skipping sequence")
                continue
        except Exception as e:
            print(f"     Error loading images: {e}")
            continue
        
        try:
            if config.model.inference_mode.value == 'video':
                target_fps = config.model.model_kwargs.get('target_video_fps')
                source_fps = config.data.video_fps
                fps = target_fps or source_fps
                # Video mode - returns segment-based predictions
                if (config.preprocessing.enable_chunking and 
                    loaded > config.preprocessing.chunk_size):
                    print(f"    Sequence is long ({loaded} frames), using chunked inference...")
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
                    segment_prediction = model.predict_video(chunks=chunks)
                else:
                    video, frame_ids = preprocessor.preprocess_for_video_with_ids(
                        sequence,
                        source_fps=source_fps,
                        target_fps=target_fps
                    )
                    print(f"    Video shape: {video.shape}")
                    segment_prediction = model.predict_video(video=video)
                
                # Convert segment-based prediction to per-frame predictions
                print(f"    Converting segments to frames...")
                predictions = segments_to_frames(segment_prediction, frame_ids, fps)
                print(f"    Generated {len(predictions)} frame predictions")
                
            else:
                # Single-image mode - already returns per-frame predictions
                samples = preprocessor.preprocess_for_single_images(sequence)
                images = [s[0] for s in samples]
                frame_ids = [s[1] for s in samples]
                print(f"    Processing {len(images)} frames...")
                
                raw_predictions = model.predict_batch(images)
                
                # Add frame_ids to predictions
                predictions = []
                for frame_id, pred in zip(frame_ids, raw_predictions):
                    pred['frame_id'] = frame_id
                    predictions.append(pred)
            
            all_predictions[sequence_key] = predictions
            print(f"    Processed successfully")
            
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
    
    try:
        postprocessor = create_postprocessor(config)
        print("  Postprocessor created")
    except Exception as e:
        print(f" Failed to create postprocessor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    processed_results = {}
    for sequence_key, predictions in tqdm(all_predictions.items(), desc="Post-processing"):
        try:
            # Get actual sequence length
            seq = sequence_lookup.get(sequence_key)
            actual_num_frames = seq.num_frames if seq else len(predictions)
            
            result = postprocessor.process_sequence(
                predictions, 
                actual_num_frames=actual_num_frames,
                apply_quality_control=True
            )
            processed_results[sequence_key] = result
        except Exception as e:
            print(f"   Error post-processing {sequence_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Post-processing complete")
    
    # Stage 7: Output generation
    print("\n" + "-"*80)
    print("STAGE 7: Output Generation")
    print("-"*80)
    
    try:
        output_generator = create_output_generator(config)
        
        # Save predictions
        summary = output_generator.save_dataset_predictions(processed_results)
        print(f"  Saved predictions")
        print(f"  Output directory: {config.experiment.output_dir}")
        
    except Exception as e:
        print(f" Failed to save outputs: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Also save individual sequence JSON files for easy notebook loading
    sequences_dir = Path(config.experiment.output_dir) / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    
    for sequence_key, result in processed_results.items():
        try:
            seq = sequence_lookup.get(sequence_key)
            if seq:
                base_id = seq.sequence_id
                track_id = seq.track_id
            else:
                base_id = sequence_key
                track_id = None
            safe_id = base_id.replace('/', '_').replace('\\', '_')
            if track_id is not None:
                safe_id = f"{safe_id}__track_{track_id}"
            seq_file = sequences_dir / f"{safe_id}.json"
            
            # Find ground truth from dataset
            gt_labels = [f.true_label for f in seq.frames] if seq and seq.has_ground_truth else None
            
            # Get actual sequence length
            actual_num_frames = seq.num_frames if seq else len(gt_labels) if gt_labels else len(result['predictions'])
            
            with open(seq_file, 'w') as f:
                json.dump({
                    'sequence_id': base_id,
                    'track_id': track_id,
                    'num_frames': actual_num_frames, 
                    'num_predictions': len(result['predictions']),
                    'predictions': result['predictions'],
                    'ground_truth_labels': gt_labels,
                    'ground_truth_sequence': seq.ground_truth_label if seq and seq.has_ground_truth else None,
                    'quality_report': result.get('quality_report'),
                    'stats': result.get('stats')
                }, f, indent=2, default=str)
        except Exception as e:
            print(f"   Warning: Failed to save {sequence_key}: {e}")
            continue
    
    print(f"  Saved {len(processed_results)} individual sequence files to {sequences_dir}")
    
    # Create visualizations
    print(f"\n  Creating visualizations...")
    
    try:
        # We need to reload images for visualization
        viz_data = {}
        for sequence_key, result in processed_results.items():
            # Find corresponding sequence
            seq = sequence_lookup.get(sequence_key)
            if seq:
                # Reload images for this sequence
                image_loader.load_sequence_images(seq, load_full_frames=False, show_progress=False)
                
                # Collect images
                frames_with_images = [f for f in seq.frames if f.crop_image is not None]
                images = [f.crop_image for f in frames_with_images]
                frame_ids = [f.frame_id for f in frames_with_images]
                gt_labels = [f.true_label for f in frames_with_images] if seq.has_ground_truth else None
                
                viz_data[sequence_key] = {
                    'images': images,
                    'frame_ids': frame_ids,
                    'predictions': result['predictions'],
                    'ground_truth': gt_labels
                }
                
                # Clear after visualization
                for frame in seq.frames:
                    frame.crop_image = None
                    frame.full_image = None
        
        output_generator.create_visualizations(viz_data)
        print(f"  Visualizations created")
    except Exception as e:
        print(f"   Warning: Failed to create visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate report
    try:
        dataset_stats = {
            'total_sequences': len(processed_results),
            'total_frames': sum(r.get('quality_report', {}).get('total_frames', 0) 
                               for r in processed_results.values()),
            'total_flagged': sum(r.get('quality_report', {}).get('total_flagged', 0)
                                for r in processed_results.values()),
            'label_distribution': {}
        }
        
        for result in processed_results.values():
            for label, count in result.get('stats', {}).get('label_distribution', {}).items():
                dataset_stats['label_distribution'][label] = \
                    dataset_stats['label_distribution'].get(label, 0) + count
        
        report_path = output_generator.generate_report(dataset_stats, model_metrics)
        print(f"  Report: {report_path}")
    except Exception as e:
        print(f"   Warning: Failed to generate report: {e}")

    # Evaluation metrics vs ground truth (if available)
    print("\n" + "-"*80)
    print("STAGE 8: Evaluation Metrics (Ground Truth)")
    print("-"*80)
    
    target_fps = config.model.model_kwargs.get('target_video_fps')
    eval_fps = target_fps or config.data.video_fps
    all_true = []
    all_pred = []
    event_tp = event_fp = event_fn = 0
    per_sequence_rows = []
    
    for sequence_key, result in processed_results.items():
        seq = sequence_lookup.get(sequence_key)
        if not seq or not seq.has_ground_truth:
            continue
        
        pred_by_id = {p.get('frame_id'): _normalize_label(p.get('label', 'none'))
                      for p in result.get('predictions', []) if p.get('frame_id') is not None}
        
        y_true = []
        y_pred = []
        for f in seq.frames:
            if not f.has_ground_truth:
                continue
            if f.frame_id not in pred_by_id:
                continue
            y_true.append(_normalize_label(f.true_label))
            y_pred.append(pred_by_id[f.frame_id])
        
        if not y_true:
            continue
        
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        ev = _compute_event_metrics(y_true, y_pred, eval_fps)
        event_tp += ev["tp"]
        event_fp += ev["fp"]
        event_fn += ev["fn"]
        
        seq_frame_metrics = _compute_frame_metrics(y_true, y_pred)
        per_sequence_rows.append({
            "sequence_key": sequence_key,
            "support": seq_frame_metrics["support"],
            "frame_accuracy": seq_frame_metrics["accuracy"],
            "frame_macro_f1": seq_frame_metrics["macro_f1"],
            "event_f1": ev["f1"],
            "event_precision": ev["precision"],
            "event_recall": ev["recall"]
        })
    
    if all_true:
        frame_metrics = _compute_frame_metrics(all_true, all_pred)
        precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0.0
        recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0.0
        event_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        event_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": event_f1,
            "tp": event_tp,
            "fp": event_fp,
            "fn": event_fn
        }
        
        eval_summary = {
            "frame_metrics": frame_metrics,
            "event_metrics": event_metrics,
            "num_sequences": len(per_sequence_rows),
            "num_frames": frame_metrics.get("support", 0),
            "fps": eval_fps
        }
        
        eval_path = Path(config.experiment.output_dir) / "evaluation_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"  Saved evaluation metrics: {eval_path}")
        
        if per_sequence_rows:
            import csv
            per_seq_path = Path(config.experiment.output_dir) / "evaluation_per_sequence.csv"
            with open(per_seq_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_sequence_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_sequence_rows)
            print(f"  Saved per-sequence metrics: {per_seq_path}")
        
        print(f"  Frame Macro F1: {frame_metrics['macro_f1']:.3f}")
        print(f"  Frame Accuracy: {frame_metrics['accuracy']:.3f}")
        print(f"  Event F1: {event_metrics['f1']:.3f}")
    else:
        print("  No ground truth available for evaluation.")
    
    # Final summary
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nResults Summary:")
    print(f"  Sequences: {dataset_stats['total_sequences']}")
    print(f"  Frames: {dataset_stats['total_frames']}")
    print(f"  Flagged: {dataset_stats['total_flagged']}")
    print(f"  Label distribution: {dataset_stats['label_distribution']}")
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
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
