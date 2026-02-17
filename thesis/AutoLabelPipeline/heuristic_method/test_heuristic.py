#!/usr/bin/env python
"""
Heuristic-based turn signal detection pipeline runner.

Produces outputs compatible with compare_prompts.py for direct comparison
with VLM-based models.
"""
import sys
import argparse
from pathlib import Path
import json
import logging
import yaml
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Optional
import os

# Add src to path for data loading utilities
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)
from data.data_structures import Dataset
from data.csv_loader import CSVLoader

# Handle both module import and direct script execution
try:
    from .heuristic_detector import HeuristicDetector
except ImportError:
    from heuristic_detector import HeuristicDetector


class HeuristicConfig:
    """
    Lightweight config loader for heuristic method.
    Only loads data and preprocessing sections, avoiding model requirements.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        self.data = self._make_data_config(raw.get('data', {}))
        self.preprocessing = self._make_preprocessing_config(raw.get('preprocessing', {}))
        
        # Extract video_fps for heuristic use
        self.video_fps = raw.get('data', {}).get('video_fps', 10)
        
        # Try to get target_video_fps from model_kwargs if present
        model_kwargs = raw.get('model', {}).get('model_kwargs', {})
        self.target_video_fps = model_kwargs.get('target_video_fps', None)
    
    def _make_data_config(self, data_dict: dict):
        """Create a simple namespace for data config."""
        class DataConfig:
            pass
        cfg = DataConfig()
        cfg.input_csv = data_dict.get('input_csv', '')
        cfg.crop_base_dir = data_dict.get('crop_base_dir', '')
        cfg.frame_base_dir = data_dict.get('frame_base_dir', None)
        cfg.max_sequences = data_dict.get('max_sequences', None)
        cfg.sequence_filter = data_dict.get('sequence_filter', None)
        cfg.video_fps = data_dict.get('video_fps', 10)
        return cfg
    
    def _make_preprocessing_config(self, preproc_dict: dict):
        """Create a simple namespace for preprocessing config."""
        class PreprocessingConfig:
            pass
        cfg = PreprocessingConfig()
        cfg.resize_resolution = preproc_dict.get('resize_resolution', [640, 480])
        cfg.normalize = preproc_dict.get('normalize', True)
        cfg.maintain_aspect_ratio = preproc_dict.get('maintain_aspect_ratio', True)
        cfg.padding_color = preproc_dict.get('padding_color', [0, 0, 0])
        cfg.max_sequence_length = preproc_dict.get('max_sequence_length', None)
        cfg.sequence_stride = preproc_dict.get('sequence_stride', 1)
        cfg.enable_chunking = preproc_dict.get('enable_chunking', True)
        cfg.chunk_size = preproc_dict.get('chunk_size', 50)
        return cfg


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def _normalize_label(label: str) -> str:
    """Normalize label to standard format."""
    if label is None:
        return "none"
    label = str(label).strip().lower()
    if label == "both":
        label = "hazard"
    return label if label in {"left", "right", "hazard", "none"} else "none"


def _get_frame_ids_for_video(
    sequence,
    preprocessor,
    use_crops: bool = True,
    apply_max_length: bool = True,
    source_fps: float = None,
    target_fps: float = None
) -> List[int]:
    """Get frame IDs after preprocessing (stride, max_length, resampling)."""
    import numpy as np
    
    if use_crops:
        frames = [f for f in sequence.frames if f.crop_image is not None]
    else:
        frames = [f for f in sequence.frames if f.full_image is not None]
    
    frames = preprocessor._maybe_resample_frames(frames, source_fps, target_fps)
    
    if preprocessor.stride > 1:
        frames = frames[::preprocessor.stride]
    
    if apply_max_length and preprocessor.max_length and len(frames) > preprocessor.max_length:
        indices = np.linspace(0, len(frames) - 1, preprocessor.max_length, dtype=int)
        frames = [frames[i] for i in indices]
    
    return [f.frame_id for f in frames]


def _segments_to_frames(
    segment_prediction: Dict,
    frame_ids: List[int],
    fps: float
) -> List[Dict]:
    """
    Convert segment-based prediction to per-frame predictions.
    Compatible with compare_prompts._segments_to_frames.
    """
    num_frames = len(frame_ids)
    predictions = []
    segments = segment_prediction.get('segments', [])
    
    if not segments:
        label = _normalize_label(segment_prediction.get('label', 'none'))
        for frame_id in frame_ids:
            predictions.append({
                'frame_id': frame_id,
                'label': label,
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
        return predictions
    
    for seg in segments:
        label = _normalize_label(seg.get('label', 'none'))
        start = seg.get('start_frame', 0)
        end = seg.get('end_frame', num_frames - 1)
        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))
        start_frame_id = frame_ids[start]
        end_frame_id = frame_ids[end]
        
        for idx in range(start, end + 1):
            predictions.append({
                'frame_id': frame_ids[idx],
                'label': label,
                'start_frame': start_frame_id,
                'end_frame': end_frame_id,
                'start_time_seconds': round(start_frame_id / fps, 2),
                'end_time_seconds': round(end_frame_id / fps, 2),
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
    
    # Sort and deduplicate
    predictions.sort(key=lambda x: x['frame_id'])
    seen = set()
    unique_predictions = []
    for pred in predictions:
        if pred['frame_id'] not in seen:
            unique_predictions.append(pred)
            seen.add(pred['frame_id'])
    
    # Fill gaps with 'none'
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
                    'reasoning': 'Gap filled'
                })
        return complete_predictions
    
    return unique_predictions


def _compute_frame_metrics(y_true: List[str], y_pred: List[str], labels=None) -> Dict:
    """
    Compute frame-level metrics.
    Compatible with compare_prompts._compute_frame_metrics.
    """
    if labels is None:
        labels = ["left", "right", "hazard", "none"]
    
    results = {"per_class": {}, "macro_f1": 0.0, "accuracy": 0.0}
    if not y_true:
        return results
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    results["accuracy"] = correct / total if total else 0.0
    
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = tp / support if support > 0 else 0.0
        results["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "accuracy": accuracy
        }
        f1s.append(f1)
    
    results["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return results


def _extract_events(labels: List[str]) -> List[Dict]:
    """Extract contiguous events from label sequence."""
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
    
    # Filter out 'none' events
    return [e for e in events if e["label"] != "none"]


def _event_iou(a: Dict, b: Dict) -> float:
    """Compute IoU between two events."""
    overlap_start = max(a["start"], b["start"])
    overlap_end = min(a["end"], b["end"])
    overlap = max(0, overlap_end - overlap_start + 1)
    union = (a["end"] - a["start"] + 1) + (b["end"] - b["start"] + 1) - overlap
    return overlap / union if union > 0 else 0.0


def _compute_event_metrics(
    y_true_labels: List[str],
    y_pred_labels: List[str],
    fps: float,
    iou_threshold: float = 0.5,
    tolerance_seconds: float = 0.5
) -> Dict:
    """
    Compute event-level metrics.
    Compatible with compare_prompts._compute_event_metrics.
    """
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


def test_heuristic(
    config_path: str,
    test_sequences_file: str,
    output_dir: str = "prompt_comparison",
    verbose: bool = False,
    fps: float = None,
    activity_threshold: float = 6200.0,
    hazard_ratio_threshold: float = 0.7
) -> Dict:
    """
    Run heuristic detector on test sequences.
    
    Produces output format compatible with compare_prompts.py.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Load lightweight config (only data and preprocessing, no model required)
    config = HeuristicConfig(config_path)
    
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    
    # Determine effective FPS
    target_fps = config.target_video_fps
    effective_fps = target_fps or config.video_fps
    
    # Override FPS if provided
    if fps is not None:
        effective_fps = fps
    
    # Load test sequence IDs
    with open(test_sequences_file, 'r') as f:
        test_set = json.load(f)
        sequence_ids = test_set['sequence_ids']
    
    # Override config for test sequences
    config.data.sequence_filter = sequence_ids
    config.data.max_sequences = None
    
    print(f"\nTesting heuristic method")
    print(f"Test sequences: {len(sequence_ids)}")
    print(f"FPS: {effective_fps}")
    print(f"Activity threshold: {activity_threshold}")
    print(f"Hazard ratio threshold: {hazard_ratio_threshold}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "heuristic"
    
    run_output_dir = Path(output_dir) / f"{model_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_output_dir}")
    
    # Load dataset
    dataset = load_dataset_from_config(config.data)
    print(f"Loaded {dataset.num_sequences} sequences")
    
    # Setup
    image_loader = create_image_loader(config.data, lazy=False)
    preprocessor = SequencePreprocessor(config.preprocessing)
    
    # Create heuristic detector
    detector = HeuristicDetector(
        fps=effective_fps,
        activity_threshold=activity_threshold,
        hazard_ratio_threshold=hazard_ratio_threshold
    )
    
    # Run inference
    results = []
    all_y_true = []
    all_y_pred = []
    event_metrics_accum = {"tp": 0, "fp": 0, "fn": 0}
    
    for sequence in tqdm(dataset.sequences, desc="Processing"):
        # Load images
        image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
        
        loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
        if loaded == 0:
            logger.warning(f"No images loaded for {sequence.sequence_id}")
            continue
        
        try:
            # Get frame IDs after applying stride/max_length (but NOT preprocessing)
            source_fps = config.data.video_fps
            frame_ids = _get_frame_ids_for_video(
                sequence,
                preprocessor,
                use_crops=True,
                apply_max_length=True,
                source_fps=source_fps,
                target_fps=target_fps
            )
            
            # For event metrics, account for temporal downsampling from stride
            stride = max(1, preprocessor.stride)
            event_fps = effective_fps / stride
            
            # Get RAW images (not normalized) for heuristic analysis
            # The heuristic needs uint8 RGB images, not normalized float tensors
            frame_id_set = set(frame_ids)
            raw_images = []
            actual_frame_ids = []
            for f in sequence.frames:
                if f.crop_image is not None and f.frame_id in frame_id_set:
                    # crop_image is already loaded as uint8 RGB by image_loader
                    raw_images.append(f.crop_image.copy())
                    actual_frame_ids.append(f.frame_id)
            
            # Sort by frame_id to maintain temporal order
            sorted_pairs = sorted(zip(actual_frame_ids, raw_images), key=lambda x: x[0])
            actual_frame_ids = [p[0] for p in sorted_pairs]
            raw_images = [p[1] for p in sorted_pairs]
            
            # Run heuristic detection on RAW images
            prediction = detector.predict_sequence(raw_images)
            
            # Convert to frame predictions for metrics
            frame_preds = _segments_to_frames(prediction, actual_frame_ids, effective_fps)
            
            # Compute metrics if ground truth available
            if sequence.has_ground_truth:
                true_by_id = {
                    f.frame_id: _normalize_label(f.true_label)
                    for f in sequence.frames if f.crop_image is not None
                }
                y_true = [true_by_id.get(fid, "none") for fid in actual_frame_ids]
                y_pred = [_normalize_label(p["label"]) for p in frame_preds]
                
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                # Event metrics
                ev = _compute_event_metrics(y_true, y_pred, event_fps)
                event_metrics_accum["tp"] += ev["tp"]
                event_metrics_accum["fp"] += ev["fp"]
                event_metrics_accum["fn"] += ev["fn"]
            
            # Store result
            results.append({
                'sequence_id': sequence.sequence_id,
                'num_frames': sequence.num_frames,
                'ground_truth': sequence.ground_truth_label if sequence.has_ground_truth else None,
                'prediction': prediction,
                'mode': 'video'
            })
        
        except Exception as e:
            logger.error(f"Error processing {sequence.sequence_id}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clear images to save memory
            if hasattr(image_loader, 'clear_sequence'):
                image_loader.clear_sequence(sequence)
            else:
                for frame in sequence.frames:
                    frame.crop_image = None
                    frame.full_image = None
    
    # Get metrics from detector
    metrics = detector.get_metrics()
    
    # Calculate sequence-level accuracy
    accuracy = None
    if any(r['ground_truth'] for r in results):
        correct = sum(
            1 for r in results
            if r['ground_truth'] and _normalize_label(r['prediction']['label']) == _normalize_label(r['ground_truth'])
        )
        total = sum(1 for r in results if r['ground_truth'])
        accuracy = correct / total if total > 0 else 0
    
    # Compute frame metrics
    frame_metrics = _compute_frame_metrics(all_y_true, all_y_pred) if all_y_true else None
    
    # Compute event metrics
    event_metrics = None
    if event_metrics_accum["tp"] + event_metrics_accum["fp"] + event_metrics_accum["fn"] > 0:
        tp = event_metrics_accum["tp"]
        fp = event_metrics_accum["fp"]
        fn = event_metrics_accum["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        event_metrics = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    
    # Save results (compatible with compare_prompts.py)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full results in run directory
    result_file = run_output_dir / "results.json"
    output = {
        'prompt_file': 'heuristic_method',
        'test_set': test_sequences_file,
        'timestamp': timestamp,
        'model': model_name,
        'inference_mode': 'video',
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'frame_metrics': frame_metrics,
        'event_metrics': event_metrics,
        'results': results,
        'heuristic_params': {
            'fps': effective_fps,
            'activity_threshold': activity_threshold,
            'hazard_ratio_threshold': hazard_ratio_threshold
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Summary file in main output_dir for easy comparison
    summary_file = output_path / f"{model_name}_{timestamp}_summary.json"
    summary = {
        'prompt_file': 'heuristic_method',
        'timestamp': timestamp,
        'model': model_name,
        'inference_mode': 'video',
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'run_directory': str(run_output_dir),
        'frame_metrics': frame_metrics,
        'event_metrics': event_metrics
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print results
    print(f"\n Results saved to {result_file}")
    print(f" Summary saved to {summary_file}")
    print(f"\nPerformance:")
    if accuracy is not None:
        print(f"  Sequence Accuracy: {accuracy:.1%}")
    else:
        print("  Sequence Accuracy: N/A (no ground truth)")
    if frame_metrics:
        print(f"  Frame Macro F1: {frame_metrics['macro_f1']:.3f}")
        print(f"  Frame Accuracy: {frame_metrics['accuracy']:.3f}")
        print(f"  Per-class F1:")
        for label in ["left", "right", "hazard", "none"]:
            f1 = frame_metrics['per_class'].get(label, {}).get('f1', 0)
            support = frame_metrics['per_class'].get(label, {}).get('support', 0)
            print(f"    {label}: {f1:.3f} (support={support})")
    if event_metrics:
        print(f"  Event F1: {event_metrics['f1']:.3f}")
        print(f"  Event Precision: {event_metrics['precision']:.3f}")
        print(f"  Event Recall: {event_metrics['recall']:.3f}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Parse Success: {metrics['parse_success_rate']:.1%}")
    
    return output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test heuristic turn signal detector on sequences"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--test-sequences",
        type=str,
        required=True,
        help="JSON file with test sequence_ids"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prompt_comparison",
        help="Output directory for results (default: prompt_comparison)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS for detection (default: from config)"
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=6200.0,
        help="Activity threshold for signal detection (default: 6200)"
    )
    parser.add_argument(
        "--hazard-ratio",
        type=float,
        default=0.7,
        help="Ratio threshold for hazard detection (default: 0.7)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    test_heuristic(
        config_path=args.config,
        test_sequences_file=args.test_sequences,
        output_dir=args.output_dir,
        verbose=args.verbose,
        fps=args.fps,
        activity_threshold=args.activity_threshold,
        hazard_ratio_threshold=args.hazard_ratio
    )


if __name__ == "__main__":
    main()
