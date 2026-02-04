#!/usr/bin/env python
"""
Prompt Comparison Tool - Test multiple prompts on same sequences.

Helps you:
1. Compare different prompt formulations
2. A/B test prompt changes
3. Track prompt performance over iterations
"""
import sys
import argparse
from pathlib import Path
import json
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config, set_random_seeds
from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)
from models import load_model


def _normalize_label(label: str) -> str:
    if label is None:
        return "none"
    label = str(label).strip().lower()
    return label if label in {"left", "right", "both", "none"} else "none"

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


def _segments_to_frames(segment_prediction, frame_ids, fps: float):
    """
    Convert segment-based prediction to per-frame predictions aligned to frame_ids.
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
                'confidence': segment_prediction.get('confidence', 0.0),
                'raw_output': segment_prediction.get('raw_output', ''),
                'reasoning': segment_prediction.get('reasoning', '')
            })
        return predictions
    
    for seg in segments:
        label = _normalize_label(seg.get('label', 'none'))
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


def _compute_frame_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = ["left", "right", "both", "none"]
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


def test_prompt(config_path: str, prompt_file: str, test_sequences_file: str,
                output_dir: str = "prompt_comparison"):
    """
    Test a single prompt on test sequences.
    """
    # Load config
    config = load_config(config_path)
    set_random_seeds(config.experiment.random_seed)
    target_fps = config.model.model_kwargs.get('target_video_fps')
    config.model.model_kwargs['video_fps'] = target_fps or config.data.video_fps
    
    # Load test sequence IDs
    with open(test_sequences_file, 'r') as f:
        test_set = json.load(f)
        sequence_ids = test_set['sequence_ids']
    
    # Override config
    config.data.sequence_filter = sequence_ids
    config.data.max_sequences = None
    config.model.prompt_template_path = prompt_file
    
    print(f"\nTesting prompt: {prompt_file}")
    print(f"Test sequences: {len(sequence_ids)}")
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_name = Path(prompt_file).stem
    model_name = config.model.type.value
    
    run_output_dir = Path(output_dir) / f"{model_name}_{prompt_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_output_dir}")
    
    # Load dataset
    dataset = load_dataset_from_config(config.data)
    print(f"Loaded {dataset.num_sequences} sequences")
    
    # Setup
    image_loader = create_image_loader(config.data, lazy=False)
    preprocessor = SequencePreprocessor(config.preprocessing)
    
    # Load model
    print("Loading model...")
    model = load_model(config.model, warmup=True)
    
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
            continue
        
        try:
            # Predict
            target_fps = config.model.model_kwargs.get('target_video_fps')
            source_fps = config.data.video_fps
            effective_fps = target_fps or source_fps
            
            if config.model.inference_mode.value == 'video':
                if (config.preprocessing.enable_chunking and
                    loaded > config.preprocessing.chunk_size):
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
                    prediction = model.predict_video(chunks=chunks)
                else:
                    video, frame_ids = preprocessor.preprocess_for_video_with_ids(
                        sequence,
                        source_fps=source_fps,
                        target_fps=target_fps
                    )
                    prediction = model.predict_video(video)
                predictions = [prediction]
                
                # Frame-level labels for metrics
                frame_preds = _segments_to_frames(prediction, frame_ids, effective_fps)
                if sequence.has_ground_truth:
                    true_labels = [_normalize_label(f.true_label) for f in sequence.frames if f.crop_image is not None]
                    # Align true labels to sampled frame_ids
                    true_by_id = {f.frame_id: _normalize_label(f.true_label) for f in sequence.frames if f.crop_image is not None}
                    y_true = [true_by_id.get(fid, "none") for fid in frame_ids]
                    y_pred = [_normalize_label(p["label"]) for p in frame_preds]
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
                    ev = _compute_event_metrics(y_true, y_pred, effective_fps)
                    event_metrics_accum["tp"] += ev["tp"]
                    event_metrics_accum["fp"] += ev["fp"]
                    event_metrics_accum["fn"] += ev["fn"]
            else:
                samples = preprocessor.preprocess_for_single_images(sequence)
                images = [s[0] for s in samples]
                predictions = model.predict_batch(images)
                
                if sequence.has_ground_truth:
                    frame_ids = [s[1] for s in samples]
                    true_by_id = {f.frame_id: _normalize_label(f.true_label) for f in sequence.frames if f.crop_image is not None}
                    y_true = [true_by_id.get(fid, "none") for fid in frame_ids]
                    y_pred = [_normalize_label(p["label"]) for p in predictions]
                    all_y_true.extend(y_true)
                    all_y_pred.extend(y_pred)
        
        # Store result
        results.append({
                'sequence_id': sequence.sequence_id,
                'num_frames': sequence.num_frames,
                'ground_truth': sequence.ground_truth_label if sequence.has_ground_truth else None,
                'prediction': predictions[0] if config.model.inference_mode.value == 'video' else predictions,
                'mode': config.model.inference_mode.value
            })
        
        except Exception as e:
            logging.error(f"Error processing {sequence.sequence_id}: {e}")
        
        finally:
            if hasattr(image_loader, 'clear_sequence'):
                image_loader.clear_sequence(sequence)
            else:
                for frame in sequence.frames:
                    frame.crop_image = None
                    frame.full_image = None
    
    # Get metrics
    metrics = model.get_metrics()
    
    # Calculate accuracy
    accuracy = None
    if any(r['ground_truth'] for r in results):
        if config.model.inference_mode.value == 'video':
            correct = sum(1 for r in results
                         if r['ground_truth'] and r['prediction']['label'] == r['ground_truth'])
            total = sum(1 for r in results if r['ground_truth'])
            accuracy = correct / total if total > 0 else 0
        else:
            # Majority vote for single-image
            correct = 0
            total = 0
            for r in results:
                if r['ground_truth']:
                    from collections import Counter
                    labels = [p['label'] for p in r['prediction']]
                    most_common = Counter(labels).most_common(1)[0][0]
                    if most_common == r['ground_truth']:
                        correct += 1
                    total += 1
            accuracy = correct / total if total > 0 else 0
    
    # Compute additional metrics if ground truth exists
    frame_metrics = _compute_frame_metrics(all_y_true, all_y_pred) if all_y_true else None
    event_metrics = None
    if event_metrics_accum["tp"] + event_metrics_accum["fp"] + event_metrics_accum["fn"] > 0:
        tp = event_metrics_accum["tp"]
        fp = event_metrics_accum["fp"]
        fn = event_metrics_accum["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        event_metrics = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in the run-specific directory
    result_file = run_output_dir / "results.json"
    
    output = {
        'prompt_file': prompt_file,
        'test_set': test_sequences_file,
        'timestamp': timestamp,
        'model': model_name,
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'frame_metrics': frame_metrics,
        'event_metrics': event_metrics,
        'results': results
    }
    
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Also save summary to main output_dir for easy comparison
    summary_file = output_path / f"{model_name}_{prompt_name}_{timestamp}_summary.json"
    summary = {
        'prompt_file': prompt_file,
        'timestamp': timestamp,
        'model': model_name,
        'num_sequences': len(results),
        'accuracy': accuracy,
        'metrics': metrics,
        'run_directory': str(run_output_dir),
        'frame_metrics': frame_metrics,
        'event_metrics': event_metrics
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n Results saved to {result_file}")
    print(f" Summary saved to {summary_file}")
    print(f"\nPerformance:")
    print(f"  Accuracy: {accuracy:.1%}" if accuracy else "  Accuracy: N/A (no ground truth)")
    if frame_metrics:
        print(f"  Frame Macro F1: {frame_metrics['macro_f1']:.3f}")
        print(f"  Frame Accuracy: {frame_metrics['accuracy']:.3f}")
    if event_metrics:
        print(f"  Event F1: {event_metrics['f1']:.3f}")
    print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Parse Success: {metrics['parse_success_rate']:.1%}")
    
    return output


def compare_prompts(comparison_dir: str):
    """
    Compare multiple prompt test results.
    """
    result_files = list(Path(comparison_dir).glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {comparison_dir}")
        return
    
    print(f"\nComparing {len(result_files)} prompt results:")
    print("="*80)
    
    comparison_data = []
    
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        comparison_data.append({
            'Prompt': Path(data['prompt_file']).stem,
            'Timestamp': data['timestamp'],
            'Sequences': data['num_sequences'],
            'Accuracy': f"{data['accuracy']:.1%}" if data['accuracy'] else "N/A",
            'Frame Macro F1': f"{data.get('frame_metrics', {}).get('macro_f1', 0):.3f}" if data.get('frame_metrics') else "N/A",
            'Event F1': f"{data.get('event_metrics', {}).get('f1', 0):.3f}" if data.get('event_metrics') else "N/A",
            'Avg Latency (ms)': f"{data['metrics']['avg_latency_ms']:.1f}",
            'Parse Success': f"{data['metrics']['parse_success_rate']:.1%}"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Save comparison
    comparison_file = Path(comparison_dir) / "comparison_summary.csv"
    df.to_csv(comparison_file, index=False)
    print(f"\n Comparison saved to {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare different prompts')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a prompt')
    test_parser.add_argument('--config', type=str, required=True,
                            help='Configuration file')
    test_parser.add_argument('--prompt', type=str, required=True,
                            help='Prompt file to test')
    test_parser.add_argument('--test-set', type=str, required=True,
                            help='Test sequences JSON file')
    test_parser.add_argument('--output-dir', type=str, default='prompt_comparison',
                            help='Output directory')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare prompt results')
    compare_parser.add_argument('--dir', type=str, required=True,
                               help='Directory with result files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'test':
        test_prompt(
            args.config,
            args.prompt,
            args.test_set,
            args.output_dir
        )
    
    elif args.command == 'compare':
        compare_prompts(args.dir)


if __name__ == '__main__':
    main()
