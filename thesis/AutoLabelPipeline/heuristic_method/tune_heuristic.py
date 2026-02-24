#!/usr/bin/env python
"""
Tune heuristic parameters by grid search over threshold values.
Finds optimal activity_threshold and hazard_ratio for the test set.
"""
import sys
import argparse
from pathlib import Path
import json
import logging
import yaml
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import (
    load_dataset_from_config,
    create_image_loader,
    SequencePreprocessor
)

try:
    from .heuristic_detector import HeuristicDetector
except ImportError:
    from heuristic_detector import HeuristicDetector


class HeuristicConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        self.data = self._make_data_config(raw.get('data', {}))
        self.preprocessing = self._make_preprocessing_config(raw.get('preprocessing', {}))
        self.video_fps = raw.get('data', {}).get('video_fps', 10)
        model_kwargs = raw.get('model', {}).get('model_kwargs', {})
        self.target_video_fps = model_kwargs.get('target_video_fps', None)
    
    def _make_data_config(self, data_dict: dict):
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


def _normalize_label(label: str) -> str:
    if label is None:
        return "none"
    label = str(label).strip().lower()
    if label == "both":
        label = "hazard"
    return label if label in {"left", "right", "hazard", "none"} else "none"


def extract_features_for_sequence(
    sequence,
    image_loader,
    preprocessor,
    detector: HeuristicDetector,
    source_fps: float,
    target_fps: float,
    cached_images: Dict = None
) -> Dict:
    """
    Extract heuristic features for a single sequence without making prediction.
    Returns raw analysis values for threshold tuning.
    
    If cached_images is provided, uses those instead of loading.
    """
    if cached_images is not None and sequence.sequence_id in cached_images:
        raw_images = cached_images[sequence.sequence_id]
    else:
        image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
        
        loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
        if loaded == 0:
            return None
        
        # Get raw images
        raw_images = [f.crop_image.copy() for f in sequence.frames if f.crop_image is not None]
        
        # Clean up if not caching
        if cached_images is None:
            for frame in sequence.frames:
                frame.crop_image = None
                frame.full_image = None
    
    if not raw_images:
        return None
    
    # Extract yellow intensities
    intensities = detector.extract_yellow_intensity_series(raw_images)
    periodic_result = detector.detect_periodic_signal(intensities)
    
    # Get left/right activity
    sample_img = raw_images[0]
    left_roi = detector.detect_rear_lamp_roi(sample_img, 'left')
    right_roi = detector.detect_rear_lamp_roi(sample_img, 'right')
    
    left_intensity = detector.extract_yellow_intensity_series(raw_images, left_roi)
    right_intensity = detector.extract_yellow_intensity_series(raw_images, right_roi)
    
    left_activity = float(np.std(left_intensity))
    right_activity = float(np.std(right_intensity))
    max_activity = max(left_activity, right_activity)
    
    # Coefficient of variation (CV) = std/mean - measures relative variability
    # High CV = blinking pattern, Low CV = constant light
    left_mean = float(np.mean(left_intensity))
    right_mean = float(np.mean(right_intensity))
    left_cv = left_activity / left_mean if left_mean > 0 else 0
    right_cv = right_activity / right_mean if right_mean > 0 else 0
    max_cv = max(left_cv, right_cv)
    
    return {
        'sequence_id': sequence.sequence_id,
        'ground_truth': _normalize_label(sequence.ground_truth_label) if sequence.has_ground_truth else None,
        'num_frames': len(raw_images),
        'left_activity': left_activity,
        'right_activity': right_activity,
        'max_activity': max_activity,
        'min_activity': min(left_activity, right_activity),
        'activity_ratio': min(left_activity, right_activity) / max_activity if max_activity > 0 else 0,
        'left_cv': left_cv,
        'right_cv': right_cv,
        'max_cv': max_cv,
        'is_periodic': periodic_result['is_periodic'],
        'peak_frequency': periodic_result['peak_frequency'],
        'intensity_mean': float(np.mean(intensities)),
        'intensity_std': float(np.std(intensities))
    }


def predict_with_thresholds(
    features: Dict,
    activity_threshold: float,
    hazard_ratio: float,
    min_hazard_activity: float = 0.0,
    max_ratio_for_directional: float = 1.0,
    require_periodicity: bool = False,
    min_cv: float = 0.0  # Minimum coefficient of variation
) -> str:
    """Make prediction using given thresholds."""
    # If requiring periodicity, check first
    if require_periodicity and not features.get('is_periodic', False):
        return 'none'
    
    # Check coefficient of variation - filters out constant lights
    if features.get('max_cv', 1.0) < min_cv:
        return 'none'
    
    if features['max_activity'] < activity_threshold:
        return 'none'
    
    ratio = features['activity_ratio']
    min_activity = features.get('min_activity', min(features['left_activity'], features['right_activity']))
    
    # For hazard: need BOTH high ratio AND both sides active
    if ratio > hazard_ratio and min_activity >= min_hazard_activity:
        return 'hazard'
    # For directional: require clear asymmetry
    elif ratio < max_ratio_for_directional:
        if features['left_activity'] > features['right_activity']:
            return 'left'
        else:
            return 'right'
    # Ambiguous case
    elif features['left_activity'] > features['right_activity']:
        return 'left'
    else:
        return 'right'


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute accuracy and per-class metrics."""
    labels = ["left", "right", "hazard", "none"]
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total else 0.0
    
    f1s = []
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': support}
        f1s.append(f1)
    
    return {
        'accuracy': accuracy,
        'macro_f1': sum(f1s) / len(f1s) if f1s else 0.0,
        'per_class': per_class
    }


def tune_heuristic(
    config_path: str,
    test_sequences_file: str,
    output_dir: str = "heuristic_tuning",
    activity_thresholds: List[float] = None,
    hazard_ratios: List[float] = None,
    min_hazard_activities: List[float] = None,
    max_ratio_for_directionals: List[float] = None,
    min_cv_thresholds: List[float] = None
):
    """
    Grid search over heuristic parameters.
    """
    # Default search ranges
    if activity_thresholds is None:
        activity_thresholds = [500, 1000, 1500, 2000, 3000, 5000]
    if hazard_ratios is None:
        hazard_ratios = [0.6, 0.7, 0.8, 0.9]
    if min_hazard_activities is None:
        min_hazard_activities = [0, 500, 1000, 1500]
    if max_ratio_for_directionals is None:
        max_ratio_for_directionals = [0.3, 0.5, 0.7, 1.0]
    if min_cv_thresholds is None:
        # CV threshold: 0 = disabled, higher = require more variability
        min_cv_thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    print(f"Loading config from {config_path}")
    config = HeuristicConfig(config_path)
    
    # Load test sequences
    with open(test_sequences_file, 'r') as f:
        test_set = json.load(f)
        sequence_ids = test_set['sequence_ids']
    
    config.data.sequence_filter = sequence_ids
    config.data.max_sequences = None
    
    print(f"Loading {len(sequence_ids)} test sequences...")
    dataset = load_dataset_from_config(config.data)
    image_loader = create_image_loader(config.data, lazy=False)
    preprocessor = SequencePreprocessor(config.preprocessing)
    
    # Dummy detector for feature extraction
    detector = HeuristicDetector(fps=config.video_fps)
    
    # Extract features for all sequences
    print("Extracting features from all sequences...")
    all_features = []
    for sequence in tqdm(dataset.sequences, desc="Extracting features"):
        features = extract_features_for_sequence(
            sequence, image_loader, preprocessor, detector,
            config.video_fps, config.target_video_fps
        )
        if features:
            all_features.append(features)
    
    print(f"Extracted features from {len(all_features)} sequences")
    
    # Save raw features for analysis
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    features_file = output_path / "extracted_features.json"
    with open(features_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    print(f"Saved features to {features_file}")
    
    # Print feature statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS BY CLASS")
    print("="*60)
    
    for label in ["left", "right", "hazard", "none"]:
        class_features = [f for f in all_features if f['ground_truth'] == label]
        if class_features:
            activities = [f['max_activity'] for f in class_features]
            ratios = [f['activity_ratio'] for f in class_features]
            cvs = [f.get('max_cv', 0) for f in class_features]
            periodic_count = sum(1 for f in class_features if f.get('is_periodic', False))
            print(f"\n{label.upper()} (n={len(class_features)}):")
            print(f"  Max Activity: min={min(activities):.1f}, max={max(activities):.1f}, "
                  f"median={np.median(activities):.1f}, mean={np.mean(activities):.1f}")
            print(f"  Activity Ratio: min={min(ratios):.2f}, max={max(ratios):.2f}, "
                  f"median={np.median(ratios):.2f}")
            print(f"  Max CV (std/mean): min={min(cvs):.3f}, max={max(cvs):.3f}, "
                  f"median={np.median(cvs):.3f}")
            print(f"  Periodic (FFT): {periodic_count}/{len(class_features)} ({100*periodic_count/len(class_features):.1f}%)")
    
    # Grid search
    print("\n" + "="*60)
    print("GRID SEARCH")
    print("="*60)
    
    # Periodicity doesn't help at 5fps, so we skip it
    # Focus on CV threshold instead
    
    print(f"Parameter space:")
    print(f"  activity_thresholds: {len(activity_thresholds)} values")
    print(f"  hazard_ratios: {len(hazard_ratios)} values")
    print(f"  min_hazard_activities: {len(min_hazard_activities)} values")
    print(f"  max_ratio_for_directionals: {len(max_ratio_for_directionals)} values")
    print(f"  min_cv_thresholds: {len(min_cv_thresholds)} values")
    total = len(activity_thresholds) * len(hazard_ratios) * len(min_hazard_activities) * len(max_ratio_for_directionals) * len(min_cv_thresholds)
    print(f"  Total combinations: {total}")
    
    results = []
    best_f1 = 0
    best_params = {}
    
    y_true = [f['ground_truth'] for f in all_features]
    
    from itertools import product
    param_combos = list(product(activity_thresholds, hazard_ratios, min_hazard_activities, max_ratio_for_directionals, min_cv_thresholds))
    
    for threshold, hazard_ratio, min_hazard, max_dir_ratio, min_cv in tqdm(param_combos, desc="Grid search"):
        y_pred = [predict_with_thresholds(f, threshold, hazard_ratio, min_hazard, max_dir_ratio, False, min_cv) for f in all_features]
        metrics = compute_metrics(y_true, y_pred)
        
        results.append({
            'activity_threshold': threshold,
            'hazard_ratio': hazard_ratio,
            'min_hazard_activity': min_hazard,
            'max_ratio_for_directional': max_dir_ratio,
            'min_cv': min_cv,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'f1_left': metrics['per_class']['left']['f1'],
            'f1_right': metrics['per_class']['right']['f1'],
            'f1_hazard': metrics['per_class']['hazard']['f1'],
            'f1_none': metrics['per_class']['none']['f1'],
        })
        
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            best_params = {
                'activity_threshold': threshold,
                'hazard_ratio': hazard_ratio,
                'min_hazard_activity': min_hazard,
                'max_ratio_for_directional': max_dir_ratio,
                'min_cv': min_cv,
                'metrics': metrics
            }
    
    # Save grid search results
    results_df = pd.DataFrame(results)
    results_file = output_path / "grid_search_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved grid search results to {results_file}")
    
    # Print best results
    print("\n" + "="*60)
    print("BEST PARAMETERS")
    print("="*60)
    print(f"Activity Threshold: {best_params['activity_threshold']}")
    print(f"Hazard Ratio: {best_params['hazard_ratio']}")
    print(f"Min Hazard Activity: {best_params['min_hazard_activity']}")
    print(f"Max Ratio for Directional: {best_params['max_ratio_for_directional']}")
    print(f"Min CV (coefficient of variation): {best_params['min_cv']}")
    print(f"\nMetrics:")
    print(f"  Accuracy: {best_params['metrics']['accuracy']:.3f}")
    print(f"  Macro F1: {best_params['metrics']['macro_f1']:.3f}")
    print(f"  Per-class F1:")
    for label, m in best_params['metrics']['per_class'].items():
        print(f"    {label}: {m['f1']:.3f} (support={m['support']})")
    
    # Save best params
    best_file = output_path / "best_params.json"
    with open(best_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {best_file}")
    
    # Print top 10 configurations
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    top10 = results_df.nlargest(10, 'macro_f1')
    print(top10.to_string(index=False))
    
    return best_params


def tune_heuristic_with_hsv(
    config_path: str,
    test_sequences_file: str,
    output_dir: str = "heuristic_tuning",
    activity_thresholds: List[float] = None,
    hazard_ratios: List[float] = None,
    hue_ranges: List[Tuple[int, int]] = None,
    sat_mins: List[int] = None,
    val_mins: List[int] = None
):
    """
    Grid search over heuristic parameters INCLUDING HSV color thresholds.
    This is slower since features must be re-extracted for each HSV config.
    """
    # Default search ranges
    if activity_thresholds is None:
        activity_thresholds = [1000, 2000, 3000, 5000]
    if hazard_ratios is None:
        hazard_ratios = [0.6, 0.7, 0.8]
    if hue_ranges is None:
        # Different yellow/orange ranges to test
        hue_ranges = [
            (10, 30),   # Narrower yellow
            (15, 35),   # Default (yellow-orange)
            (10, 40),   # Wider (includes more orange)
            (5, 35),    # Includes red-orange
            (15, 45),   # Extends into green-yellow
        ]
    if sat_mins is None:
        sat_mins = [50, 80, 100, 120]  # Saturation minimum
    if val_mins is None:
        val_mins = [80, 100, 120]  # Value/brightness minimum
    
    print(f"Loading config from {config_path}")
    config = HeuristicConfig(config_path)
    
    # Load test sequences
    with open(test_sequences_file, 'r') as f:
        test_set = json.load(f)
        sequence_ids = test_set['sequence_ids']
    
    config.data.sequence_filter = sequence_ids
    config.data.max_sequences = None
    
    print(f"Loading {len(sequence_ids)} test sequences...")
    dataset = load_dataset_from_config(config.data)
    image_loader = create_image_loader(config.data, lazy=False)
    preprocessor = SequencePreprocessor(config.preprocessing)
    
    # Cache raw images for all sequences (to avoid reloading)
    print("Caching raw images for all sequences...")
    cached_images = {}
    ground_truths = {}
    
    for sequence in tqdm(dataset.sequences, desc="Loading images"):
        image_loader.load_sequence_images(sequence, load_full_frames=False, show_progress=False)
        loaded = sum(1 for f in sequence.frames if f.crop_image is not None)
        if loaded > 0:
            cached_images[sequence.sequence_id] = [
                f.crop_image.copy() for f in sequence.frames if f.crop_image is not None
            ]
            if sequence.has_ground_truth:
                ground_truths[sequence.sequence_id] = _normalize_label(sequence.ground_truth_label)
        # Clear loaded images from frames
        for frame in sequence.frames:
            frame.crop_image = None
            frame.full_image = None
    
    print(f"Cached images for {len(cached_images)} sequences")
    
    # Calculate total combinations
    total_hsv = len(hue_ranges) * len(sat_mins) * len(val_mins)
    total_thresh = len(activity_thresholds) * len(hazard_ratios)
    print(f"\nGrid search space:")
    print(f"  HSV combinations: {total_hsv}")
    print(f"  Threshold combinations: {total_thresh}")
    print(f"  Total: {total_hsv * total_thresh}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Grid search
    results = []
    best_f1 = 0
    best_params = {}
    
    for hue_min, hue_max in tqdm(hue_ranges, desc="HSV Hue ranges"):
        for sat_min in sat_mins:
            for val_min in val_mins:
                # Create detector with this HSV config
                detector = HeuristicDetector(
                    fps=config.video_fps,
                    hue_min=hue_min,
                    hue_max=hue_max,
                    sat_min=sat_min,
                    val_min=val_min
                )
                
                # Extract features for all sequences with this HSV config
                all_features = []
                for seq_id, images in cached_images.items():
                    if not images:
                        continue
                    
                    # Extract intensities
                    sample_img = images[0]
                    left_roi = detector.detect_rear_lamp_roi(sample_img, 'left')
                    right_roi = detector.detect_rear_lamp_roi(sample_img, 'right')
                    
                    left_intensity = detector.extract_yellow_intensity_series(images, left_roi)
                    right_intensity = detector.extract_yellow_intensity_series(images, right_roi)
                    
                    left_activity = float(np.std(left_intensity))
                    right_activity = float(np.std(right_intensity))
                    max_activity = max(left_activity, right_activity)
                    
                    all_features.append({
                        'sequence_id': seq_id,
                        'ground_truth': ground_truths.get(seq_id, 'none'),
                        'left_activity': left_activity,
                        'right_activity': right_activity,
                        'max_activity': max_activity,
                        'min_activity': min(left_activity, right_activity),
                        'activity_ratio': min(left_activity, right_activity) / max_activity if max_activity > 0 else 0
                    })
                
                # Sweep over activity thresholds and hazard ratios
                y_true = [f['ground_truth'] for f in all_features]
                
                for threshold in activity_thresholds:
                    for ratio in hazard_ratios:
                        y_pred = [predict_with_thresholds(f, threshold, ratio) for f in all_features]
                        metrics = compute_metrics(y_true, y_pred)
                        
                        results.append({
                            'hue_min': hue_min,
                            'hue_max': hue_max,
                            'sat_min': sat_min,
                            'val_min': val_min,
                            'activity_threshold': threshold,
                            'hazard_ratio': ratio,
                            'accuracy': metrics['accuracy'],
                            'macro_f1': metrics['macro_f1'],
                            'f1_left': metrics['per_class']['left']['f1'],
                            'f1_right': metrics['per_class']['right']['f1'],
                            'f1_hazard': metrics['per_class']['hazard']['f1'],
                            'f1_none': metrics['per_class']['none']['f1'],
                        })
                        
                        if metrics['macro_f1'] > best_f1:
                            best_f1 = metrics['macro_f1']
                            best_params = {
                                'hue_min': hue_min,
                                'hue_max': hue_max,
                                'sat_min': sat_min,
                                'sat_max': 255,
                                'val_min': val_min,
                                'val_max': 255,
                                'activity_threshold': threshold,
                                'hazard_ratio': ratio,
                                'metrics': metrics
                            }
    
    # Save grid search results
    results_df = pd.DataFrame(results)
    results_file = output_path / "grid_search_hsv_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved grid search results to {results_file}")
    
    # Print best results
    print("\n" + "="*60)
    print("BEST PARAMETERS (with HSV tuning)")
    print("="*60)
    print(f"HSV Range: H=[{best_params['hue_min']}, {best_params['hue_max']}], "
          f"S>={best_params['sat_min']}, V>={best_params['val_min']}")
    print(f"Activity Threshold: {best_params['activity_threshold']}")
    print(f"Hazard Ratio: {best_params['hazard_ratio']}")
    print(f"\nMetrics:")
    print(f"  Accuracy: {best_params['metrics']['accuracy']:.3f}")
    print(f"  Macro F1: {best_params['metrics']['macro_f1']:.3f}")
    print(f"  Per-class F1:")
    for label, m in best_params['metrics']['per_class'].items():
        print(f"    {label}: {m['f1']:.3f} (support={m['support']})")
    
    # Save best params
    best_file = output_path / "best_params_hsv.json"
    with open(best_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {best_file}")
    
    # Print top 10 configurations
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS")
    print("="*60)
    top10 = results_df.nlargest(10, 'macro_f1')
    print(top10.to_string(index=False))
    
    return best_params


def main():
    parser = argparse.ArgumentParser(description="Tune heuristic parameters")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--test-sequences", type=str, required=True, help="Test sequences JSON")
    parser.add_argument("--output-dir", type=str, default="heuristic_tuning", help="Output directory")
    parser.add_argument("--tune-hsv", action="store_true", 
                        help="Also tune HSV color thresholds (slower but more thorough)")
    
    args = parser.parse_args()
    
    if args.tune_hsv:
        tune_heuristic_with_hsv(args.config, args.test_sequences, args.output_dir)
    else:
        tune_heuristic(args.config, args.test_sequences, args.output_dir)


if __name__ == "__main__":
    main()
