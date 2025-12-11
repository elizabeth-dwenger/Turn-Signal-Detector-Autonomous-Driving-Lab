import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.fft import fft, fftfreq

# COMPUTATIONAL ANALYSIS

def isolate_yellow_channel(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    return mask


def extract_yellow_intensity_series(image_paths: List[str],
                                    roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    intensities = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            intensities.append(0)
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if roi is not None:
            x1, y1, x2, y2 = roi
            img = img[y1:y2, x1:x2]
            if img.size == 0:
                intensities.append(0)
                continue
        
        mask = isolate_yellow_channel(img)
        intensities.append(int(mask.sum()))
    
    return np.array(intensities)


def detect_periodic_signal(intensities: np.ndarray, fps: float = 5.0) -> dict:
    # Detect periodic blinking using Fourier Transforms (FFT)
    # can detect periodic blinking in a sequence of light-intensity values
    n = len(intensities)
    intensities_norm = intensities - np.mean(intensities)
    
    freqs = fftfreq(n, 1/fps)
    fft_vals = np.abs(fft(intensities_norm))
    
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    
    # Turn signals blink at 1-2.5 Hz
    freq_mask = (freqs >= 1.0) & (freqs <= 2.5)
    
    if np.any(freq_mask):
        peak_idx = np.argmax(fft_vals[freq_mask])
        peak_freq = freqs[freq_mask][peak_idx]
        peak_power = fft_vals[freq_mask][peak_idx]
        mean_power = np.mean(fft_vals)
        is_periodic = peak_power > 3 * mean_power
    else:
        peak_freq = 0
        peak_power = 0
        is_periodic = False
    
    return {
        'is_periodic': bool(is_periodic),
        'peak_frequency': peak_freq,
        'blinks_per_minute': peak_freq * 60
    }


def detect_rear_lamp_roi(image: np.ndarray, side: str) -> Tuple[int, int, int, int]:
    h, w = image.shape[:2]
    y1, y2 = int(h * 0.4), h
    
    if side == 'left':
        x1, x2 = 0, int(w * 0.4)
    elif side == 'right':
        x1, x2 = int(w * 0.6), w
    else:
        x1, x2 = 0, w
    
    return (x1, y1, x2, y2)


def analyze_sequence_computational(image_paths: List[str], fps: float = 5.0) -> dict:
    # Overall yellow intensity
    intensities = extract_yellow_intensity_series(image_paths)
    periodic_result = detect_periodic_signal(intensities, fps)
    
    # Check if blinking detected
    is_blinking = (
        periodic_result['is_periodic'] or
        np.std(intensities) > np.mean(intensities) * 0.05
    )
    
    if not is_blinking:
        return {'predicted_signal': 'none', 'periodic_analysis': periodic_result}
    
    # Analyze left vs right
    img_sample = cv2.imread(image_paths[0])
    img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
    
    left_roi = detect_rear_lamp_roi(img_sample, 'left')
    right_roi = detect_rear_lamp_roi(img_sample, 'right')
    
    left_intensity = extract_yellow_intensity_series(image_paths, left_roi)
    right_intensity = extract_yellow_intensity_series(image_paths, right_roi)
    
    left_activity = np.std(left_intensity)
    right_activity = np.std(right_intensity)
    
    # Minimum activity threshold
    max_activity = max(left_activity, right_activity)
    if max_activity < 6200:  # Tuned this threshold (over-tuned?)
        return {'predicted_signal': 'none', 'periodic_analysis': periodic_result}
    
    # Determine signal type
    ratio = min(left_activity, right_activity) / max_activity
    
    if ratio > 0.7:  # Both sides blinking similarly
        predicted = 'hazard'
    elif left_activity > right_activity:
        predicted = 'left'
    else:
        predicted = 'right'
    
    return {
        'predicted_signal': predicted,
        'left_activity': float(left_activity),
        'right_activity': float(right_activity)
    }


# BATCH PROCESSING

def process_all_sequences(
    label_csv: str = '../jupyter/filtered_1010.csv',
    output_path: str = 'heuristic_results.json',
    max_sequences: Optional[int] = None,
    sample_every_n: int = 4
) -> pd.DataFrame:
    """
    Process sequences in blocks of size `sample_every_n` using computational method.
    Each block can have its own true label.
    """
    
    df = pd.read_csv(label_csv)
    df = df.rename(columns={'predicted_label': 'true_label'})  # rename for clarity
    
    results = []
    sequences = df.groupby('sequence_id')
    
    if max_sequences:
        sequences = list(sequences)[:max_sequences]
    
    total = len(sequences) if isinstance(sequences, list) else len(df['sequence_id'].unique())
    
    for idx, (seq_id, group) in enumerate(sequences if isinstance(sequences, list) else sequences):
        print(f"\n[{idx+1}/{total}] {seq_id}")
        
        group = group.sort_values('frame_id')
        
        # Split sequence into blocks of sample_every_n frames
        for start_idx in range(0, len(group), sample_every_n):
            block = group.iloc[start_idx:start_idx+sample_every_n]
            image_paths = block['crop_path'].tolist()
            true_label = block['true_label'].iloc[0]  # label for this block
            
            if not Path(image_paths[0]).exists():
                print(f"  WARNING: Image not found at {image_paths[0]}, skipping")
                continue
            
            result_entry = {
                'sequence_id': str(seq_id),
                'true_label': str(true_label),
                'start_frame': int(block['frame_id'].iloc[0]),
                'num_frames': int(len(image_paths)),
                'total_frames': int(len(group)),
                'frame_paths': [str(p) for p in image_paths]
            }
            
            # Run computational analysis
            try:
                result = analyze_sequence_computational(image_paths)
                result_entry['result'] = {
                    k: (float(v) if isinstance(v, np.generic) else v)
                    for k, v in result.items()
                }
                pred = result.get('predicted_signal', 'error')
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                result_entry['result'] = {'error': str(e)}
                pred = 'error'
            
            result_entry['correct'] = pred == true_label
            print(f"  True: {true_label}, Predicted: {pred} {'✓' if result_entry['correct'] else '✗'}")
            
            results.append(result_entry)
            
            # Save intermediate results
            with open(output_path, 'w') as f:
                json.dump(json.loads(pd.DataFrame(results).to_json(orient='records')), f, indent=2)
    
    return pd.DataFrame(results)


def evaluate_results(results_df: pd.DataFrame) -> Dict:
    """
    Calculate accuracy and confusion matrix from results.
    """
    def get_prediction(row):
        result = row.get('result', {})
        if isinstance(result, dict):
            return result.get('predicted_signal', 'error')
        return 'error'
    
    predictions = results_df.apply(get_prediction, axis=1)
    true_labels = results_df['true_label']
    
    # Remove errors from evaluation
    valid_mask = predictions != 'error'
    predictions_valid = predictions[valid_mask]
    true_labels_valid = true_labels[valid_mask]
    
    if len(predictions_valid) == 0:
        print("No valid predictions to evaluate!")
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'errors': len(predictions)
        }
    
    accuracy = (predictions_valid == true_labels_valid).sum() / len(predictions_valid)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.1%} ({(predictions_valid == true_labels_valid).sum()}/{len(predictions_valid)})")
    print(f"Errors/Skipped: {(predictions == 'error').sum()}")
    print(f"\nConfusion Matrix:")
    confusion = pd.crosstab(true_labels_valid, predictions_valid, rownames=['True'], colnames=['Predicted'])
    print(confusion)
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for label in confusion.index:
        class_correct = confusion.loc[label, label] if label in confusion.columns else 0
        class_total = confusion.loc[label].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {label}: {class_acc:.1%} ({class_correct}/{class_total})")
    
    return {
        'accuracy': float(accuracy),
        'correct': int((predictions_valid == true_labels_valid).sum()),
        'total': len(predictions_valid),
        'errors': int((predictions == 'error').sum()),
        'confusion_matrix': confusion.to_dict()
    }


if __name__ == '__main__':
    # Process all sequences
    print("Processing sequences with computational method...")
    results_df = process_all_sequences(
        label_csv='../jupyter/filtered_1010.csv',
        output_path='heuristic_results.json',
        max_sequences=None,  # Set to a number to test on subset
        sample_every_n=4  # Use every 4th frame
    )
    
    # Evaluate results
    print("\n" + "="*60)
    evaluation = evaluate_results(results_df)
    
    # Save evaluation metrics
    with open('heuristic_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"\nResults saved to: heuristic_results.json")
    print(f"Evaluation saved to: heuristic_evaluation.json")
