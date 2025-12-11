import os
import json
import base64
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from scipy.fft import fft, fftfreq
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
        # 'periodic_analysis': periodic_result,
        'left_activity': float(left_activity),
        'right_activity': float(right_activity)
    }

# ------------------------------------------------------------------
# API HELPERS

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_image_grid(image_paths: List[str], max_cols: int = 5) -> Image.Image:
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    img_width, img_height = images[0].size
    n_images = len(images)
    n_cols = min(max_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    grid = Image.new('RGB', (n_cols * img_width, n_rows * img_height), 'white')
    
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        grid.paste(img, (col * img_width, row * img_height))
    
    return grid


def create_annotated_grid(image_paths: List[str], 
                         computational_result: dict,
                         max_cols: int = 4) -> Image.Image:
    grid = create_image_grid(image_paths, max_cols)
    draw = ImageDraw.Draw(grid)
    
    # Add text annotation
    text = f"Computational Analysis:\n"
    text += f"Predicted: {computational_result.get('predicted_signal', 'unknown')}\n"
    
    if 'periodic_analysis' in computational_result:
        pa = computational_result['periodic_analysis']
        text += f"Periodic: {pa.get('is_periodic', False)}\n"
        text += f"Frequency: {pa.get('peak_frequency', 0):.2f} Hz\n"
    
    # Draw text box
    try:
        font = ImageFont.truetype("/Users/elizabethdwenger/Desktop/2025Projects/Fonts_GT_Super/Desktop/GT-Eesti/GT-Eesti-Display-Light-Trial.otf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), text, fill='yellow', font=font, stroke_width=2, stroke_fill='black')
    
    return grid


# API LABELING METHODS

def label_with_api_grid(image_paths: List[str], 
                       model: str = "gpt-4o-mini",
                       use_computational_hint: bool = True) -> Dict:
    
    # Get computational analysis if requested
    comp_result = None
    if use_computational_hint:
        comp_result = analyze_sequence_computational(image_paths)
    
    # Create grid
    if use_computational_hint and comp_result:
        grid = create_annotated_grid(image_paths, comp_result)
    else:
        grid = create_image_grid(image_paths)
    
    base64_grid = encode_pil_to_base64(grid)
    
    prompt = f"""Analyze this grid of {len(image_paths)} sequential car images (left-to-right, top-to-bottom).

Task: Determine turn signal status:
- "left": Left turn signal blinking
- "right": Right turn signal blinking  
- "hazard": Both signals blinking
- "none": No turn signals active

Key points:
- Turn signals BLINK (on/off pattern across frames)
- Look for amber/orange lights
- Hazard = both sides blink together
"""
    
    if use_computational_hint and comp_result:
        prompt += f"""

COMPUTATIONAL ANALYSIS (use as guide):
- Predicted: {comp_result.get('predicted_signal')}
- Periodic detected: {comp_result.get('periodic_analysis', {}).get('is_periodic', False)}

Use your visual analysis to verify or correct this prediction.
"""
    
    prompt += """
Respond with JSON only:
{
  "label": "left"|"right"|"hazard"|"none",
  "reasoning": "brief explanation",
}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_grid}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=500,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        result['method'] = 'api_grid'
        result['used_computational_hint'] = use_computational_hint
        
        if comp_result:
            result['computational_prediction'] = comp_result.get('predicted_signal')
            result['agrees_with_computational'] = (
                result['label'] == comp_result.get('predicted_signal')
            )
        
        return result
        
    except Exception as e:
        return {'label': 'error', 'error': str(e), 'method': 'api_grid'}


# BATCH PROCESSING

def process_all_sequences(
    label_csv: str,
    image_base_path: str = '../seq_img',
    method: str = 'computational',  # 'computational', 'api', 'api_with_hint'
    model: str = 'gpt-4o-mini',
    output_path: str = 'results.json',
    max_sequences: Optional[int] = None,
    delay: float = 1.0
) -> pd.DataFrame:
    
    df = pd.read_csv(label_csv)
    
    # Prepend base path
    if image_base_path:
        df['crop_path'] = image_base_path + df['crop_path']
    
    results = []
    sequences = df.groupby('sequence_id')
    
    if max_sequences:
        sequences = list(sequences)[:max_sequences]
    
    total = len(sequences) if isinstance(sequences, list) else len(df['sequence_id'].unique())
    
    for idx, (seq_id, group) in enumerate(sequences if isinstance(sequences, list) else sequences):
        print(f"\n[{idx+1}/{total}] {seq_id}")
        
        group = group.sort_values('frame_idx')
        image_paths = group['crop_path'].tolist()#[:4]  # First 4 frames
        true_label = group['label'].iloc[0]
        
        if not Path(image_paths[0]).exists():
            print(f"  WARNING: Image not found, skipping")
            continue
        
        print(f"  True: {true_label}, Frames: {len(image_paths)}")
        
        result_entry = {
            'sequence_id': seq_id,
            'true_label': true_label,
            'num_frames': len(image_paths),
            'frame_paths': image_paths
        }
        
        # Process based on method
        if method == 'computational':
            result = analyze_sequence_computational(image_paths)
            result_entry['result'] = result
            pred = result.get('predicted_signal', 'error')
            
        elif method == 'api':
            result = label_with_api_grid(image_paths, model, use_computational_hint=False)
            result_entry['result'] = result
            pred = result.get('label', 'error')
            time.sleep(delay)
            
        elif method == 'api_with_hint':
            result = label_with_api_grid(image_paths, model, use_computational_hint=True)
            result_entry['result'] = result
            pred = result.get('label', 'error')
            time.sleep(delay)
        
        is_correct = pred == true_label
        result_entry['correct'] = is_correct
        
        print(f"  Predicted: {pred}")
        
        results.append(result_entry)
        
        # Save intermediate
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return pd.DataFrame(results)


def evaluate_results(results_df: pd.DataFrame) -> Dict:    
    def get_prediction(row):
        result = row.get('result', {})
        if isinstance(result, dict):
            return result.get('predicted_signal') or result.get('label', 'error')
        return 'error'
    
    predictions = results_df.apply(get_prediction, axis=1)
    true_labels = results_df['true_label']
    
    accuracy = (predictions == true_labels).sum() / len(results_df)
    
    print(f"\nAccuracy: {accuracy:.1%} ({(predictions == true_labels).sum()}/{len(results_df)})")
    print("\nConfusion Matrix:")
    print(pd.crosstab(true_labels, predictions, rownames=['True'], colnames=['Predicted']))
    
    return {
        'accuracy': float(accuracy),
        'correct': int((predictions == true_labels).sum()),
        'total': len(results_df),
        'confusion_matrix': pd.crosstab(true_labels, predictions).to_dict()
    }