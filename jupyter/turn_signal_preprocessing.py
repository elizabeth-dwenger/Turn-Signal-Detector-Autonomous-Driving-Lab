import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd

def compute_frame_differences(image_paths: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute pixel-wise differences between consecutive frames.
    Ensures all frames have identical shapes by center-cropping
    to the smallest size among the loaded frames.
    """
    valid_frames = []

    # 1. Load frames safely
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read image: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        valid_frames.append(img)

    # If no frames or only 1 frame, return gracefully
    if len(valid_frames) < 2:
        return [], np.array([])

    # 2. Compute smallest frame size (Hmin, Wmin)
    heights = [f.shape[0] for f in valid_frames]
    widths =  [f.shape[1] for f in valid_frames]

    Hmin = min(heights)
    Wmin = min(widths)

    # 3. Center-crop all frames to (Hmin, Wmin)
    cropped_frames = []
    for f in valid_frames:
        h, w = f.shape[:2]

        top = (h - Hmin) // 2
        left = (w - Wmin) // 2

        cropped = f[top:top+Hmin, left:left+Wmin]
        cropped_frames.append(cropped)

    # 4. Compute differences safely (all shapes match now)
    diff_images = []
    diff_magnitudes = []

    for i in range(1, len(cropped_frames)):
        diff = cv2.absdiff(cropped_frames[i], cropped_frames[i-1])
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        magnitude = np.sum(diff_gray)

        diff_images.append(diff)
        diff_magnitudes.append(magnitude)

    return diff_images, np.array(diff_magnitudes)

def isolate_yellow_channel(image: np.ndarray, 
                           lower_hsv: Tuple[int, int, int] = (15, 100, 100),
                           upper_hsv: Tuple[int, int, int] = (35, 255, 255)) -> np.ndarray:
    """
    Isolate yellow/amber regions (turn signal color).
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create mask for yellow
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    return mask


def extract_yellow_intensity_series(image_paths: List[str], 
                                    roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    intensities = []
    
    for path in image_paths:
        img = cv2.imread(path)
        
        # If image failed to load
        if img is None:
            intensities.append(0)
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply ROI safely
        if roi is not None:
            x1, y1, x2, y2 = roi
            cropped = img[y1:y2, x1:x2]
            
            # Empty or invalid crop → fallback to 0 intensity
            if cropped.size == 0:
                intensities.append(0)
                continue
            
            img = cropped
        
        # Get yellow mask
        mask = isolate_yellow_channel(img)
        
        # Sum intensity
        intensities.append(int(mask.sum()))
    
    return np.array(intensities)



def detect_periodic_signal(intensities: np.ndarray, 
                          fps: float = 5.0,
                          expected_freq_range: Tuple[float, float] = (1.0, 2.5)) -> dict:
    n = len(intensities)
    
    # Normalize and remove DC component
    intensities_norm = intensities - np.mean(intensities)
    
    # Apply FFT
    freqs = fftfreq(n, 1/fps)
    fft_vals = np.abs(fft(intensities_norm))
    
    # Only look at positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    
    # Find peak in expected frequency range
    freq_mask = (freqs >= expected_freq_range[0]) & (freqs <= expected_freq_range[1])
    if np.any(freq_mask):
        peak_idx = np.argmax(fft_vals[freq_mask])
        peak_freq = freqs[freq_mask][peak_idx]
        peak_power = fft_vals[freq_mask][peak_idx]
        
        # Check if peak is significant (simple threshold)
        mean_power = np.mean(fft_vals)
        is_periodic = peak_power > 3 * mean_power
    else:
        peak_freq = 0
        peak_power = 0
        is_periodic = False
    
    return {
        'is_periodic': is_periodic,
        'peak_frequency': peak_freq,
        'peak_power': peak_power,
        'blinks_per_minute': peak_freq * 60,
        'frequencies': freqs,
        'fft_values': fft_vals
    }

def visualize_yellow_intensity_analysis(image_paths, roi=None, fps=5.0, figsize=(16, 12)):
    intensities = extract_yellow_intensity_series(image_paths, roi)
    periodic = detect_periodic_signal(intensities, fps)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    axes[0].plot(intensities, 'o-', linewidth=2, color='orange')
    axes[0].set_title("Raw Yellow Intensity")

    smoothed = np.convolve(intensities, np.ones(3)/3, mode='valid')
    axes[1].plot(smoothed, linewidth=2, color='red')
    axes[1].set_title("Smoothed Intensity")

    axes[2].plot(periodic['frequencies'], 
                periodic['fft_values'], 
                linewidth=2)
    axes[2].axvline(periodic['peak_frequency'], 
                   color='red', linestyle='--', 
                   label=f"Peak: {periodic['peak_frequency']:.2f} Hz ({periodic['blinks_per_minute']:.0f} bpm)")
    axes[2].axvspan(1.0, 2.5, alpha=0.2, color='green', label='Expected range (1-2.5 Hz)')
    axes[2].set_title("FFT")

    plt.tight_layout()
    return fig, periodic



def enhance_image(image: np.ndarray, 
                 gamma: float = 2.0,
                 use_clahe: bool = True,
                 brightness_threshold: int = 150) -> np.ndarray:
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # 1. Gamma correction - emphasize bright areas
    img_gamma = np.power(img_float, 1/gamma)
    
    # Convert back to uint8 for CLAHE
    img_gamma_uint8 = (img_gamma * 255).astype(np.uint8)
    
    # 2. CLAHE on luminance channel
    if use_clahe:
        # Convert to LAB
        lab = cv2.cvtColor(img_gamma_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        img_enhanced = img_gamma_uint8
    
    # 3. Highlight very bright areas (likely turn signals)
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    bright_mask = gray > brightness_threshold
    
    # Boost brightness in these areas
    img_enhanced = img_enhanced.astype(np.float32)
    img_enhanced[bright_mask] = np.clip(img_enhanced[bright_mask] * 1.3, 0, 255)
    img_enhanced = img_enhanced.astype(np.uint8)
    
    return img_enhanced


def create_enhanced_sequence(image_paths: List[str], 
                            gamma: float = 2.0,
                            isolate_yellow: bool = True) -> List[np.ndarray]:
    enhanced_images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Enhance
        enhanced = enhance_image(img, gamma=gamma)
        
        # Optionally overlay yellow mask
        if isolate_yellow:
            yellow_mask = isolate_yellow_channel(enhanced)
            # Create colored overlay
            overlay = enhanced.copy()
            overlay[yellow_mask > 0] = [255, 255, 0]  # Highlight yellow regions
            enhanced = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)
        
        enhanced_images.append(enhanced)
    
    return enhanced_images


def visualize_enhancement_comparison(image_paths: List[str], figsize=(18, 12)):
    """Compare original vs enhanced images side by side."""
    n_samples = min(6, len(image_paths))
    indices = np.linspace(0, len(image_paths)-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    for col, idx in enumerate(indices):
        # Original
        img_orig = cv2.imread(image_paths[idx])
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(img_orig)
        axes[0, col].set_title(f'Original {idx}', fontsize=9)
        axes[0, col].axis('off')
        
        # Enhanced
        img_enhanced = enhance_image(img_orig, gamma=2.0)
        axes[1, col].imshow(img_enhanced)
        axes[1, col].set_title(f'Enhanced {idx}', fontsize=9)
        axes[1, col].axis('off')
        
        # Yellow isolation
        yellow_mask = isolate_yellow_channel(img_enhanced)
        axes[2, col].imshow(yellow_mask, cmap='hot')
        axes[2, col].set_title(f'Yellow Only {idx}', fontsize=9)
        axes[2, col].axis('off')
    
    axes[0, 0].text(-0.3, 0.5, 'Original', transform=axes[0, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Enhanced', transform=axes[1, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.3, 0.5, 'Yellow\nIsolation', transform=axes[2, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    return fig


def detect_rear_lamp_roi(image: np.ndarray, side: str = 'both') -> Tuple[int, int, int, int]:
    h, w = image.shape[:2]
    
    # Simple heuristic: rear lamps are usually in bottom 60% and outer 40% of image
    y1 = int(h * 0.4)
    y2 = h
    
    if side == 'left':
        x1 = 0
        x2 = int(w * 0.4)
    elif side == 'right':
        x1 = int(w * 0.6)
        x2 = w
    else:  # both
        x1 = 0
        x2 = w
    
    return (x1, y1, x2, y2)


def visualize_roi(image_path, roi):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = roi

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img)
    rect = Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', linewidth=3, facecolor='none')
    axes[0].add_patch(rect)
    axes[0].axis('off')

    axes[1].imshow(img[y1:y2, x1:x2])
    axes[1].axis('off')

    plt.tight_layout()
    return fig

def valid_roi(roi, img_shape):
    """
    Validate ROI is inside image boundaries.
    """
    if roi is None:
        return False
    x1, y1, x2, y2 = roi
    h, w = img_shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def analyze_sequence_for_turn_signals(image_paths: List[str],
                                     fps: float = 5.0,
                                     roi: Optional[Tuple[int, int, int, int]] = None,
                                     visualize: bool = True) -> dict:
    """
    Complete analysis pipeline combining all techniques.
    """
    results = {
        'num_frames': len(image_paths),
        'fps': fps
    }
    
    # 1. Frame difference analysis
    _, diff_magnitudes = compute_frame_differences(image_paths)
    diff_variance = np.var(diff_magnitudes)
    results['frame_difference_variance'] = float(diff_variance)
    results['has_motion'] = diff_variance > np.mean(diff_magnitudes) * 0.5
    
    # 2. Yellow intensity analysis
    intensities = extract_yellow_intensity_series(image_paths, roi)
    periodic_result = detect_periodic_signal(intensities, fps)
    results['periodic_analysis'] = periodic_result
    results['yellow_intensity_mean'] = float(np.mean(intensities))
    results['yellow_intensity_std'] = float(np.std(intensities))
    
    # 3. Determine likely signal type
    is_blinking = (
        periodic_result['is_periodic'] or
        periodic_result['peak_power'] > 1e5 or
        results['yellow_intensity_std'] > results['yellow_intensity_mean'] * 0.05
    )
    
    if is_blinking:
        if roi is None:
            img_sample = cv2.imread(image_paths[0])
            img_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2RGB)
    
            left_roi = detect_rear_lamp_roi(img_sample, 'left')
            right_roi = detect_rear_lamp_roi(img_sample, 'right')
    
            if not valid_roi(left_roi, img_sample.shape):
                left_roi = None
            if not valid_roi(right_roi, img_sample.shape):
                right_roi = None
    
            left_intensity = extract_yellow_intensity_series(image_paths, left_roi)
            right_intensity = extract_yellow_intensity_series(image_paths, right_roi)
    
            left_activity = np.std(left_intensity)
            right_activity = np.std(right_intensity)
    
            # New minimum activity check
            max_activity = max(left_activity, right_activity)
            min_activity = 6200  # tune based on your data
            if max_activity < min_activity:
                results['predicted_signal'] = 'none'
            else:
                threshold = 0.7
                ratio = min(left_activity, right_activity) / max_activity
                if ratio > threshold:
                    results['predicted_signal'] = 'hazard'
                elif left_activity > right_activity:
                    results['predicted_signal'] = 'left'
                else:
                    results['predicted_signal'] = 'right'
    else:
        results['predicted_signal'] = 'none'
    
    if visualize:
        visualize_yellow_intensity_analysis(image_paths, roi, fps)
        plt.show()
    
    return results


def preprocess_all_images_in_folder(
    input_folder: str,
    output_folder: str,
    preprocess_mode: str = "enhanced",  # "enhanced", "yellow", or "both"
    gamma: float = 2.0,
    skip_existing: bool = True,
    preserve_structure: bool = True
):
    """
    Preprocess all images in a folder and save to output folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    total_images = len(image_files)
    print(f"Found {total_images} images in {input_folder}")
    print(f"Preprocessing mode: {preprocess_mode}")
    print(f"Output folder: {output_folder}")
    print("-" * 60)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for idx, img_path in enumerate(image_files):
        try:
            # Determine output path(s)
            if preserve_structure:
                # Maintain folder structure relative to input_folder
                relative_path = img_path.relative_to(input_path)
            else:
                # Flatten structure
                relative_path = Path(img_path.name)
            
            # Create output paths based on mode
            if preprocess_mode == "both":
                output_enhanced = output_path / "enhanced" / relative_path
                output_yellow = output_path / "yellow" / relative_path
                output_paths = [
                    ("enhanced", output_enhanced),
                    ("yellow", output_yellow)
                ]
            elif preprocess_mode == "enhanced":
                output_enhanced = output_path / relative_path
                output_paths = [("enhanced", output_enhanced)]
            else:  # yellow
                output_yellow = output_path / relative_path
                output_paths = [("yellow", output_yellow)]
            
            # Check if we should skip
            if skip_existing and all(out_path.exists() for _, out_path in output_paths):
                skipped += 1
                if idx % 100 == 0:
                    print(f"[{idx+1}/{total_images}] Skipping (exists): {img_path.name}")
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}")
                errors += 1
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process and save based on mode
            for mode, out_path in output_paths:
                # Create output directory
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                if mode == "enhanced":
                    # Enhanced preprocessing
                    img_processed = enhance_image(img_rgb, gamma=gamma)
                    img_save = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
                    
                elif mode == "yellow":
                    # Yellow isolation as heatmap
                    yellow_mask = isolate_yellow_channel(img_rgb)
                    img_save = cv2.applyColorMap(yellow_mask, cv2.COLORMAP_HOT)
                
                # Save
                cv2.imwrite(str(out_path), img_save)
            
            processed += 1
            
            # Progress update
            if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
                print(f"[{idx+1}/{total_images}] Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            errors += 1
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Total images: {total_images}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    
    return {
        'total': total_images,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'output_folder': str(output_path)
    }


def create_preprocessed_label_csvs(
    original_label_csv: str,
    preprocessed_folder: str,
    output_prefix: str = "preprocessed_"
):
    df = pd.read_csv(original_label_csv)
    preprocessed_path = Path(preprocessed_folder)
    output_csvs = {}
    
    # Check which preprocessing modes exist
    modes = []
    if (preprocessed_path / "enhanced").exists():
        modes.append("enhanced")
    if (preprocessed_path / "yellow").exists():
        modes.append("yellow")
    if len(modes) == 0 and preprocessed_path.exists():
        # Assume single mode in root
        modes.append("default")
    
    for mode in modes:
        df_copy = df.copy()
        
        # Update crop_path to point to preprocessed images
        if mode == "default":
            df_copy['crop_path'] = df_copy['crop_path'].apply(
                lambda x: str(preprocessed_folder + x)
            )
            output_csv = str(Path(original_label_csv).parent / f"{output_prefix}labels_lstm.csv")
        else:
            df_copy['crop_path'] = df_copy['crop_path'].apply(
                lambda x: str(preprocessed_path / mode / x)
            )
            output_csv = str(Path(original_label_csv).parent / f"{output_prefix}{mode}_labels_lstm.csv")
        
        # Save
        df_copy.to_csv(output_csv, index=False)
        output_csvs[mode] = output_csv
        print(f"Created {mode} label CSV: {output_csv}")
    
    return output_csvs