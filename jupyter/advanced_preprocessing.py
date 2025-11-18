"""
Advanced preprocessing strategies to make turn signals more visible to the API.
Focuses on: zooming in on signals, creating difference maps, and temporal summaries.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def detect_and_crop_rear_lights(image: np.ndarray, 
                                side: str = 'both',
                                expand_factor: float = 1.5) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Detect bright regions (likely lights) and crop around them.
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Find bright regions
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright regions
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # No bright regions found, use heuristic
        y1 = int(h * 0.4)
        y2 = h
        if side == 'left':
            x1, x2 = 0, int(w * 0.5)
        elif side == 'right':
            x1, x2 = int(w * 0.5), w
        else:
            x1, x2 = 0, w
    else:
        # Get bounding box of all bright regions
        all_points = np.vstack([c.squeeze() for c in contours if c.squeeze().ndim == 2])
        x, y, bw, bh = cv2.boundingRect(all_points)
        
        # Expand the region
        cx, cy = x + bw//2, y + bh//2
        new_w = int(bw * expand_factor)
        new_h = int(bh * expand_factor)
        
        x1 = max(0, cx - new_w//2)
        y1 = max(0, cy - new_h//2)
        x2 = min(w, cx + new_w//2)
        y2 = min(h, cy + new_h//2)
        
        # Apply side filter
        if side == 'left' and x1 > w * 0.5:
            x1, x2 = 0, int(w * 0.5)
        elif side == 'right' and x2 < w * 0.5:
            x1, x2 = int(w * 0.5), w
    
    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def create_zoomed_sequence(image_paths: List[str], 
                          side: str = 'both') -> List[Tuple[np.ndarray, Tuple]]:
    """
    Create zoomed versions of all frames focusing on rear lights.
    """
    zoomed_frames = []
    
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cropped, bbox = detect_and_crop_rear_lights(img, side)
        zoomed_frames.append((cropped, bbox))
    
    return zoomed_frames


def create_temporal_difference_map(image_paths: List[str]) -> np.ndarray:
    """
    Create a single image showing where changes occur over time.
    Blinking lights will show up as bright spots.
    """
    frames = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    # Compute differences between consecutive frames
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        diffs.append(diff)
    
    # Stack differences - sum them to see cumulative changes
    diff_stack = np.array(diffs)
    
    # Create visualization: mean and max
    diff_mean = np.mean(diff_stack, axis=0).astype(np.uint8)
    diff_max = np.max(diff_stack, axis=0).astype(np.uint8)
    
    # Combine: use max for intensity, mean for color
    combined = np.maximum(diff_mean, diff_max)
    
    # Apply colormap for better visibility
    combined_gray = cv2.cvtColor(combined, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(combined_gray, cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap


def create_blinking_animation_grid(image_paths: List[str], 
                                   grid_cols: int = 4) -> np.ndarray:
    """
    Create a grid showing frames side-by-side with difference overlays.
    """
    frames = [cv2.imread(p) for p in image_paths]
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    
    # Resize frames to smaller size
    target_size = (200, 150)
    frames_resized = [cv2.resize(f, target_size) for f in frames]
    
    # Calculate grid dimensions
    n_frames = len(frames_resized)
    n_rows = (n_frames + grid_cols - 1) // grid_cols
    
    # Create blank canvas
    canvas_h = n_rows * target_size[1]
    canvas_w = grid_cols * target_size[0]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place frames
    for idx, frame in enumerate(frames_resized):
        row = idx // grid_cols
        col = idx % grid_cols
        y = row * target_size[1]
        x = col * target_size[0]
        canvas[y:y+target_size[1], x:x+target_size[0]] = frame
        
        # Add frame number
        cv2.putText(canvas, f"F{idx+1}", (x+5, y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return canvas


def extract_left_right_regions(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split image into left and right halves to isolate each signal.
    """
    h, w = image.shape[:2]
    mid = w // 2
    
    # Focus on bottom half where lights are
    y_start = int(h * 0.4)
    
    left_region = image[y_start:, :mid]
    right_region = image[y_start:, mid:]
    
    return left_region, right_region


def create_side_by_side_regions(image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract left and right regions from all frames.
    """
    left_regions = []
    right_regions = []
    
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        left, right = extract_left_right_regions(img)
        left_regions.append(left)
        right_regions.append(right)
    
    return left_regions, right_regions


def create_annotated_frame(image: np.ndarray,
                          detection_results: dict,
                          frame_idx: int) -> np.ndarray:
    """
    Annotate frame with arrows and text showing signal detection.
    """
    annotated = image.copy()
    pil_img = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_img)
    
    h, w = image.shape[:2]
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add frame number
    draw.text((10, 10), f"Frame {frame_idx + 1}", fill=(0, 255, 0), font=font)
    
    # Add signal detection info
    if 'frames_with_signal' in detection_results:
        if frame_idx in detection_results['frames_with_signal']:
            draw.text((10, h - 30), "⚠ SIGNAL ON", fill=(255, 255, 0), font=font)
            
            # Draw arrows pointing to likely signal locations
            # Left side arrow
            draw.polygon([(30, h//2), (10, h//2 - 10), (10, h//2 + 10)], 
                        fill=(255, 255, 0))
            # Right side arrow
            draw.polygon([(w-30, h//2), (w-10, h//2 - 10), (w-10, h//2 + 10)], 
                        fill=(255, 255, 0))
    
    return np.array(pil_img)


def create_comprehensive_summary_image(image_paths: List[str],
                                       analysis_results: dict = None) -> np.ndarray:
    """
    Create a single comprehensive image showing:
    - Original frames in a grid
    - Temporal difference map
    - Yellow isolation
    - Signal intensity graph
    
    This single image gives the API ALL the information at once.
    """
    from turn_signal_preprocessing import (
        isolate_yellow_channel, 
        extract_yellow_intensity_series,
        enhance_image
    )
    
    # Load frames
    frames = [cv2.imread(p) for p in image_paths]
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    
    # Resize for display
    display_size = (160, 120)
    frames_small = [cv2.resize(f, display_size) for f in frames]
    
    # Create layout
    n_frames = len(frames_small)
    grid_cols = min(5, n_frames)
    grid_rows = (n_frames + grid_cols - 1) // grid_cols
    
    grid_h = grid_rows * display_size[1]
    grid_w = grid_cols * display_size[0]
    
    # Create canvas (frames + difference + intensity graph)
    canvas_h = grid_h + 300  # Extra space for diff map and graph
    canvas_w = max(grid_w, 800)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 1. Place original frames
    for idx, frame in enumerate(frames_small):
        row = idx // grid_cols
        col = idx % grid_cols
        y = row * display_size[1]
        x = col * display_size[0]
        canvas[y:y+display_size[1], x:x+display_size[0]] = frame
        
        # Frame number
        cv2.putText(canvas, f"{idx+1}", (x+5, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 2. Difference map
    diff_map = create_temporal_difference_map(image_paths)
    diff_resized = cv2.resize(diff_map, (300, 200))
    canvas[grid_h:grid_h+200, 0:300] = diff_resized
    cv2.putText(canvas, "Changes Over Time", (10, grid_h+20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 3. Yellow intensity graph
    intensities = extract_yellow_intensity_series(image_paths)
    
    # Create intensity plot as image
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(intensities, 'o-', linewidth=2, color='orange', markersize=8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Yellow Intensity')
    ax.set_title('Turn Signal Intensity Over Time')
    ax.grid(True, alpha=0.3)
    
    # Convert plot to image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.toarray_argb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plot_img = plot_img[:, :, 1:]  # Remove alpha
    plt.close(fig)
    
    # Place plot
    plot_resized = cv2.resize(plot_img, (500, 200))
    canvas[grid_h:grid_h+200, 310:810] = plot_resized
    
    # 4. Add text annotations
    if analysis_results:
        y_text = grid_h + 220
        cv2.putText(canvas, f"Periodic: {analysis_results.get('is_periodic', False)}", 
                   (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"Predicted: {analysis_results.get('predicted_signal', 'unknown')}", 
                   (10, y_text+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return canvas


def analyze_and_label_all_sequences(label_csv: str,
                                   image_base_path: str = '../seq_img',
                                   fps: float = 5.0,
                                   output_path: str = 'computational_results.json') -> dict:
    """
    Use pure computational analysis (no API) to label sequences.
    """
    from turn_signal_preprocessing import analyze_sequence_for_turn_signals
    import pandas as pd
    import json
    
    df = pd.read_csv(label_csv)
    
    results = []
    correct = 0
    total = 0
    
    for seq_id, group in df.groupby('sequence_id'):
        group = group.sort_values('frame_idx')
        
        # Get image paths
        if image_base_path:
            image_paths = [str(image_base_path + p) for p in group['crop_path'].tolist()]
        else:
            image_paths = group['crop_path'].tolist()
        
        true_label = group['label'].iloc[0]
        
        # Analyze
        analysis = analyze_sequence_for_turn_signals(image_paths, fps=fps, visualize=False)
        predicted_label = analysis['predicted_signal']
        
        is_correct = bool(predicted_label == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'sequence_id': seq_id,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': bool(is_correct)
        })
        
        print(f"{seq_id}: True={true_label}, Pred={predicted_label}, "
              f"{'✓' if is_correct else '✗'}")
    
    accuracy = correct / total
    print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'results': results,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    image_paths = ['../seq_img/...'] * 10  # Your image paths
    
    # Strategy 1: Zoom into lights
    zoomed = create_zoomed_sequence(image_paths)
    
    # Strategy 2: Difference map
    diff_map = create_temporal_difference_map(image_paths)
    plt.imshow(diff_map)
    plt.title("Temporal Difference Map")
    plt.show()
    
    # Strategy 3: Split left/right
    left_regions, right_regions = create_side_by_side_regions(image_paths)
    
    # Strategy 4: Animation grid
    grid = create_blinking_animation_grid(image_paths)
    plt.imshow(grid)
    plt.show()
    
    # Strategy 5: Comprehensive summary
    summary = create_comprehensive_summary_image(image_paths)
    plt.figure(figsize=(12, 8))
    plt.imshow(summary)
    plt.title("Comprehensive Analysis Summary")
    plt.axis('off')
    plt.show()
    
    # Strategy 6: Pure computational analysis (NO API!)
    results = analyze_and_label_all_sequences(
        label_csv='labels_output/labels_lstm.csv',
        image_base_path='../seq_img'
    )