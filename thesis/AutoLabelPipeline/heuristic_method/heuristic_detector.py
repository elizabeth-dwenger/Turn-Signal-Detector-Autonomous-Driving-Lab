import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from scipy.fft import fft, fftfreq


class HeuristicDetector:
    """
    Heuristic detector for turn signal analysis.
    
    Uses yellow channel isolation and FFT-based periodic signal detection
    to identify left, right, hazard, or no turn signal.
    """
    
    # Valid labels matching the VLM pipeline
    VALID_LABELS = {"left", "right", "hazard", "none"}
    
    def __init__(
        self,
        fps: float = 5.0,
        activity_threshold: float = 6200.0,
        hazard_ratio_threshold: float = 0.7,
        freq_min: float = 1.0,
        freq_max: float = 2.5,
        peak_power_multiplier: float = 3.0,
        variance_threshold: float = 0.05
    ):
        """
        Initialize the heuristic detector.
        """
        self.fps = fps
        self.activity_threshold = activity_threshold
        self.hazard_ratio_threshold = hazard_ratio_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.peak_power_multiplier = peak_power_multiplier
        self.variance_threshold = variance_threshold
        
        # Track metrics for compatibility with VLM pipeline
        self._latencies = []
        self._predictions_count = 0
        self._parse_successes = 0
    
    def isolate_yellow_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Isolate yellow pixels using HSV color space.
        
        Args:
            image: RGB image array
            
        Returns:
            Binary mask of yellow pixels
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        return mask
    
    def extract_yellow_intensity_series(
        self,
        images: List[np.ndarray],
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Extract yellow intensity time series from image sequence.
        """
        intensities = []
        
        for img in images:
            if img is None:
                intensities.append(0)
                continue
            
            if roi is not None:
                x1, y1, x2, y2 = roi
                img = img[y1:y2, x1:x2]
                if img.size == 0:
                    intensities.append(0)
                    continue
            
            mask = self.isolate_yellow_channel(img)
            intensities.append(int(mask.sum()))
        
        return np.array(intensities)
    
    def detect_periodic_signal(self, intensities: np.ndarray) -> Dict:
        """
        Detect periodic blinking using FFT analysis.
        """
        n = len(intensities)
        if n < 4:
            return {
                'is_periodic': False,
                'peak_frequency': 0.0,
                'blinks_per_minute': 0.0
            }
        
        intensities_norm = intensities - np.mean(intensities)
        
        freqs = fftfreq(n, 1 / self.fps)
        fft_vals = np.abs(fft(intensities_norm))
        
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # Turn signals blink at 1-2.5 Hz
        freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        
        if np.any(freq_mask):
            peak_idx = np.argmax(fft_vals[freq_mask])
            peak_freq = freqs[freq_mask][peak_idx]
            peak_power = fft_vals[freq_mask][peak_idx]
            mean_power = np.mean(fft_vals)
            is_periodic = peak_power > self.peak_power_multiplier * mean_power
        else:
            peak_freq = 0.0
            peak_power = 0.0
            is_periodic = False
        
        return {
            'is_periodic': bool(is_periodic),
            'peak_frequency': float(peak_freq),
            'blinks_per_minute': float(peak_freq * 60)
        }
    
    def detect_rear_lamp_roi(
        self,
        image: np.ndarray,
        side: str
    ) -> Tuple[int, int, int, int]:
        """
        Detect region of interest for left or right rear lamp.
        
        Args:
            image: Image array
            side: 'left', 'right', or 'both'
            
        Returns:
            ROI as (x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        y1, y2 = int(h * 0.4), h
        
        if side == 'left':
            x1, x2 = 0, int(w * 0.4)
        elif side == 'right':
            x1, x2 = int(w * 0.6), w
        else:
            x1, x2 = 0, w
        
        return (x1, y1, x2, y2)
    
    def predict_sequence(self, images: List[np.ndarray]) -> Dict:
        """
        Predict turn signal for an image sequence.
        """
        import time
        start_time = time.time()
        
        self._predictions_count += 1
        
        if not images or all(img is None for img in images):
            self._latencies.append(time.time() - start_time)
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': 'No valid images provided'
            }
        
        # Filter out None images
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            self._latencies.append(time.time() - start_time)
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': 'No valid images after filtering'
            }
        
        # Overall yellow intensity analysis
        intensities = self.extract_yellow_intensity_series(valid_images)
        periodic_result = self.detect_periodic_signal(intensities)
        
        # Check if blinking detected
        intensity_std = np.std(intensities)
        intensity_mean = np.mean(intensities)
        is_blinking = (
            periodic_result['is_periodic'] or
            (intensity_mean > 0 and intensity_std > intensity_mean * self.variance_threshold)
        )
        
        if not is_blinking:
            self._parse_successes += 1
            self._latencies.append(time.time() - start_time)
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': f'No periodic blinking detected. Periodic: {periodic_result["is_periodic"]}, Variance ratio: {intensity_std / max(intensity_mean, 1e-6):.4f}'
            }
        
        # Analyze left vs right using first valid image for ROI detection
        sample_img = valid_images[0]
        left_roi = self.detect_rear_lamp_roi(sample_img, 'left')
        right_roi = self.detect_rear_lamp_roi(sample_img, 'right')
        
        left_intensity = self.extract_yellow_intensity_series(valid_images, left_roi)
        right_intensity = self.extract_yellow_intensity_series(valid_images, right_roi)
        
        left_activity = float(np.std(left_intensity))
        right_activity = float(np.std(right_intensity))
        
        # Minimum activity threshold
        max_activity = max(left_activity, right_activity)
        if max_activity < self.activity_threshold:
            self._parse_successes += 1
            self._latencies.append(time.time() - start_time)
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': f'Activity below threshold. Left: {left_activity:.1f}, Right: {right_activity:.1f}, Threshold: {self.activity_threshold}'
            }
        
        # Determine signal type
        ratio = min(left_activity, right_activity) / max_activity
        
        if ratio > self.hazard_ratio_threshold:
            predicted = 'hazard'
            reasoning = f'Both sides blinking similarly (ratio={ratio:.2f})'
        elif left_activity > right_activity:
            predicted = 'left'
            reasoning = f'Left side more active ({left_activity:.1f} > {right_activity:.1f})'
        else:
            predicted = 'right'
            reasoning = f'Right side more active ({right_activity:.1f} > {left_activity:.1f})'
        
        self._parse_successes += 1
        self._latencies.append(time.time() - start_time)
        
        # Return prediction with segment covering entire sequence
        # This matches the VLM pipeline output format
        num_frames = len(images)
        return {
            'label': predicted,
            'segments': [{
                'label': predicted,
                'start_frame': 0,
                'end_frame': num_frames - 1
            }],
            'raw_output': 'heuristic',
            'reasoning': reasoning,
            'analysis': {
                'periodic': periodic_result,
                'left_activity': left_activity,
                'right_activity': right_activity,
                'activity_ratio': ratio
            }
        }
    
    def predict_video(
        self,
        video: Optional[np.ndarray] = None,
        chunks: Optional[List] = None
    ) -> Dict:
        """
        Predict turn signal from video tensor or chunks.
        
        Compatible interface with VLM models' predict_video method.
        
        Args:
            video: Video tensor (T, H, W, C) or (T, C, H, W)
            chunks: List of video chunks (not typically used for heuristic)
            
        Returns:
            Prediction dict
        """
        if chunks is not None:
            # Handle chunked video - process all chunks and aggregate
            all_images = []
            for chunk in chunks:
                if isinstance(chunk, np.ndarray):
                    if chunk.ndim == 4:
                        # (T, H, W, C) or (T, C, H, W)
                        if chunk.shape[-1] in [1, 3, 4]:
                            all_images.extend([chunk[i] for i in range(chunk.shape[0])])
                        else:
                            # Assume (T, C, H, W), transpose to (T, H, W, C)
                            chunk = np.transpose(chunk, (0, 2, 3, 1))
                            all_images.extend([chunk[i] for i in range(chunk.shape[0])])
            return self.predict_sequence(all_images)
        
        if video is None:
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': 'No video provided'
            }
        
        # Convert video tensor to list of images
        if video.ndim == 4:
            if video.shape[-1] in [1, 3, 4]:
                # (T, H, W, C)
                images = [video[i] for i in range(video.shape[0])]
            else:
                # (T, C, H, W), transpose
                video = np.transpose(video, (0, 2, 3, 1))
                images = [video[i] for i in range(video.shape[0])]
        else:
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': f'Invalid video shape: {video.shape}'
            }
        
        return self.predict_sequence(images)
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics compatible with VLM pipeline.
        
        Returns:
            Dict with avg_latency_ms, parse_success_rate, total_predictions
        """
        avg_latency = np.mean(self._latencies) * 1000 if self._latencies else 0.0
        parse_rate = self._parse_successes / max(self._predictions_count, 1)
        
        return {
            'avg_latency_ms': float(avg_latency),
            'parse_success_rate': float(parse_rate),
            'total_predictions': self._predictions_count
        }
    
    def reset_metrics(self):
        """Reset internal metrics tracking."""
        self._latencies = []
        self._predictions_count = 0
        self._parse_successes = 0


def load_images_from_paths(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load images from file paths.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of RGB image arrays (None for failed loads)
    """
    images = []
    for path in image_paths:
        if not Path(path).exists():
            images.append(None)
            continue
        
        img = cv2.imread(str(path))
        if img is None:
            images.append(None)
            continue
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    return images
