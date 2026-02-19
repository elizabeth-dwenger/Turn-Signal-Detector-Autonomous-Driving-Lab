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
        activity_threshold: float = 5500.0,  
        hazard_ratio_threshold: float = 0.9,
        freq_min: float = 1.0,
        freq_max: float = 2.5,
        peak_power_multiplier: float = 3.0,
        variance_threshold: float = 0.05,
        # HSV color range for yellow/orange detection
        hue_min: int = 20,
        hue_max: int = 45,
        sat_min: int = 100,
        sat_max: int = 255,
        val_min: int = 100,
        val_max: int = 255,
        # Additional hazard detection parameters
        min_hazard_activity: float = 10000.0,  # Very high to prevent false hazards
        roi_split: float = 0.4,  # Left ROI: [0, roi_split], Right ROI: [1-roi_split, 1]
        max_ratio_for_directional: float = 0.3,  # Ratio must be below this for left/right
        require_periodicity: bool = False,  # Require FFT-detected periodicity for non-none
        min_cv: float = 0.0  # Disabled - CV doesn't help (none has higher CV than signals)
    ):
        """
        Initialize the heuristic detector.
        
        HSV ranges for turn signal detection:
        - Hue: 0-179 in OpenCV (15-35 = yellow/orange, 0-15 = red/orange)
        - Saturation: 0-255 (higher = more colorful)
        - Value: 0-255 (higher = brighter)
        """
        self.fps = fps
        self.activity_threshold = activity_threshold
        self.hazard_ratio_threshold = hazard_ratio_threshold
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.peak_power_multiplier = peak_power_multiplier
        self.variance_threshold = variance_threshold
        
        # HSV color thresholds
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        
        # Additional hazard detection parameters
        self.min_hazard_activity = min_hazard_activity
        self.roi_split = roi_split
        self.max_ratio_for_directional = max_ratio_for_directional
        self.require_periodicity = require_periodicity
        self.min_cv = min_cv
        
        # Track metrics for compatibility with VLM pipeline
        self._latencies = []
        self._predictions_count = 0
        self._parse_successes = 0
    
    def isolate_yellow_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Isolate yellow/orange pixels using HSV color space.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv,
            (self.hue_min, self.sat_min, self.val_min),
            (self.hue_max, self.sat_max, self.val_max)
        )
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
        Uses self.roi_split to define boundaries.
        """
        h, w = image.shape[:2]
        y1, y2 = int(h * 0.4), h
        
        if side == 'left':
            x1, x2 = 0, int(w * self.roi_split)
        elif side == 'right':
            x1, x2 = int(w * (1 - self.roi_split)), w
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
        
        if self.require_periodicity:
            # Strict mode: only trust FFT-detected periodicity
            is_blinking = periodic_result['is_periodic']
        else:
            # Relaxed mode: also consider high variance as potential blinking
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
        
        # Compute coefficient of variation (CV = std/mean)
        left_mean = float(np.mean(left_intensity))
        right_mean = float(np.mean(right_intensity))
        left_cv = left_activity / left_mean if left_mean > 0 else 0
        right_cv = right_activity / right_mean if right_mean > 0 else 0
        max_cv = max(left_cv, right_cv)
        
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
        
        # Check coefficient of variation (filters out constant lights)
        if self.min_cv > 0 and max_cv < self.min_cv:
            self._parse_successes += 1
            self._latencies.append(time.time() - start_time)
            return {
                'label': 'none',
                'segments': [],
                'raw_output': 'heuristic',
                'reasoning': f'CV below threshold (constant light). Max CV: {max_cv:.3f}, Threshold: {self.min_cv}'
            }
        
        # Determine signal type
        ratio = min(left_activity, right_activity) / max_activity
        min_activity = min(left_activity, right_activity)
        
        # For hazard: both sides must be active AND similar
        if ratio > self.hazard_ratio_threshold and min_activity >= self.min_hazard_activity:
            predicted = 'hazard'
            reasoning = f'Both sides blinking similarly (ratio={ratio:.2f}, min_activity={min_activity:.1f})'
        # For directional: require clear asymmetry (ratio below threshold)
        elif ratio < self.max_ratio_for_directional:
            if left_activity > right_activity:
                predicted = 'left'
                reasoning = f'Left side clearly more active ({left_activity:.1f} >> {right_activity:.1f}, ratio={ratio:.2f})'
            else:
                predicted = 'right'
                reasoning = f'Right side clearly more active ({right_activity:.1f} >> {left_activity:.1f}, ratio={ratio:.2f})'
        # Ambiguous case: classify based on which side is more active
        elif left_activity > right_activity:
            predicted = 'left'
            reasoning = f'Left side more active ({left_activity:.1f} > {right_activity:.1f}, ratio={ratio:.2f} ambiguous)'
        else:
            predicted = 'right'
            reasoning = f'Right side more active ({right_activity:.1f} > {left_activity:.1f}, ratio={ratio:.2f} ambiguous)'
        
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
