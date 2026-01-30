"""
Temporal smoothing for turn signal predictions.
Removes flickering and enforces temporal consistency.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import logging
import sys
from pathlib import Path

# Add src to path if needed
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.enums import SmoothingMethod


logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Applies temporal smoothing to sequence predictions.
    Removes single-frame outliers and enforces consistency.
    """
    
    def __init__(self, postprocessing_config):
        """
        Args:
            postprocessing_config: PostprocessingConfig from configuration
        """
        self.config = postprocessing_config
        self.method = postprocessing_config.smoothing_method
        self.window_size = postprocessing_config.smoothing_window_size
        self.enabled = postprocessing_config.temporal_smoothing_enabled
        
        # Validate window size is odd
        if self.window_size % 2 == 0:
            logger.warning(f"Window size {self.window_size} is even, using {self.window_size + 1}")
            self.window_size += 1
    
    def smooth_sequence(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to sequence predictions.
        
        Args:
            predictions: List of prediction dicts with 'label' and 'confidence'
        
        Returns:
            List of smoothed predictions (same format)
        """
        if not self.enabled:
            logger.debug("Temporal smoothing disabled")
            return predictions
        
        if len(predictions) < self.window_size:
            logger.debug(f"Sequence too short ({len(predictions)} < {self.window_size}), skipping smoothing")
            return predictions
        
        # Apply smoothing based on method
        if self.method == SmoothingMethod.MEDIAN:
            return self._median_filter(predictions)
        elif self.method == SmoothingMethod.MODE:
            return self._mode_filter(predictions)
        elif self.method == SmoothingMethod.HMM:
            return self._hmm_filter(predictions)
        elif self.method == SmoothingMethod.THRESHOLD:
            return self._threshold_filter(predictions)
        else:
            logger.warning(f"Unknown smoothing method: {self.method}")
            return predictions
    
    def _median_filter(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply median filter to labels.
        For categorical data, this is effectively a mode filter.
        """
        return self._mode_filter(predictions)  # Median for categories = mode
    
    def _mode_filter(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply mode (most common) filter to labels.
        """
        smoothed = []
        half_window = self.window_size // 2
        
        for i in range(len(predictions)):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            
            # Get labels in window
            window_labels = [predictions[j]['label'] for j in range(start, end)]
            
            # Find most common label
            label_counts = Counter(window_labels)
            smoothed_label = label_counts.most_common(1)[0][0]
            
            # Average confidence in window for this label
            window_confidences = [
                predictions[j]['confidence']
                for j in range(start, end)
                if predictions[j]['label'] == smoothed_label
            ]
            
            smoothed_confidence = np.mean(window_confidences) if window_confidences else predictions[i]['confidence']
            
            # Create smoothed prediction
            smoothed_pred = predictions[i].copy()
            smoothed_pred['label'] = smoothed_label
            smoothed_pred['confidence'] = smoothed_confidence
            smoothed_pred['original_label'] = predictions[i]['label']
            smoothed_pred['smoothed'] = (smoothed_label != predictions[i]['label'])
            
            smoothed.append(smoothed_pred)
        
        # Count how many changed
        changed = sum(1 for p in smoothed if p.get('smoothed', False))
        logger.debug(f"Mode filter: {changed}/{len(predictions)} labels changed ({changed/len(predictions)*100:.1f}%)")
        
        return smoothed
    
    def _hmm_filter(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply Hidden Markov Model for temporal smoothing.
        Models state transitions and uses Viterbi algorithm.
        """
        # Simple HMM implementation
        # States: none, left, right, both
        states = ['none', 'left', 'right', 'both']
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        # Transition probabilities (higher for staying in same state)
        transition_prob = np.array([
            [0.8, 0.1, 0.1, 0.0],  # from none
            [0.1, 0.8, 0.05, 0.05],  # from left
            [0.1, 0.05, 0.8, 0.05],  # from right
            [0.1, 0.3, 0.3, 0.3],   # from both
        ])
        
        # Build emission probabilities from predictions
        emission_probs = []
        for pred in predictions:
            # Confidence distribution
            probs = np.full(len(states), 0.1)  # Small base probability
            label_idx = state_to_idx.get(pred['label'], 0)
            probs[label_idx] = pred['confidence']
            probs = probs / probs.sum()  # Normalize
            emission_probs.append(probs)
        
        emission_probs = np.array(emission_probs)
        
        # Viterbi algorithm
        T = len(predictions)
        N = len(states)
        
        # Initialize
        viterbi = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        
        # Initial probabilities (uniform)
        initial_prob = np.ones(N) / N
        viterbi[0] = initial_prob * emission_probs[0]
        
        # Forward pass
        for t in range(1, T):
            for s in range(N):
                prob = viterbi[t-1] * transition_prob[:, s] * emission_probs[t, s]
                viterbi[t, s] = np.max(prob)
                backpointer[t, s] = np.argmax(prob)
        
        # Backward pass
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi[-1])
        
        for t in range(T-2, -1, -1):
            path[t] = backpointer[t+1, path[t+1]]
        
        # Convert path to labels
        smoothed = []
        for i, pred in enumerate(predictions):
            smoothed_label = states[path[i]]
            
            smoothed_pred = pred.copy()
            smoothed_pred['label'] = smoothed_label
            smoothed_pred['original_label'] = pred['label']
            smoothed_pred['smoothed'] = (smoothed_label != pred['label'])
            
            smoothed.append(smoothed_pred)
        
        changed = sum(1 for p in smoothed if p.get('smoothed', False))
        logger.debug(f"HMM filter: {changed}/{len(predictions)} labels changed ({changed/len(predictions)*100:.1f}%)")
        
        return smoothed
    
    def _threshold_filter(self, predictions: List[Dict]) -> List[Dict]:
        """
        Filter based on confidence threshold and minimum duration.
        Removes brief signals below threshold.
        """
        min_duration = self.config.min_signal_duration_frames
        conf_threshold = self.config.confidence_threshold
        
        smoothed = predictions.copy()
        
        # First pass: mark low-confidence as 'none'
        for pred in smoothed:
            if pred['confidence'] < conf_threshold and pred['label'] != 'none':
                pred['original_label'] = pred['label']
                pred['label'] = 'none'
                pred['smoothed'] = True
        
        # Second pass: remove brief signal episodes
        i = 0
        while i < len(smoothed):
            if smoothed[i]['label'] != 'none':
                # Found signal start
                signal_label = smoothed[i]['label']
                signal_start = i
                
                # Find end of this signal
                j = i
                while j < len(smoothed) and smoothed[j]['label'] == signal_label:
                    j += 1
                
                signal_duration = j - i
                
                # If too short, mark as 'none'
                if signal_duration < min_duration:
                    for k in range(signal_start, j):
                        if 'original_label' not in smoothed[k]:
                            smoothed[k]['original_label'] = smoothed[k]['label']
                        smoothed[k]['label'] = 'none'
                        smoothed[k]['smoothed'] = True
                
                i = j
            else:
                i += 1
        
        changed = sum(1 for p in smoothed if p.get('smoothed', False))
        logger.debug(f"Threshold filter: {changed}/{len(predictions)} labels changed")
        
        return smoothed


class EpisodeReconstructor:
    """
    Reconstructs turn signal episodes for single-image mode.
    Implements the "first ON to last ON" strategy.
    """
    
    def __init__(self, single_image_config):
        """
        Args:
            single_image_config: SingleImageConfig from postprocessing config
        """
        self.config = single_image_config
        self.min_duration = single_image_config.min_signal_duration_frames
        self.max_gap = single_image_config.max_gap_frames
        self.threshold_start = single_image_config.confidence_threshold_start
        self.threshold_continue = single_image_config.confidence_threshold_continue
        self.interpolate = single_image_config.interpolate_gaps
    
    def reconstruct_episodes(self, predictions: List[Dict]) -> List[Dict]:
        """
        Reconstruct signal episodes from single-frame predictions.
        
        Strategy:
        1. Find frames with high confidence signal detection (>= threshold_start)
        2. Group into episodes with max_gap tolerance
        3. Fill gaps within episodes
        4. Filter episodes by min_duration
        
        Args:
            predictions: List of per-frame predictions
        
        Returns:
            List of predictions with reconstructed labels
        """
        if not predictions:
            return predictions
        
        # Process each signal type separately
        result = [p.copy() for p in predictions]
        
        for signal_label in ['left', 'right', 'both']:
            episodes = self._find_episodes(predictions, signal_label)
            
            # Apply episodes to result
            for start, end in episodes:
                for i in range(start, end + 1):
                    if result[i]['label'] != signal_label:
                        result[i]['original_label'] = result[i]['label']
                        result[i]['label'] = signal_label
                        result[i]['reconstructed'] = True
        
        reconstructed = sum(1 for p in result if p.get('reconstructed', False))
        logger.debug(f"Episode reconstruction: {reconstructed}/{len(predictions)} frames updated")
        
        return result
    
    def _find_episodes(self, predictions: List[Dict], signal_label: str) -> List[tuple]:
        """
        Find episodes for a specific signal label.
        
        Returns:
            List of (start_idx, end_idx) tuples
        """
        # Find high-confidence detections
        high_conf_frames = [
            i for i, p in enumerate(predictions)
            if p['label'] == signal_label and p['confidence'] >= self.threshold_start
        ]
        
        if not high_conf_frames:
            return []
        
        # Group into episodes with gap tolerance
        episodes = []
        current_start = high_conf_frames[0]
        current_end = high_conf_frames[0]
        
        for i in range(1, len(high_conf_frames)):
            frame_idx = high_conf_frames[i]
            gap = frame_idx - current_end - 1
            
            if gap <= self.max_gap:
                # Extend current episode
                current_end = frame_idx
            else:
                # Save current episode and start new one
                if self._is_valid_episode(predictions, current_start, current_end, signal_label):
                    episodes.append((current_start, current_end))
                
                current_start = frame_idx
                current_end = frame_idx
        
        # Don't forget last episode
        if self._is_valid_episode(predictions, current_start, current_end, signal_label):
            episodes.append((current_start, current_end))
        
        # Interpolate gaps if enabled
        if self.interpolate:
            episodes = self._interpolate_episodes(predictions, episodes, signal_label)
        
        return episodes
    
    def _is_valid_episode(self, predictions: List[Dict], start: int, end: int, signal_label: str) -> bool:
        """Check if episode meets minimum duration requirement"""
        # Count frames with this label (not gaps)
        signal_frames = sum(
            1 for i in range(start, end + 1)
            if predictions[i]['label'] == signal_label
        )
        
        return signal_frames >= self.min_duration
    
    def _interpolate_episodes(self, predictions: List[Dict], episodes: List[tuple], signal_label: str) -> List[tuple]:
        """
        Extend episodes to cover gaps between high-confidence frames.
        Fill frames between first detection and last detection.
        """
        interpolated = []
        
        for start, end in episodes:
            # Find actual first and last frame with signal in this episode
            first_signal = start
            last_signal = end
            
            # Extend to cover the full range
            interpolated.append((first_signal, last_signal))
        
        return interpolated
