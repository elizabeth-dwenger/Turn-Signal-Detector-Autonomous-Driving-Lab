"""
Temporal smoothing for turn signal predictions.
Removes flickering and enforces temporal consistency.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
import logging

from utils.enums import SmoothingMethod


logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Applies temporal smoothing to sequence predictions.
    Removes single-frame outliers and enforces consistency.
    """
    def __init__(self, postprocessing_config):
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
        """
        if not self.enabled:
            logger.debug("Temporal smoothing disabled")
            return predictions
        
        if len(predictions) < self.window_size:
            logger.debug(f"Sequence too short ({len(predictions)} < {self.window_size}), skipping smoothing")
            return predictions
        
        # Apply smoothing based on method
        if self.method == SmoothingMethod.MODE:
            return self._mode_filter(predictions)
        elif self.method == SmoothingMethod.HMM:
            return self._hmm_filter(predictions)
        elif self.method == SmoothingMethod.THRESHOLD:
            logger.warning("Threshold smoothing is disabled; returning raw predictions.")
            return predictions
        else:
            logger.warning(f"Unknown smoothing method: {self.method}")
            return predictions
    
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
            
            # Create smoothed prediction
            smoothed_pred = predictions[i].copy()
            smoothed_pred['label'] = smoothed_label
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
        # States: none, left, right, hazard
        states = ['none', 'left', 'right', 'hazard']
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        # Transition probabilities (higher for staying in same state)
        transition_prob = np.array([
            [0.8, 0.1, 0.1, 0.0],  # from none
            [0.1, 0.8, 0.05, 0.05],  # from left
            [0.1, 0.05, 0.8, 0.05],  # from right
            [0.1, 0.3, 0.3, 0.3],   # from hazard
        ])
        
        # Build emission probabilities from predictions
        emission_probs = []
        for pred in predictions:
            # Emission distribution
            probs = np.full(len(states), 0.1)  # Small base probability
            pred_label = pred['label']
            if pred_label == 'both':
                pred_label = 'hazard'
            label_idx = state_to_idx.get(pred_label, 0)
            probs[label_idx] = 1.0
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
        self.interpolate = single_image_config.interpolate_gaps
    
    def reconstruct_episodes(self, predictions: List[Dict]) -> List[Dict]:
        """
        Reconstruct signal episodes from single-frame predictions.
        
        Strategy:
        1. Find frames with signal labels
        2. Group into episodes with max_gap tolerance
        3. Fill gaps within episodes
        4. Filter episodes by min_duration
        """
        if not predictions:
            return predictions
        
        # Process each signal type separately
        result = [p.copy() for p in predictions]
        
        for signal_label in ['left', 'right', 'hazard']:
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
        """
        if signal_label == 'hazard':
            match_labels = {'hazard', 'both'}
        else:
            match_labels = {signal_label}

        signal_frames = [
            i for i, p in enumerate(predictions)
            if p['label'] in match_labels
        ]
        
        if not signal_frames:
            return []
        
        # Group into episodes with gap tolerance
        episodes = []
        current_start = signal_frames[0]
        current_end = signal_frames[0]
        
        for i in range(1, len(signal_frames)):
            frame_idx = signal_frames[i]
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
        Extend episodes to cover gaps between signal frames.
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
