"""
Batch processing utilities for efficient model inference.
Handles batching for both video and single-image modes.
"""
import numpy as np
from typing import List, Iterator, Tuple, Optional
import logging

from .data_structures import Sequence, Frame


logger = logging.getLogger(__name__)


class BatchBuilder:
    """
    Builds batches of preprocessed data for model inference.
    """
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    def batch_sequences(self, sequences: List[Sequence]) -> Iterator[List[Sequence]]:
        """
        Batch sequences for video-mode processing.
        Note:
            For video mode, batch_size is typically 1 since each sequence
            is already a video tensor. But this allows processing multiple
            sequences in parallel if the model supports it.
        """
        for i in range(0, len(sequences), self.batch_size):
            yield sequences[i:i + self.batch_size]
    
    def batch_frames(self, frames: List[Frame]) -> Iterator[List[Frame]]:
        """
        Batch frames for single-image mode processing.
        
        Args:
            frames: List of Frame objects
        
        Yields:
            Batches of frames
        """
        for i in range(0, len(frames), self.batch_size):
            yield frames[i:i + self.batch_size]
    
    def batch_images(self, images: List[np.ndarray]) -> Iterator[np.ndarray]:
        """
        Batch preprocessed images into tensors.
        """
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            yield np.stack(batch, axis=0)
    
    def batch_with_metadata(self, items: List[Tuple[np.ndarray, any]]) -> Iterator[Tuple[np.ndarray, List]]:
        """
        Batch items that include metadata (e.g., (image, frame_id)).
        """
        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            
            # Separate data and metadata
            data = [item[0] for item in batch_items]
            metadata = [item[1] for item in batch_items]
            
            # Stack data
            batch_tensor = np.stack(data, axis=0)
            
            yield (batch_tensor, metadata)


class SequenceBatcher:
    """
    Advanced batching for sequences with different lengths.
    Handles padding and masking for variable-length sequences.
    """
    
    def __init__(self, batch_size: int, max_sequence_length: Optional[int] = None):
        """
        Args:
            batch_size: Number of sequences per batch
            max_sequence_length: Maximum sequence length (for padding)
        """
        self.batch_size = batch_size
        self.max_length = max_sequence_length
    
    def batch_videos(self, videos: List[np.ndarray],
                     pad_value: float = 0.0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch video tensors with padding for different lengths.
        
        Args:
            videos: List of video tensors (each T, H, W, C) with potentially different T
            pad_value: Value to use for padding
        
        Yields:
            (batch_tensor, mask) where:
              - batch_tensor: (B, max_T, H, W, C) padded videos
              - mask: (B, max_T) boolean mask (True = valid frame, False = padding)
        """
        for i in range(0, len(videos), self.batch_size):
            batch_videos = videos[i:i + self.batch_size]
            
            # Find max length in this batch
            lengths = [v.shape[0] for v in batch_videos]
            max_len = max(lengths)
            
            # Apply max_length constraint if set
            if self.max_length and max_len > self.max_length:
                max_len = self.max_length
                # Truncate videos that exceed max_length
                batch_videos = [v[:max_len] for v in batch_videos]
                lengths = [min(l, max_len) for l in lengths]
            
            # Get spatial dimensions from first video
            _, h, w, c = batch_videos[0].shape
            
            # Create padded batch
            batch_tensor = np.full(
                (len(batch_videos), max_len, h, w, c),
                pad_value,
                dtype=batch_videos[0].dtype
            )
            
            # Create mask
            mask = np.zeros((len(batch_videos), max_len), dtype=bool)
            
            # Fill in actual videos and masks
            for j, video in enumerate(batch_videos):
                actual_len = min(video.shape[0], max_len)
                batch_tensor[j, :actual_len] = video[:actual_len]
                mask[j, :actual_len] = True
            
            yield (batch_tensor, mask)
    
    def collate_videos(self, videos: List[np.ndarray],
                      pad_to_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collate multiple videos into a single batch (no iteration).
        """
        if not videos:
            raise ValueError("Cannot collate empty list of videos")
        
        # Determine padding length
        lengths = [v.shape[0] for v in videos]
        max_len = pad_to_length if pad_to_length else max(lengths)
        
        # Get spatial dimensions
        _, h, w, c = videos[0].shape
        
        # Create padded batch
        batch_tensor = np.zeros((len(videos), max_len, h, w, c), dtype=videos[0].dtype)
        mask = np.zeros((len(videos), max_len), dtype=bool)
        
        # Fill
        for i, video in enumerate(videos):
            actual_len = min(video.shape[0], max_len)
            batch_tensor[i, :actual_len] = video[:actual_len]
            mask[i, :actual_len] = True
        
        return batch_tensor, mask


class DynamicBatcher:
    """
    Dynamic batching that groups sequences of similar length together.
    Reduces padding waste.
    """
    
    def __init__(self, batch_size: int, length_tolerance: int = 10):
        self.batch_size = batch_size
        self.length_tolerance = length_tolerance
    
    def batch_by_length(self, sequences: List[Sequence]) -> Iterator[List[Sequence]]:
        """
        Batch sequences by grouping similar lengths together.
        """
        # Sort by length
        sorted_seqs = sorted(sequences, key=lambda s: s.num_frames)
        
        current_batch = []
        current_length = 0
        
        for seq in sorted_seqs:
            if not current_batch:
                # Start new batch
                current_batch.append(seq)
                current_length = seq.num_frames
            elif len(current_batch) >= self.batch_size:
                # Batch full, yield and start new
                yield current_batch
                current_batch = [seq]
                current_length = seq.num_frames
            elif abs(seq.num_frames - current_length) <= self.length_tolerance:
                # Similar length, add to current batch
                current_batch.append(seq)
            else:
                # Length too different, yield current and start new
                yield current_batch
                current_batch = [seq]
                current_length = seq.num_frames
        
        # Yield final batch
        if current_batch:
            yield current_batch


def create_batcher(batch_size: int, mode: str = 'simple', **kwargs):
    """
    Factory function to create appropriate batcher.
    """
    if mode == 'simple':
        return BatchBuilder(batch_size)
    elif mode == 'sequence':
        max_length = kwargs.get('max_sequence_length')
        return SequenceBatcher(batch_size, max_length)
    elif mode == 'dynamic':
        tolerance = kwargs.get('length_tolerance', 10)
        return DynamicBatcher(batch_size, tolerance)
    else:
        raise ValueError(f"Unknown batcher mode: {mode}")
