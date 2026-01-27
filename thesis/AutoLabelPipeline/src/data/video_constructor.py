"""
Video sequence construction for video-mode models.
Constructs video tensors from frame sequences.
"""
import numpy as np
from typing import List, Optional, Tuple
import logging

from .data_structures import Sequence, Frame


logger = logging.getLogger(__name__)


class VideoConstructor:
    """
    Constructs video sequences from frames for video-mode inference.
    """
    
    def __init__(self, preprocessing_config):
        """
        Args:
            preprocessing_config: PreprocessingConfig from configuration
        """
        self.config = preprocessing_config
        self.max_length = preprocessing_config.max_sequence_length
        self.stride = preprocessing_config.sequence_stride
    
    def construct_video(self, sequence: Sequence, use_crops: bool = True) -> np.ndarray:
        """
        Construct video tensor from sequence.
        
        Note:
            This method requires all images to have the same shape.
            If images have different sizes,
            they must be resized first in Preprocessing.
        """
        # Get images from frames
        if use_crops:
            images = [f.crop_image for f in sequence.frames if f.crop_image is not None]
        else:
            images = [f.full_image for f in sequence.frames if f.full_image is not None]
        
        if not images:
            raise ValueError(f"No images loaded for sequence {sequence.sequence_id}")
        
        # Apply stride if needed (subsample long sequences)
        if self.stride > 1:
            images = images[::self.stride]
            logger.debug(f"Applied stride {self.stride}: {len(sequence.frames)} -> {len(images)} frames")
        
        # Apply max length if needed
        if self.max_length and len(images) > self.max_length:
            # Sample uniformly to get max_length frames
            indices = np.linspace(0, len(images) - 1, self.max_length, dtype=int)
            images = [images[i] for i in indices]
            logger.debug(f"Applied max length {self.max_length}: sampled {len(images)} frames")
        
        # Check if all images have the same shape
        shapes = [img.shape for img in images]
        if len(set(shapes)) > 1:
            # Images have different shapes - this is expected for tracking data
            # where vehicles get closer/farther. Need preprocessing.
            raise ValueError(
                f"Cannot stack images with different shapes. "
                f"Found shapes: {set(shapes)}. "
                f"Images must be resized to a consistent size in preprocessing (Stage 2). "
                f"For now, use FrameSampler for single-image mode, or implement preprocessing first."
            )
        
        # Stack into video tensor
        video = np.stack(images, axis=0)  # (T, H, W, C)
        
        return video
    
    def construct_windowed_videos(self, sequence: Sequence, window_size: int,
                                  stride: int = 1, use_crops: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Construct overlapping temporal windows from sequence.
        Useful for models that process fixed-length windows.
        """
        # Get images
        if use_crops:
            frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            frames = [f for f in sequence.frames if f.full_image is not None]
        
        if len(frames) < window_size:
            logger.warning(f"Sequence {sequence.sequence_id} has only {len(frames)} frames, "
                         f"less than window size {window_size}")
            # Pad or skip?
            if len(frames) == 0:
                return []
            # Return single window with available frames
            images = [f.crop_image if use_crops else f.full_image for f in frames]
            video = np.stack(images, axis=0)
            center_frame_id = frames[len(frames) // 2].frame_id
            return [(video, center_frame_id)]
        
        windows = []
        for i in range(0, len(frames) - window_size + 1, stride):
            window_frames = frames[i:i + window_size]
            images = [f.crop_image if use_crops else f.full_image for f in window_frames]
            video = np.stack(images, axis=0)
            
            # Center frame of window
            center_idx = window_size // 2
            center_frame_id = window_frames[center_idx].frame_id
            
            windows.append((video, center_frame_id))
        
        return windows
    
    def get_frame_indices(self, sequence: Sequence) -> List[int]:
        """
        Get which frame indices will be used after stride/max_length.
        Useful for mapping predictions back to original frames.
        """
        total_frames = len(sequence.frames)
        
        # Apply stride
        indices = list(range(0, total_frames, self.stride))
        
        # Apply max length
        if self.max_length and len(indices) > self.max_length:
            indices = np.linspace(0, total_frames - 1, self.max_length, dtype=int).tolist()
        
        return indices


class FrameSampler:
    """
    Samples individual frames from sequences for single-image mode.
    """
    
    def __init__(self, preprocessing_config):
        """
        Args:
            preprocessing_config: PreprocessingConfig from configuration
        """
        self.config = preprocessing_config
        self.stride = preprocessing_config.sequence_stride
    
    def sample_frames(self, sequence: Sequence, use_crops: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Sample frames from sequence for single-image processing.
        """
        # Get frames with images
        if use_crops:
            valid_frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            valid_frames = [f for f in sequence.frames if f.full_image is not None]
        
        # Apply stride
        if self.stride > 1:
            valid_frames = valid_frames[::self.stride]
        
        # Extract images and frame IDs
        samples = []
        for frame in valid_frames:
            image = frame.crop_image if use_crops else frame.full_image
            samples.append((image, frame.frame_id))
        
        return samples
    
    def sample_all_frames(self, sequence: Sequence, use_crops: bool = True) -> List[Frame]:
        """
        Get all frames (with stride applied) for processing.
        """
        # Get frames with images
        if use_crops:
            valid_frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            valid_frames = [f for f in sequence.frames if f.full_image is not None]
        
        # Apply stride
        if self.stride > 1:
            valid_frames = valid_frames[::self.stride]
        
        return valid_frames


def verify_sequence_loaded(sequence: Sequence, use_crops: bool = True) -> bool:
    """
    Verify that a sequence has all images loaded.
    
    True if all images loaded, False otherwise
    """
    for frame in sequence.frames:
        image = frame.crop_image if use_crops else frame.full_image
        if image is None:
            return False
    return True


def get_sequence_shape(sequence: Sequence, use_crops: bool = True) -> Optional[Tuple[int, int, int]]:
    """
    Get the shape of images in sequence.
    
    (H, W, C) shape or None if no images loaded
    """
    for frame in sequence.frames:
        image = frame.crop_image if use_crops else frame.full_image
        if image is not None:
            return image.shape
    return None
