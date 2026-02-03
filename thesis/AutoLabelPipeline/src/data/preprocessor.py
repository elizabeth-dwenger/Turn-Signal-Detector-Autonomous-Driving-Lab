"""
Image preprocessing for turn signal detection.
Handles resizing, normalization, and preparation for model input.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

from .data_structures import Frame, Sequence
from utils.config import PreprocessingConfig


logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses images for model input.
    Handles resizing, aspect ratio, normalization.
    """
    
    def __init__(self, preprocessing_config):
        self.config = preprocessing_config
        self.target_size = tuple(preprocessing_config.resize_resolution)  # (width, height)
        self.maintain_aspect = preprocessing_config.maintain_aspect_ratio
        self.normalize = preprocessing_config.normalize
        self.padding_color = tuple(preprocessing_config.padding_color)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """        
        Preprocessed image (target_height, target_width, C)
        """
        if image is None:
            raise ValueError("Cannot preprocess None image")
        
        # 1. Resize
        resized = self._resize_image(image)
        
        # 2. Normalize if enabled
        if self.normalize:
            normalized = self._normalize_image(resized)
        else:
            normalized = resized
        
        return normalized
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Batch tensor (B, H, W, C)
        """
        preprocessed = [self.preprocess_image(img) for img in images]
        return np.stack(preprocessed, axis=0)
    
    def preprocess_sequence(self, sequence: Sequence, use_crops: bool = True) -> Sequence:
        """
        Preprocess all images in a sequence (in-place).
        
        Sequence with preprocessed images (modifies in place)
        """
        for frame in sequence.frames:
            if use_crops and frame.crop_image is not None:
                frame.crop_image = self.preprocess_image(frame.crop_image)
            elif not use_crops and frame.full_image is not None:
                frame.full_image = self.preprocess_image(frame.full_image)
        
        return sequence
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            Resized image (target_H, target_W, C)
        """
        target_w, target_h = self.target_size
        
        if not self.maintain_aspect:
            # Simple resize - may distort
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized
        
        # Maintain aspect ratio with padding
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # New size maintaining aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), self.padding_color, dtype=image.dtype)
        
        # Calculate padding offsets (center the image)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
       Normalized image (H, W, C) with values [0, 1]
        """
        # Convert to float and scale to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def get_preprocessed_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of preprocessed images.
        
        Returns:
            (H, W, C) tuple
        """
        return (self.target_size[1], self.target_size[0], 3)  # (height, width, channels)


class SequencePreprocessor:
    """
    Preprocesses entire sequences for video-mode inference.
    Combines ImagePreprocessor with temporal operations.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stride = config.sequence_stride
        self.max_length = config.max_sequence_length
        self.image_preprocessor = ImagePreprocessor(config)
    
    def preprocess_for_video(self, sequence: Sequence, use_crops: bool = True) -> np.ndarray:
        """
        Preprocess sequence and construct video tensor (T, H, W, C)
        """
        # Get images
        if use_crops:
            images = [f.crop_image for f in sequence.frames if f.crop_image is not None]
        else:
            images = [f.full_image for f in sequence.frames if f.full_image is not None]
        
        if not images:
            raise ValueError(f"No images in sequence {sequence.sequence_id}")
        
        # Apply stride (temporal subsampling)
        if self.stride > 1:
            images = images[::self.stride]
            logger.debug(f"Applied stride {self.stride}: {len(sequence.frames)} -> {len(images)} frames")
        
        # Apply max length
        if self.max_length and len(images) > self.max_length:
            # Sample uniformly
            indices = np.linspace(0, len(images) - 1, self.max_length, dtype=int)
            images = [images[i] for i in indices]
            logger.debug(f"Applied max length {self.max_length}: sampled {len(images)} frames")
        
        # Preprocess all images
        preprocessed = [self.image_preprocessor.preprocess_image(img) for img in images]
        
        # Stack into video tensor
        video = np.stack(preprocessed, axis=0)  # (T, H, W, C)
        
        return video

    def preprocess_for_video_with_ids(self, sequence: Sequence, use_crops: bool = True) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess sequence and return video tensor plus original frame_ids.
        """
        # Get frames with images
        if use_crops:
            frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            frames = [f for f in sequence.frames if f.full_image is not None]
        
        if not frames:
            raise ValueError(f"No images in sequence {sequence.sequence_id}")
        
        # Apply stride (temporal subsampling)
        if self.stride > 1:
            frames = frames[::self.stride]
            logger.debug(f"Applied stride {self.stride}: {len(sequence.frames)} -> {len(frames)} frames")
        
        # Apply max length
        if self.max_length and len(frames) > self.max_length:
            indices = np.linspace(0, len(frames) - 1, self.max_length, dtype=int)
            frames = [frames[i] for i in indices]
            logger.debug(f"Applied max length {self.max_length}: sampled {len(frames)} frames")
        
        images = [f.crop_image if use_crops else f.full_image for f in frames]
        frame_ids = [f.frame_id for f in frames]
        
        preprocessed = [self.image_preprocessor.preprocess_image(img) for img in images]
        video = np.stack(preprocessed, axis=0)
        
        return video, frame_ids
    
    def preprocess_for_single_images(self, sequence: Sequence,
                                     use_crops: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Preprocess sequence for single-image mode.
        
        List of (preprocessed_image, frame_id) tuples
        """
        # Get frames with images
        if use_crops:
            valid_frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            valid_frames = [f for f in sequence.frames if f.full_image is not None]
        
        # Apply stride
        if self.stride > 1:
            valid_frames = valid_frames[::self.stride]
        
        # Preprocess each frame
        results = []
        for frame in valid_frames:
            image = frame.crop_image if use_crops else frame.full_image
            preprocessed = self.image_preprocessor.preprocess_image(image)
            results.append((preprocessed, frame.frame_id))
        
        return results
    
    def preprocess_windowed(self, sequence: Sequence, window_size: int,
                           window_stride: int = 1, use_crops: bool = True) -> List[Tuple[np.ndarray, int]]:
        """
        Preprocess sequence into overlapping temporal windows.
        
        List of (video_window, center_frame_id) tuples
        """
        # Get frames
        if use_crops:
            frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            frames = [f for f in sequence.frames if f.full_image is not None]
        
        if len(frames) < window_size:
            logger.warning(f"Sequence {sequence.sequence_id} has {len(frames)} frames, "
                         f"less than window size {window_size}")
            if len(frames) == 0:
                return []
            
            # Process all available frames as single window
            images = [f.crop_image if use_crops else f.full_image for f in frames]
            preprocessed = [self.image_preprocessor.preprocess_image(img) for img in images]
            video = np.stack(preprocessed, axis=0)
            center_frame_id = frames[len(frames) // 2].frame_id
            return [(video, center_frame_id)]
        
        # Create windows
        windows = []
        for i in range(0, len(frames) - window_size + 1, window_stride):
            window_frames = frames[i:i + window_size]
            
            # Get images
            images = [f.crop_image if use_crops else f.full_image for f in window_frames]
            
            # Preprocess
            preprocessed = [self.image_preprocessor.preprocess_image(img) for img in images]
            
            # Stack
            video = np.stack(preprocessed, axis=0)
            
            # Center frame
            center_idx = window_size // 2
            center_frame_id = window_frames[center_idx].frame_id
            
            windows.append((video, center_frame_id))
        
        return windows

    def preprocess_for_video_chunked(self, sequence: Sequence, 
                                     chunk_size: int = 50,
                                     use_crops: bool = True) -> List[Tuple[np.ndarray, int, int]]:
        """
        Preprocess sequence into fixed-size chunks for memory-efficient inference.
        """
        # Get images
        if use_crops:
            frames = [f for f in sequence.frames if f.crop_image is not None]
        else:
            frames = [f for f in sequence.frames if f.full_image is not None]
        
        if not frames:
            raise ValueError(f"No images in sequence {sequence.sequence_id}")
        
        # Apply stride if needed
        if self.stride > 1:
            frames = frames[::self.stride]
        
        # Create chunks
        chunks = []
        for i in range(0, len(frames), chunk_size):
            chunk_frames = frames[i:i + chunk_size]
            start_idx = i
            end_idx = min(i + chunk_size - 1, len(frames) - 1)
            
            # Preprocess chunk
            preprocessed = [
                self.image_preprocessor.preprocess_image(f.crop_image if use_crops else f.full_image) 
                for f in chunk_frames
            ]
            
            # Stack into video tensor
            video = np.stack(preprocessed, axis=0)  # (T, H, W, C)
            chunks.append((video, start_idx, end_idx))
            
            logger.debug(
                f"Chunk {len(chunks)}: frames {start_idx}–{end_idx} "
                f"({len(chunk_frames)} frames, shape {video.shape})"
            )
        
        return chunks


def create_preprocessor(preprocessing_config, mode: str = 'auto'):
    """
    Factory function to create appropriate preprocessor.
    """
    if mode == 'image':
        return ImagePreprocessor(preprocessing_config)
    elif mode == 'sequence':
        return SequencePreprocessor(preprocessing_config)
    else:  # auto
        return SequencePreprocessor(preprocessing_config)
