"""
Image loading utilities for frames and sequences.
Loads crop images (and optionally full frames) into memory.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .data_structures import Frame, Sequence


logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Loads images for frames and sequences.
    Can load in parallel for speed.
    """
    
    def __init__(self, crop_base_dir: str, frame_base_dir: Optional[str] = None,
                 num_workers: int = 4):
        """
        Args:
            crop_base_dir: Base directory for crop images
            frame_base_dir: Base directory for full frame images (optional)
            num_workers: Number of parallel workers for loading
        """
        self.crop_base_dir = Path(crop_base_dir)
        self.frame_base_dir = Path(frame_base_dir) if frame_base_dir else None
        self.num_workers = num_workers
        
        if not self.crop_base_dir.exists():
            raise FileNotFoundError(f"Crop directory not found: {self.crop_base_dir}")
    
    def load_frame_images(self, frame: Frame, load_full_frame: bool = False) -> Frame:
        """
        Load images for a single frame.
        """
        # Load crop image
        crop_path = self._resolve_path(frame.crop_path, self.crop_base_dir)
        frame.crop_image = self._load_image(crop_path)
        
        if frame.crop_image is None:
            logger.warning(f"Failed to load crop image: {crop_path}")
        
        # Load full frame if requested
        if load_full_frame and frame.img_path and self.frame_base_dir:
            img_path = self._resolve_path(frame.img_path, self.frame_base_dir)
            frame.full_image = self._load_image(img_path)
            
            if frame.full_image is None:
                logger.warning(f"Failed to load full image: {img_path}")
        
        return frame
    
    def load_sequence_images(self, sequence: Sequence,
                            load_full_frames: bool = False,
                            show_progress: bool = False) -> Sequence:
        """
        Load images for all frames in a sequence.
        """
        frames = sequence.frames
        
        if self.num_workers > 1:
            # Parallel loading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self.load_frame_images, frame, load_full_frames): frame
                    for frame in frames
                }
                
                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=len(frames),
                                  desc=f"Loading {sequence.sequence_id[:30]}...")
                
                for future in iterator:
                    try:
                        future.result()
                    except Exception as e:
                        frame = futures[future]
                        logger.error(f"Error loading images for frame {frame.frame_id}: {e}")
        else:
            # Sequential loading
            iterator = frames
            if show_progress:
                iterator = tqdm(iterator, desc=f"Loading {sequence.sequence_id[:30]}...")
            
            for frame in iterator:
                try:
                    self.load_frame_images(frame, load_full_frames)
                except Exception as e:
                    logger.error(f"Error loading images for frame {frame.frame_id}: {e}")
        
        return sequence
    
    def load_dataset_images(self, dataset, load_full_frames: bool = False,
                          show_progress: bool = True):
        """
        Load images for all sequences in dataset.
        """
        iterator = dataset.sequences
        if show_progress:
            iterator = tqdm(iterator, desc="Loading sequences")
        
        for sequence in iterator:
            self.load_sequence_images(sequence, load_full_frames, show_progress=False)
        
        # Validate loading
        total_frames = dataset.total_frames
        loaded_crops = sum(1 for s in dataset.sequences for f in s.frames if f.crop_image is not None)
        
        logger.info(f"Loaded {loaded_crops}/{total_frames} crop images ({loaded_crops/total_frames*100:.1f}%)")
        
        if load_full_frames:
            loaded_full = sum(1 for s in dataset.sequences for f in s.frames if f.full_image is not None)
            logger.info(f"Loaded {loaded_full}/{total_frames} full images ({loaded_full/total_frames*100:.1f}%)")
        
        return dataset
    
    def _resolve_path(self, path_str: str, base_dir: Path) -> Path:
        """
        Resolve image path relative to base directory.
        Handles both absolute and relative paths.
        """
        path = Path(path_str)
        
        # If absolute and exists, use as-is
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to base_dir
        relative_path = base_dir / path
        if relative_path.exists():
            return relative_path
        
        # Try without base_dir (path might already include it)
        if path.exists():
            return path
        
        # Try extracting just the filename and looking in base_dir
        filename = path.name
        direct_path = base_dir / filename
        if direct_path.exists():
            return direct_path
        
        # If nothing works, return the relative path and let it fail later
        logger.debug(f"Could not resolve path: {path_str}, using: {relative_path}")
        return relative_path
    
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Load a single image from disk.
        
        Returns: numpy array (H, W, 3) in RGB format, or None if loading fails
        """
        try:
            if not path.exists():
                logger.error(f"Image not found: {path}")
                return None
            
            # Load with OpenCV (BGR format)
            img = cv2.imread(str(path))
            
            if img is None:
                logger.error(f"Failed to load image (cv2.imread returned None): {path}")
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img_rgb
        
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return None
    
    def preload_batch(self, frames: List[Frame], load_full_frames: bool = False) -> List[Frame]:
        """
        Preload a batch of frames (useful for batched inference).
        
        Args:
            frames: List of Frame objects
            load_full_frames: Whether to load full frames
        
        Returns:
            List of frames with images loaded
        """
        for frame in frames:
            self.load_frame_images(frame, load_full_frames)
        
        return frames


class LazyImageLoader:
    """
    Lazy image loader that loads images on-demand.
    Useful for large datasets that don't fit in memory.
    """
    
    def __init__(self, crop_base_dir: str, frame_base_dir: Optional[str] = None):
        self.crop_base_dir = Path(crop_base_dir)
        self.frame_base_dir = Path(frame_base_dir) if frame_base_dir else None
        self._loader = ImageLoader(crop_base_dir, frame_base_dir, num_workers=1)
    
    def __call__(self, frame: Frame, load_full_frame: bool = False) -> Frame:
        """
        Load images for frame on demand.
        """
        # Only load if not already loaded
        if frame.crop_image is None:
            self._loader.load_frame_images(frame, load_full_frame)
        
        return frame
    
    def clear_frame(self, frame: Frame):
        """Clear loaded images from frame to free memory"""
        frame.crop_image = None
        frame.full_image = None
    
    def clear_sequence(self, sequence: Sequence):
        """Clear all loaded images from sequence"""
        for frame in sequence.frames:
            self.clear_frame(frame)


def create_image_loader(data_config, lazy: bool = False):
    """
    Factory function to create appropriate image loader.
    """
    if lazy:
        return LazyImageLoader(
            crop_base_dir=data_config.crop_base_dir,
            frame_base_dir=data_config.frame_base_dir
        )
    else:
        return ImageLoader(
            crop_base_dir=data_config.crop_base_dir,
            frame_base_dir=data_config.frame_base_dir,
            num_workers=4
        )
