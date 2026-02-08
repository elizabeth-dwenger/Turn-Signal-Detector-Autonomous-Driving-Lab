"""
Data structures for tracking and frame information.
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import numpy as np


@dataclass
class BoundingBox:
    """Vehicle bounding box in frame"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Frame:
    """Single frame in a sequence"""
    # CSV columns
    sequence: str  # e.g., "2024-03-25-15-40-16_mapping_tartu/camera_fl_2"
    track_id: int
    frame_id: int
    class_id: int
    score: float  # Detection score
    bbox: BoundingBox
    
    # Paths
    crop_path: str  # Path to cropped vehicle image
    img_path: str   # Path to full frame image
    
    # Image dimensions
    img_width: int
    img_height: int
    
    # Sequence info
    sequence_id: str
    
    # Ground truth (if available)
    true_label: Optional[str] = None  # "left", "right", "none", etc.
    sampled_frame_id: int = -1
    signal_start_frame: int = -1
    
    # Loaded image data (populated later)
    crop_image: Optional[np.ndarray] = None
    full_image: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate paths exist when loading"""
        # Will be validated during image loading
        pass
    
    @property
    def has_ground_truth(self) -> bool:
        """Check if this frame has ground truth label"""
        return self.true_label is not None and self.true_label != ""
    
    @property
    def crop_path_absolute(self) -> Path:
        """Get absolute crop path"""
        return Path(self.crop_path)
    
    @property
    def img_path_absolute(self) -> Path:
        """Get absolute image path"""
        return Path(self.img_path)


@dataclass
class Sequence:
    """A sequence of frames showing the same tracked vehicle"""
    sequence_id: str
    track_id: int
    frames: List[Frame]
    
    # Metadata (extracted from sequence_id)
    video_name: str = ""
    camera_name: str = ""
    
    def __post_init__(self):
        """Sort frames by frame_id and extract metadata"""
        # Sort frames
        self.frames.sort(key=lambda f: f.frame_id)
        
        # Extract video and camera name from sequence_id
        # Format: "2024-03-25-15-40-16_mapping_tartu/camera_fl_2"
        parts = self.sequence_id.split('/')
        if len(parts) >= 2:
            self.video_name = parts[0]
            self.camera_name = parts[1]
        else:
            self.video_name = self.sequence_id
            self.camera_name = "unknown"
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def duration_frames(self) -> int:
        """Duration in frames"""
        if not self.frames:
            return 0
        return self.frames[-1].frame_id - self.frames[0].frame_id + 1
    
    @property
    def start_frame(self) -> int:
        return self.frames[0].frame_id if self.frames else 0
    
    @property
    def end_frame(self) -> int:
        return self.frames[-1].frame_id if self.frames else 0
    
    @property
    def has_ground_truth(self) -> bool:
        """Check if sequence has ground truth labels"""
        return any(f.has_ground_truth for f in self.frames)
    
    @property
    def ground_truth_label(self) -> Optional[str]:
        """Get most common ground truth label"""
        if not self.has_ground_truth:
            return None
        
        labels = [f.true_label for f in self.frames if f.has_ground_truth]
        if not labels:
            return None
        
        # Return most common label
        from collections import Counter
        return Counter(labels).most_common(1)[0][0]
    
    def get_frame_by_id(self, frame_id: int) -> Optional[Frame]:
        """Get specific frame by frame_id"""
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        return None
    
    def __repr__(self) -> str:
        return f"Sequence(id='{self.sequence_id}', track={self.track_id}, frames={self.num_frames})"


@dataclass
class Dataset:
    """Collection of sequences"""
    sequences: List[Sequence]
    source_csv: str
    
    def __post_init__(self):
        """Sort sequences by sequence_id"""
        self.sequences.sort(key=lambda s: (s.sequence_id, s.track_id))
    
    @property
    def num_sequences(self) -> int:
        return len(self.sequences)
    
    @property
    def total_frames(self) -> int:
        return sum(s.num_frames for s in self.sequences)
    
    @property
    def sequence_ids(self) -> List[str]:
        """Get unique sequence IDs"""
        return sorted(set(s.sequence_id for s in self.sequences))
    
    def get_sequences_by_id(self, sequence_id: str) -> List[Sequence]:
        """Get all sequences (different tracks) for a given sequence_id"""
        return [s for s in self.sequences if s.sequence_id == sequence_id]
    
    def filter_sequences(self, sequence_ids: Optional[List[str]] = None,
                        max_sequences: Optional[int] = None) -> 'Dataset':
        """Filter dataset by sequence IDs or max count"""
        filtered = self.sequences
        
        if sequence_ids:
            filtered = [s for s in filtered if s.sequence_id in sequence_ids]
        
        if max_sequences:
            filtered = filtered[:max_sequences]
        
        return Dataset(
            sequences=filtered,
            source_csv=self.source_csv
        )
    
    def __repr__(self) -> str:
        return f"Dataset(sequences={self.num_sequences}, frames={self.total_frames}, source='{self.source_csv}')"
