"""
CSV loader for turn signal tracking data.
Loads CSV file and constructs Sequence objects.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
from collections import defaultdict

from .data_structures import Frame, Sequence, Dataset, BoundingBox


logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Loads tracking data from CSV file.
    
    CSV Format:
    sequence,track_id,frame_id,class_id,score,x1,y1,x2,y2,crop_path,img_path, width,height,sequence_id,true_label,sampled_frame_id,signal_start_frame
    
    fyi not everything in the csv is needed (and "signal_start_frame" I don't think is correct)
    """
    
    def __init__(self, data_config):
        self.config = data_config
        self.csv_path = Path(data_config.input_csv)
        self.crop_base_dir = Path(data_config.crop_base_dir)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if not self.crop_base_dir.exists():
            raise FileNotFoundError(f"Crop base directory not found: {self.crop_base_dir}")
    
    def load(self) -> Dataset:
        logger.info(f"Loading CSV from: {self.csv_path}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Validate required columns
        self._validate_columns(df)
        
        # Convert to Frame objects
        frames = self._parse_frames(df)
        logger.info(f"Parsed {len(frames)} frames")
        
        # Group into sequences
        sequences = self._group_into_sequences(frames)
        logger.info(f"Grouped into {len(sequences)} sequences")
        
        # Create dataset
        dataset = Dataset(
            sequences=sequences,
            source_csv=str(self.csv_path)
        )
        
        # Apply filters from config
        if self.config.sequence_filter or self.config.max_sequences:
            logger.info("Applying filters...")
            dataset = dataset.filter_sequences(
                sequence_ids=self.config.sequence_filter,
                max_sequences=self.config.max_sequences
            )
            logger.info(f"After filtering: {dataset.num_sequences} sequences, {dataset.total_frames} frames")
        
        logger.info(f"Dataset loaded: {dataset}")
        return dataset
    
    def _validate_columns(self, df: pd.DataFrame):
        """Validate that all required columns exist"""
        required_cols = [
            'sequence', 'track_id', 'frame_id', 'class_id', 'score',
            'x1', 'y1', 'x2', 'y2', 'crop_path', 'img_path',
            'width', 'height', 'sequence_id'
        ]
        
        optional_cols = ['true_label', 'sampled_frame_id', 'signal_start_frame']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add optional columns if missing
        for col in optional_cols:
            if col not in df.columns:
                df[col] = None
                logger.warning(f"Optional column '{col}' not found, using None")
        
        # If true_label is all None / placeholder (e.g. 'none'), mark as unlabeled
        if 'true_label' in df.columns:
            unique_labels = set(df['true_label'].dropna().astype(str).str.strip().str.lower().unique())
            placeholder_labels = {'none', 'unknown', 'unlabeled', ''}
            if unique_labels.issubset(placeholder_labels):
                logger.info("All true_label values are placeholders — treating dataset as unlabeled")
                df['true_label'] = None
    
    def _parse_frames(self, df: pd.DataFrame) -> List[Frame]:
        """Convert DataFrame rows to Frame objects"""
        frames = []
        
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            try:
                frame = Frame(
                    sequence=getattr(row, 'sequence'),
                    track_id=int(getattr(row, 'track_id')),
                    frame_id=int(getattr(row, 'frame_id')),
                    class_id=int(getattr(row, 'class_id')),
                    score=float(getattr(row, 'score')),
                    bbox=BoundingBox(
                        x1=float(getattr(row, 'x1')),
                        y1=float(getattr(row, 'y1')),
                        x2=float(getattr(row, 'x2')),
                        y2=float(getattr(row, 'y2'))
                    ),
                    crop_path=str(getattr(row, 'crop_path')),
                    img_path=str(getattr(row, 'img_path')),
                    img_width=int(getattr(row, 'width')),
                    img_height=int(getattr(row, 'height')),
                    sequence_id=str(getattr(row, 'sequence_id')),
                    true_label=str(getattr(row, 'true_label')) if pd.notna(getattr(row, 'true_label')) else None,
                    sampled_frame_id=int(getattr(row, 'sampled_frame_id')) if pd.notna(getattr(row, 'sampled_frame_id')) else -1,
                    signal_start_frame=int(getattr(row, 'signal_start_frame')) if pd.notna(getattr(row, 'signal_start_frame')) else -1
                )
                frames.append(frame)
            
            except Exception as e:
                logger.error(f"Error parsing row {idx}: {e}")
                logger.error(f"Row data: {row.to_dict()}")
                raise
        
        return frames
    
    def _group_into_sequences(self, frames: List[Frame]) -> List[Sequence]:
        """Group frames by sequence_id and track_id"""
        # Group frames: (sequence_id, track_id) -> List[Frame]
        grouped = defaultdict(list)
        
        for frame in frames:
            key = (frame.sequence_id, frame.track_id)
            grouped[key].append(frame)
        
        # Create Sequence objects
        sequences = []
        for (sequence_id, track_id), frame_list in grouped.items():
            sequence = Sequence(
                sequence_id=sequence_id,
                track_id=track_id,
                frames=frame_list
            )
            sequences.append(sequence)
        
        return sequences
    
    def load_statistics(self) -> dict:
        """
        Load and compute statistics without loading images.
        Useful for quick inspection.
        """
        dataset = self.load()
        
        stats = {
            'num_sequences': dataset.num_sequences,
            'total_frames': dataset.total_frames,
            'unique_sequence_ids': len(dataset.sequence_ids),
            'avg_frames_per_sequence': dataset.total_frames / dataset.num_sequences if dataset.num_sequences > 0 else 0,
            'sequence_length_min': min(s.num_frames for s in dataset.sequences) if dataset.sequences else 0,
            'sequence_length_max': max(s.num_frames for s in dataset.sequences) if dataset.sequences else 0,
        }
        
        # Ground truth statistics
        sequences_with_gt = [s for s in dataset.sequences if s.has_ground_truth]
        if sequences_with_gt:
            from collections import Counter
            label_counts = Counter(s.ground_truth_label for s in sequences_with_gt)
            stats['ground_truth_sequences'] = len(sequences_with_gt)
            stats['ground_truth_labels'] = dict(label_counts)
        else:
            stats['ground_truth_sequences'] = 0
            stats['ground_truth_labels'] = {}
        
        return stats


def load_dataset_from_config(data_config) -> Dataset:
    """
    Convenience function to load dataset from config.
    """
    loader = CSVLoader(data_config)
    return loader.load()
