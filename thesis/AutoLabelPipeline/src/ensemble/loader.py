"""
Loader for ensemble predictions from saved model outputs.
Handles CSV and JSON formats; normalizes labels and confidence scores.
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata about a model run"""
    model_name: str
    config_id: str
    mode: str  # "single" or "video"
    prompt_version: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class FramePrediction:
    """Single frame prediction from a model"""
    frame_id: int
    label: str
    confidence: float
    raw_output: Optional[str] = None
    model_name: str = ""


@dataclass
class FramePredictionDataset:
    """Unified dataset of predictions from multiple models"""
    predictions: Dict[Tuple[str, int], List[FramePrediction]] = field(default_factory=dict)
    # Key: (sequence_id, frame_id) -> List[FramePrediction] (one per model)
    
    ground_truth: Dict[Tuple[str, int], str] = field(default_factory=dict)
    # Key: (sequence_id, frame_id) -> ground_truth_label
    
    metadata: Dict[str, ModelMetadata] = field(default_factory=dict)
    # Key: model_name -> ModelMetadata
    
    sequence_info: Dict[str, Dict] = field(default_factory=dict)
    # Key: sequence_id -> {fps, num_frames, duration, ...}


class EnsembleLoader:
    """
    Load and normalize predictions from saved model runs.
    """
    
    # Standard label normalization mapping
    LABEL_MAP = {
        "both": "hazard",
        "signal": "hazard",
        "multiple": "hazard",
    }
    
    VALID_LABELS = {"left", "right", "hazard", "none"}
    
    def __init__(self, label_harmonization: bool = True,
                 calibrate_confidence: bool = False,
                 handle_missing_frames: str = "exclude",
                 transform_sequence_ids: bool = True):
        """
        Args:
            label_harmonization: Normalize labels (e.g., "both" -> "hazard")
            calibrate_confidence: Apply confidence calibration (not implemented yet)
            handle_missing_frames: Strategy for frames missing from a model
                - "exclude": Remove frame from ensemble
                - "abstain": Treat as abstention (not voting)
            transform_sequence_ids: Transform prediction format sequence_ids to ground truth format
                - Converts: '2024-07-09-16-49-42_mapping_tartu_streets_camera_wide_right_170__track_170'
                - To: '2024-07-09-16-49-42_mapping_tartu_streets/camera_wide_right_170'
        """
        self.label_harmonization = label_harmonization
        self.calibrate_confidence = calibrate_confidence
        self.handle_missing_frames = handle_missing_frames
        self.transform_sequence_ids = transform_sequence_ids
    
    def load_from_results_dir(self, results_dir: Path,
                               model_name: str,
                               model_config: Dict = None) -> FramePredictionDataset:
        """
        Load predictions from a single model run directory.
        
        Args:
            results_dir: Directory containing prediction files
            model_name: Name/identifier for this model
            model_config: Optional config dict with mode, prompt_version, etc.
        
        Returns:
            FramePredictionDataset with predictions from this model
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        dataset = FramePredictionDataset()
        config = model_config or {}
        
        # Create metadata for this model
        metadata = ModelMetadata(
            model_name=model_name,
            config_id=config.get("config_id", "unknown"),
            mode=config.get("mode", "single"),
            prompt_version=config.get("prompt_version"),
            timestamp=config.get("timestamp"),
        )
        dataset.metadata[model_name] = metadata
        
        # Find all prediction files (CSV or JSON), excluding metadata files
        all_csv_files = list(results_dir.glob("*.csv"))
        all_json_files = list(results_dir.glob("*.json"))
        
        # Filter out metadata/summary files
        csv_files = [f for f in all_csv_files 
                    if not any(x in f.name for x in ['dataset_summary', 'pipeline_report', 
                                                      'evaluation_metrics', 'evaluation_per_sequence'])]
        json_files = [f for f in all_json_files 
                     if not any(x in f.name for x in ['dataset_summary', 'pipeline_report', 
                                                       'evaluation_metrics', 'evaluation_per_sequence',
                                                       'review_queue'])]
        
        logger.info(f"Found {len(csv_files)} CSV and {len(json_files)} JSON prediction files in {results_dir}")
        
        # Load from CSV files
        for csv_file in csv_files:
            self._load_csv_file(csv_file, model_name, dataset)
        
        # Load from JSON files
        for json_file in json_files:
            self._load_json_file(json_file, model_name, dataset)
        
        if not dataset.predictions:
            logger.warning(f"No predictions found in {results_dir}")
        
        logger.info(f"Loaded {len(dataset.predictions)} unique frames from {model_name}")
        return dataset
    
    def _transform_sequence_id(self, raw_sequence_id: str) -> str:
        """
        Transform prediction sequence_id format to match ground truth format.
        
        Converts:
          '2024-07-09-16-49-42_mapping_tartu_streets_camera_wide_right_170__track_170'
        To:
          '2024-07-09-16-49-42_mapping_tartu_streets/camera_wide_right_170'
        
        Args:
            raw_sequence_id: Sequence ID from prediction filename
        
        Returns:
            Transformed sequence ID matching ground truth format
        """
        if not self.transform_sequence_ids:
            return raw_sequence_id
        
        def _strip_track_suffix(seq_id: str):
            if '__track_' not in seq_id:
                return seq_id, None
            base, track = seq_id.split('__track_', 1)
            return base, track

        # If already in ground truth slash format, only normalize track suffix (if present)
        if '/' in raw_sequence_id:
            base_part, track_id = _strip_track_suffix(raw_sequence_id)
            if not track_id:
                return base_part
            left, right = base_part.split('/', 1)
            if not right.endswith(f"_{track_id}"):
                right = f"{right}_{track_id}"
            transformed = f"{left}/{right}"
            if transformed != raw_sequence_id:
                if not hasattr(self, '_transformation_logged') or len(self._transformation_logged) < 3:
                    if not hasattr(self, '_transformation_logged'):
                        self._transformation_logged = set()
                    if raw_sequence_id not in self._transformation_logged:
                        logger.info(f"Transforming sequence_id: '{raw_sequence_id}' -> '{transformed}'")
                        self._transformation_logged.add(raw_sequence_id)
            return transformed

        # Extract track_id from __track_XXX pattern
        if '__track_' in raw_sequence_id:
            base_part, track_id = _strip_track_suffix(raw_sequence_id)
            track_id = track_id or ''

            # Find where "_camera_" appears in the string
            # Everything before "_camera_" goes on the left side of /
            # Everything from "camera_" onwards goes on the right side
            if '_camera_' in base_part:
                # Split at _camera_
                split_idx = base_part.find('_camera_')
                date_location = base_part[:split_idx]
                camera_part = base_part[split_idx + 1:]  # Skip the leading underscore, keep "camera_"

                # Ground truth format: date_location/camera_part (append track_id only if missing)
                if track_id and not camera_part.endswith(f"_{track_id}"):
                    camera_part = f"{camera_part}_{track_id}"
                transformed = f"{date_location}/{camera_part}"

                # Log transformation for visibility
                if not hasattr(self, '_transformation_logged') or len(self._transformation_logged) < 3:
                    if not hasattr(self, '_transformation_logged'):
                        self._transformation_logged = set()
                    if raw_sequence_id not in self._transformation_logged:
                        logger.info(f"Transforming sequence_id: '{raw_sequence_id}' -> '{transformed}'")
                        self._transformation_logged.add(raw_sequence_id)

                return transformed
            else:
                # Fallback: no _camera_ found, return as-is with warning
                logger.warning(f"Could not transform sequence_id (no '_camera_' found): {raw_sequence_id}")
                return raw_sequence_id
        
        # No transformation needed
        return raw_sequence_id
    
    def _load_csv_file(self, csv_path: Path, model_name: str,
                       dataset: FramePredictionDataset):
        """Load predictions from a CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            # Determine sequence_id from filename and transform to ground truth format
            raw_sequence_id = csv_path.stem
            sequence_id = self._transform_sequence_id(raw_sequence_id)
            
            for _, row in df.iterrows():
                frame_id = int(row["frame_id"])
                label = self._harmonize_label(str(row["label"]))
                confidence = float(row.get("confidence", 0.5))
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                prediction = FramePrediction(
                    frame_id=frame_id,
                    label=label,
                    confidence=confidence,
                    raw_output=row.get("raw_output"),
                    model_name=model_name,
                )
                
                key = (sequence_id, frame_id)
                if key not in dataset.predictions:
                    dataset.predictions[key] = []
                dataset.predictions[key].append(prediction)
            
            logger.debug(f"Loaded {len(df)} predictions from {csv_path.name}")
        
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
    
    def _load_json_file(self, json_path: Path, model_name: str,
                        dataset: FramePredictionDataset):
        """Load predictions from a JSON file"""
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Determine sequence_id from filename or metadata and transform
            raw_sequence_id = json_path.stem
            if "metadata" in data and "sequence_id" in data["metadata"]:
                raw_sequence_id = data["metadata"]["sequence_id"]
            
            sequence_id = self._transform_sequence_id(raw_sequence_id)
            
            # Extract predictions
            predictions = data.get("predictions", [])
            
            for pred in predictions:
                frame_id = int(pred.get("frame_id", 0))
                label = self._harmonize_label(str(pred.get("label", "none")))
                confidence = float(pred.get("confidence", 0.5))
                
                # Clamp confidence to [0, 1]
                confidence = max(0.0, min(1.0, confidence))
                
                prediction = FramePrediction(
                    frame_id=frame_id,
                    label=label,
                    confidence=confidence,
                    raw_output=pred.get("raw_output"),
                    model_name=model_name,
                )
                
                key = (sequence_id, frame_id)
                if key not in dataset.predictions:
                    dataset.predictions[key] = []
                dataset.predictions[key].append(prediction)
            
            logger.debug(f"Loaded {len(predictions)} predictions from {json_path.name}")
        
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
    
    def _harmonize_label(self, label: str) -> str:
        """Normalize label to standard set"""
        if label is None:
            return "none"
        
        label = str(label).strip().lower()
        
        # Apply harmonization map
        if self.label_harmonization:
            label = self.LABEL_MAP.get(label, label)
        
        # Validate
        if label not in self.VALID_LABELS:
            logger.warning(f"Invalid label '{label}'; defaulting to 'none'")
            return "none"
        
        return label
    
    def merge_datasets(self, *datasets: FramePredictionDataset) -> FramePredictionDataset:
        """
        Merge multiple FramePredictionDatasets into one.
        
        Args:
            *datasets: Variable number of FramePredictionDataset objects
        
        Returns:
            Single merged dataset
        """
        merged = FramePredictionDataset()
        
        for dataset in datasets:
            # Merge predictions
            for key, preds in dataset.predictions.items():
                if key not in merged.predictions:
                    merged.predictions[key] = []
                merged.predictions[key].extend(preds)
            
            # Merge metadata
            merged.metadata.update(dataset.metadata)
            
            # Merge ground truth (skip duplicates)
            for key, label in dataset.ground_truth.items():
                if key not in merged.ground_truth:
                    merged.ground_truth[key] = label
            
            # Merge sequence info (skip duplicates)
            for seq_id, info in dataset.sequence_info.items():
                if seq_id not in merged.sequence_info:
                    merged.sequence_info[seq_id] = info
        
        logger.info(f"Merged {len(datasets)} datasets into {len(merged.predictions)} predictions")
        return merged
    
    def load_ground_truth(self, csv_path: Path,
                          dataset: FramePredictionDataset) -> FramePredictionDataset:
        """
        Load ground truth labels from a CSV file.
        
        Expected columns: sequence_id, frame_id, label (or true_label)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Determine label column name
            label_col = None
            if "label" in df.columns:
                label_col = "label"
            elif "true_label" in df.columns:
                label_col = "true_label"
            else:
                raise ValueError("CSV must have 'label' or 'true_label' column")
            
            logger.info(f"Loading ground truth from {csv_path} (using column '{label_col}')")
            
            for _, row in df.iterrows():
                sequence_id = str(row.get("sequence_id", ""))
                frame_id = int(row.get("frame_id", 0))
                label = self._harmonize_label(str(row.get(label_col, "none")))
                
                key = (sequence_id, frame_id)
                dataset.ground_truth[key] = label
            
            logger.info(f"Loaded {len(dataset.ground_truth)} ground truth labels")
        
        except Exception as e:
            logger.error(f"Error loading ground truth from {csv_path}: {e}")
        
        return dataset
    
    def to_dataframe(self, dataset: FramePredictionDataset) -> pd.DataFrame:
        """
        Convert dataset to pandas DataFrame for easier manipulation.
        
        Returns:
            DataFrame with columns:
            - sequence_id
            - frame_id
            - model_name
            - label
            - confidence
            - ground_truth (if available)
        """
        rows = []
        
        for (seq_id, frame_id), predictions in dataset.predictions.items():
            for pred in predictions:
                row = {
                    "sequence_id": seq_id,
                    "frame_id": frame_id,
                    "model_name": pred.model_name,
                    "label": pred.label,
                    "confidence": pred.confidence,
                    "ground_truth": dataset.ground_truth.get((seq_id, frame_id), None),
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def validate_dataset(self, dataset: FramePredictionDataset) -> Dict[str, any]:
        """
        Validate dataset integrity and return summary statistics.
        
        Returns:
            Dict with validation report
        """
        report = {
            "num_frames": len(dataset.predictions),
            "models": list(dataset.metadata.keys()),
            "num_models": len(dataset.metadata),
            "label_distribution": {},
            "confidence_stats": {},
            "missing_frames_per_model": {},
        }
        
        # Collect all frame keys
        all_frames = set(dataset.predictions.keys())
        
        # Label distribution
        label_counts = {}
        for key, preds in dataset.predictions.items():
            for pred in preds:
                label_counts[pred.label] = label_counts.get(pred.label, 0) + 1
        report["label_distribution"] = label_counts
        
        # Confidence stats
        all_confidences = []
        for key, preds in dataset.predictions.items():
            for pred in preds:
                all_confidences.append(pred.confidence)
        
        if all_confidences:
            report["confidence_stats"] = {
                "mean": float(np.mean(all_confidences)),
                "std": float(np.std(all_confidences)),
                "min": float(np.min(all_confidences)),
                "max": float(np.max(all_confidences)),
            }
        
        # Missing frames per model
        for model_name in dataset.metadata.keys():
            model_frames = set()
            for key, preds in dataset.predictions.items():
                if any(p.model_name == model_name for p in preds):
                    model_frames.add(key)
            
            missing = all_frames - model_frames
            report["missing_frames_per_model"][model_name] = len(missing)
        
        return report
