"""
Output formatters for saving predictions in various formats.
Supports CSV, JSON, and COCO formats.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class CSVFormatter:
    """Format predictions as CSV file"""
    
    @staticmethod
    def save(predictions: List[Dict], output_path: str):
        """
        Save predictions to CSV file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define columns
        fieldnames = ['frame_id', 'label']
        
        # Add optional fields if present
        if predictions and 'original_label' in predictions[0]:
            fieldnames.append('original_label')
        if predictions and 'smoothed' in predictions[0]:
            fieldnames.append('smoothed')
        if predictions and 'flagged' in predictions[0]:
            fieldnames.append('flagged')
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for pred in predictions:
                row = {k: pred.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        logger.info(f"Saved {len(predictions)} predictions to CSV: {output_path}")


class JSONFormatter:
    """Format predictions as JSON file"""
    
    @staticmethod
    def save(predictions: List[Dict], output_path: str,
             metadata: Dict = None, include_raw_output: bool = False):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output
        output = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'num_predictions': len(predictions),
            'predictions': []
        }
        
        # Clean predictions
        for pred in predictions:
            clean_pred = pred.copy()
            
            # Remove raw_output if not requested
            if not include_raw_output and 'raw_output' in clean_pred:
                del clean_pred['raw_output']
            
            output['predictions'].append(clean_pred)
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(predictions)} predictions to JSON: {output_path}")


class COCOFormatter:
    """Format predictions in COCO format for compatibility with vision tools"""
    
    @staticmethod
    def save(predictions: List[Dict], output_path: str,
             sequence_info: Dict = None):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        coco_output = {
            'info': {
                'description': 'Turn Signal Detection Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'none'},
                {'id': 1, 'name': 'left'},
                {'id': 2, 'name': 'right'},
                {'id': 3, 'name': 'hazard'}
            ]
        }
        
        # Add sequence info if provided
        if sequence_info:
            coco_output['info'].update(sequence_info)
        
        # Category mapping
        category_map = {'none': 0, 'left': 1, 'right': 2, 'hazard': 3, 'both': 3}
        
        # Create images and annotations
        for i, pred in enumerate(predictions):
            frame_id = pred.get('frame_id', i)
            
            # Image entry
            image = {
                'id': frame_id,
                'file_name': f"frame_{frame_id:06d}.jpg",
                'width': pred.get('width', 640),
                'height': pred.get('height', 480)
            }
            coco_output['images'].append(image)
            
            # Annotation entry
            annotation = {
                'id': i,
                'image_id': frame_id,
                'category_id': category_map.get(pred['label'], 0),
                'attributes': {
                    'smoothed': pred.get('smoothed', False),
                    'flagged': pred.get('flagged', False)
                }
            }
            coco_output['annotations'].append(annotation)
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        logger.info(f"Saved {len(predictions)} predictions to COCO format: {output_path}")


class SequenceFormatter:
    """Format predictions organized by sequences"""
    
    @staticmethod
    def save_multiple_sequences(sequences_predictions: Dict[str, List[Dict]],
                                output_dir: str, format: str = 'json'):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formatters = {
            'json': JSONFormatter,
            'csv': CSVFormatter,
            'coco': COCOFormatter
        }
        
        formatter = formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        
        # Save each sequence
        for sequence_id, predictions in sequences_predictions.items():
            # Clean sequence_id for filename
            safe_id = sequence_id.replace('/', '_').replace('\\', '_')
            filename = f"{safe_id}.{format if format != 'coco' else 'json'}"
            file_path = output_path / filename
            
            if format == 'coco':
                formatter.save(
                    predictions,
                    str(file_path),
                    sequence_info={'sequence_id': sequence_id}
                )
            else:
                formatter.save(predictions, str(file_path))
        
        logger.info(f"Saved {len(sequences_predictions)} sequences to {output_dir}")


class ReviewQueueFormatter:
    """Format flagged frames for manual review"""
    
    @staticmethod
    def save(flagged_frames: List[Dict], output_path: str,
             sequence_id: str = None):
        """
        Save review queue (flagged frames).
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        review_queue = {
            'metadata': {
                'sequence_id': sequence_id,
                'timestamp': datetime.now().isoformat(),
                'num_flagged': len(flagged_frames)
            },
            'flagged_frames': flagged_frames
        }
        
        with open(output_path, 'w') as f:
            json.dump(review_queue, f, indent=2)
        
        logger.info(f"Saved review queue with {len(flagged_frames)} flagged frames: {output_path}")


def save_predictions(predictions: List[Dict],
                    output_path: str,
                    format: str = 'json',
                    **kwargs):
    """
    Convenience function to save predictions.
    """
    formatters = {
        'json': JSONFormatter,
        'csv': CSVFormatter,
        'coco': COCOFormatter
    }
    
    formatter = formatters.get(format)
    if not formatter:
        raise ValueError(f"Unknown format: {format}. Choose from: {list(formatters.keys())}")
    
    formatter.save(predictions, output_path, **kwargs)
