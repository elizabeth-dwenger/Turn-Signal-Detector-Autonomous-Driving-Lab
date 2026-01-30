"""
Output formatters for saving predictions in various formats.
Supports CSV, JSON, and COCO formats.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class CSVFormatter:
    """
    Saves predictions to CSV format.
    Compatible with your original tracking data format.
    """
    
    @staticmethod
    def save(predictions_by_sequence: Dict[str, List[Dict]],
             output_path: str,
             include_confidence: bool = True,
             include_metadata: bool = True):
        """
        Save predictions to CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV columns
        columns = [
            'sequence_id',
            'frame_id',
            'label',
        ]
        
        if include_confidence:
            columns.append('confidence')
        
        if include_metadata:
            columns.extend([
                'original_label',
                'smoothed',
                'reconstructed',
                'constraint_enforced',
                'flagged'
            ])
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for sequence_id, predictions in predictions_by_sequence.items():
                for pred in predictions:
                    row = {
                        'sequence_id': sequence_id,
                        'frame_id': pred.get('frame_id', 0),
                        'label': pred['label'],
                    }
                    
                    if include_confidence:
                        row['confidence'] = f"{pred['confidence']:.4f}"
                    
                    if include_metadata:
                        row['original_label'] = pred.get('original_label', '')
                        row['smoothed'] = pred.get('smoothed', False)
                        row['reconstructed'] = pred.get('reconstructed', False)
                        row['constraint_enforced'] = pred.get('constraint_enforced', False)
                        row['flagged'] = pred.get('flagged', False)
                    
                    writer.writerow(row)
        
        logger.info(f"Saved CSV to {output_path}")


class JSONFormatter:
    """
    Saves predictions to JSON format.
    Includes full metadata and is easy to parse.
    """
    
    @staticmethod
    def save(predictions_by_sequence: Dict[str, List[Dict]],
             output_path: str,
             include_raw_output: bool = False,
             metadata: Optional[Dict] = None):
        """
        Save predictions to JSON file.
        
        Args:
            predictions_by_sequence: Dict mapping sequence_id to prediction list
            output_path: Path to output JSON file
            include_raw_output: Whether to include raw model outputs
            metadata: Additional metadata to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build output structure
        output = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'sequences': {}
        }
        
        for sequence_id, predictions in predictions_by_sequence.items():
            sequence_data = {
                'sequence_id': sequence_id,
                'num_frames': len(predictions),
                'predictions': []
            }
            
            for pred in predictions:
                pred_data = {
                    'frame_id': pred.get('frame_id', 0),
                    'label': pred['label'],
                    'confidence': pred['confidence'],
                }
                
                # Optional fields
                if pred.get('reasoning'):
                    pred_data['reasoning'] = pred['reasoning']
                
                if include_raw_output and pred.get('raw_output'):
                    pred_data['raw_output'] = pred['raw_output']
                
                # Processing metadata
                if pred.get('original_label'):
                    pred_data['original_label'] = pred['original_label']
                
                if pred.get('smoothed'):
                    pred_data['smoothed'] = True
                
                if pred.get('reconstructed'):
                    pred_data['reconstructed'] = True
                
                if pred.get('constraint_enforced'):
                    pred_data['constraint_enforced'] = True
                
                if pred.get('flags'):
                    pred_data['flags'] = pred['flags']
                
                sequence_data['predictions'].append(pred_data)
            
            output['sequences'][sequence_id] = sequence_data
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved JSON to {output_path}")


class COCOFormatter:
    """
    Saves predictions in COCO format.
    Compatible with many visualization and evaluation tools.
    """
    
    @staticmethod
    def save(predictions_by_sequence: Dict[str, List[Dict]],
             output_path: str,
             image_info: Optional[Dict] = None):
        """
        Save predictions to COCO format.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # COCO structure
        coco_output = {
            'info': {
                'description': 'Turn Signal Detection Predictions',
                'version': '1.0',
                'date_created': datetime.now().isoformat()
            },
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 0, 'name': 'none', 'supercategory': 'turn_signal'},
                {'id': 1, 'name': 'left', 'supercategory': 'turn_signal'},
                {'id': 2, 'name': 'right', 'supercategory': 'turn_signal'},
                {'id': 3, 'name': 'both', 'supercategory': 'turn_signal'},
            ]
        }
        
        # Map labels to category IDs
        label_to_cat_id = {
            'none': 0,
            'left': 1,
            'right': 2,
            'both': 3
        }
        
        image_id = 0
        annotation_id = 0
        
        for sequence_id, predictions in predictions_by_sequence.items():
            for pred in predictions:
                frame_id = pred.get('frame_id', 0)
                
                # Add image entry
                image_entry = {
                    'id': image_id,
                    'sequence_id': sequence_id,
                    'frame_id': frame_id,
                    'file_name': f"{sequence_id}_frame_{frame_id:06d}.jpg",
                }
                
                # Add image metadata if provided
                if image_info and sequence_id in image_info:
                    info = image_info[sequence_id]
                    image_entry['width'] = info.get('width', 0)
                    image_entry['height'] = info.get('height', 0)
                
                coco_output['images'].append(image_entry)
                
                # Add annotation
                label = pred['label']
                category_id = label_to_cat_id.get(label, 0)
                
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'confidence': pred['confidence'],
                }
                
                coco_output['annotations'].append(annotation)
                
                image_id += 1
                annotation_id += 1
        
        # Write COCO JSON
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        logger.info(f"Saved COCO format to {output_path}")


class ReviewQueueFormatter:
    """
    Formats flagged predictions for manual review.
    Creates a simple format for annotation tools.
    """
    
    @staticmethod
    def save(quality_reports: Dict[str, Dict],
             predictions_by_sequence: Dict[str, List[Dict]],
             output_path: str):
        """
        Save review queue to JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        review_queue = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'description': 'Frames flagged for manual review'
            },
            'total_flagged': 0,
            'frames': []
        }
        
        for sequence_id, quality_report in quality_reports.items():
            if not quality_report or 'flagged_frames' not in quality_report:
                continue
            
            predictions = predictions_by_sequence.get(sequence_id, [])
            
            for flagged in quality_report['flagged_frames']:
                frame_idx = flagged['frame_index']
                
                # Get full prediction
                if frame_idx < len(predictions):
                    pred = predictions[frame_idx]
                    
                    frame_entry = {
                        'sequence_id': sequence_id,
                        'frame_id': pred.get('frame_id', frame_idx),
                        'frame_index': frame_idx,
                        'label': pred['label'],
                        'confidence': pred['confidence'],
                        'flags': flagged['flags'],
                        'needs_review': True
                    }
                    
                    review_queue['frames'].append(frame_entry)
                    review_queue['total_flagged'] += 1
        
        # Write review queue
        with open(output_path, 'w') as f:
            json.dump(review_queue, f, indent=2)
        
        logger.info(f"Saved review queue to {output_path} ({review_queue['total_flagged']} frames)")


def save_predictions(predictions_by_sequence: Dict[str, List[Dict]],
                    output_dir: str,
                    formats: List[str] = ['csv', 'json'],
                    config: Optional[Dict] = None) -> Dict[str, str]:
    """
    Save predictions in multiple formats.
    
    Args:
        predictions_by_sequence: Dict mapping sequence_id to prediction list
        output_dir: Output directory
        formats: List of formats to save ('csv', 'json', 'coco')
        config: Optional configuration dict
    
    Returns:
        Dict mapping format name to output file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # CSV
    if 'csv' in formats:
        csv_path = output_dir / 'predictions.csv'
        CSVFormatter.save(
            predictions_by_sequence,
            str(csv_path),
            include_confidence=config.get('include_confidence', True) if config else True,
            include_metadata=True
        )
        output_files['csv'] = str(csv_path)
    
    # JSON
    if 'json' in formats:
        json_path = output_dir / 'predictions.json'
        JSONFormatter.save(
            predictions_by_sequence,
            str(json_path),
            include_raw_output=config.get('include_raw_output', False) if config else False,
            metadata=config.get('metadata') if config else None
        )
        output_files['json'] = str(json_path)
    
    # COCO
    if 'coco' in formats:
        coco_path = output_dir / 'predictions_coco.json'
        COCOFormatter.save(
            predictions_by_sequence,
            str(coco_path),
            image_info=config.get('image_info') if config else None
        )
        output_files['coco'] = str(coco_path)
    
    return output_files
