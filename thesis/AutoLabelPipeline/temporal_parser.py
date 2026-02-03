"""
Enhanced response parser with temporal localization support.
"""
import json
import re
from typing import Dict, Optional


class TemporalResponseParser:
    """
    Enhanced parser that extracts temporal information from model responses.
    
    Supports responses with:
    - start_frame: When signal begins
    - end_frame: When signal ends
    - frame_labels: Per-frame predictions (for validation)
    """
    
    def _parse_response_with_temporal(self, response: str) -> Dict:
        """
        Parse model response with temporal information.
        
        Returns dict with:
            - label: str
            - confidence: float
            - reasoning: str
            - start_frame: int or None
            - end_frame: int or None
            - start_time_seconds: float or None
            - end_time_seconds: float or None
            - raw_output: str
        """
        import json
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Extract standard fields
                label = parsed.get('label', 'none').lower()
                confidence = float(parsed.get('confidence', 0.5))
                reasoning = parsed.get('reasoning', '')
                
                # Extract temporal fields (frames)
                start_frame = parsed.get('start_frame')
                end_frame = parsed.get('end_frame')
                
                # Extract temporal fields (time in seconds)
                start_time = parsed.get('start_time_seconds')
                end_time = parsed.get('end_time_seconds')
                
                # Convert to appropriate types
                if start_frame is not None:
                    try:
                        start_frame = int(start_frame)
                    except (ValueError, TypeError):
                        start_frame = None
                
                if end_frame is not None:
                    try:
                        end_frame = int(end_frame)
                    except (ValueError, TypeError):
                        end_frame = None
                
                if start_time is not None:
                    try:
                        start_time = float(start_time)
                    except (ValueError, TypeError):
                        start_time = None
                
                if end_time is not None:
                    try:
                        end_time = float(end_time)
                    except (ValueError, TypeError):
                        end_time = None
                
                # Validate label
                valid_labels = {'left', 'right', 'none', 'both'}
                if label not in valid_labels:
                    label = 'none'
                    confidence = 0.3
                
                # Validate confidence
                confidence = max(0.0, min(1.0, confidence))
                
                # Validate temporal consistency
                if start_frame is not None and end_frame is not None:
                    if end_frame < start_frame:
                        # Invalid range, clear temporal info
                        start_frame = None
                        end_frame = None
                        start_time = None
                        end_time = None
                
                if start_time is not None and end_time is not None:
                    if end_time < start_time:
                        # Invalid range
                        start_time = None
                        end_time = None
                
                # If we have frames but not time, calculate time (assuming 10 FPS)
                fps = 10.0
                if start_frame is not None and start_time is None:
                    start_time = start_frame / fps
                if end_frame is not None and end_time is None:
                    end_time = end_frame / fps
                
                return {
                    'label': label,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_seconds': start_time,
                    'end_time_seconds': end_time,
                    'has_temporal_info': start_frame is not None or end_frame is not None
                }
            
            except json.JSONDecodeError as e:
                # Fall through to fallback parsing
                pass
        
        # Fallback parsing (no JSON found)
        response_lower = response.lower()
        
        for label in ['left', 'right', 'both', 'none']:
            if label in response_lower:
                return {
                    'label': label,
                    'confidence': 0.5,
                    'reasoning': '',
                    'start_frame': None,
                    'end_frame': None,
                    'start_time_seconds': None,
                    'end_time_seconds': None,
                    'has_temporal_info': False
                }
        
        # Complete failure
        return {
            'label': 'none',
            'confidence': 0.0,
            'reasoning': 'Parse failed',
            'start_frame': None,
            'end_frame': None,
            'start_time_seconds': None,
            'end_time_seconds': None,
            'has_temporal_info': False
        }
    
    def compute_temporal_metrics(self, predictions: Dict,
                                 ground_truth_start: Optional[int] = None,
                                 ground_truth_end: Optional[int] = None) -> Dict:
        """
        Compute temporal localization metrics.
        """
        if ground_truth_start is None or ground_truth_end is None:
            return {
                'temporal_iou': None,
                'start_error': None,
                'end_error': None,
                'has_gt': False
            }
        
        pred_start = predictions.get('start_frame')
        pred_end = predictions.get('end_frame')
        
        if pred_start is None or pred_end is None:
            return {
                'temporal_iou': 0.0,
                'start_error': None,
                'end_error': None,
                'has_gt': True
            }
        
        # Compute IoU
        intersection_start = max(pred_start, ground_truth_start)
        intersection_end = min(pred_end, ground_truth_end)
        
        if intersection_end >= intersection_start:
            intersection = intersection_end - intersection_start + 1
        else:
            intersection = 0
        
        union_start = min(pred_start, ground_truth_start)
        union_end = max(pred_end, ground_truth_end)
        union = union_end - union_start + 1
        
        iou = intersection / union if union > 0 else 0.0
        
        # Compute errors
        start_error = abs(pred_start - ground_truth_start)
        end_error = abs(pred_end - ground_truth_end)
        
        return {
            'temporal_iou': iou,
            'start_error': start_error,
            'end_error': end_error,
            'has_gt': True
        }


# Example usage in your model class:

class EnhancedCosmosDetector(TurnSignalDetector, TemporalResponseParser):
    """Cosmos detector with temporal localization support"""
    
    def predict_video(self, video):
        # ... existing inference code ...
        
        # Use enhanced parser instead of basic one
        parsed = self._parse_response_with_temporal(response)
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        if parsed.get('has_temporal_info'):
            self.metrics['temporal_predictions'] = self.metrics.get('temporal_predictions', 0) + 1
        
        return {
            'label': parsed['label'],
            'confidence': parsed['confidence'],
            'reasoning': parsed.get('reasoning', ''),
            'start_frame': parsed.get('start_frame'),
            'end_frame': parsed.get('end_frame'),
            'raw_output': response,
            'latency_ms': latency_ms
        }


# Example usage for evaluation:

def evaluate_temporal_predictions(results, ground_truth_data):
    """
    Evaluate temporal localization performance.
    """
    parser = TemporalResponseParser()
    
    temporal_ious = []
    start_errors = []
    end_errors = []
    
    for pred, gt in zip(results, ground_truth_data):
        if gt.get('start_frame') is not None:
            metrics = parser.compute_temporal_metrics(
                pred,
                gt['start_frame'],
                gt['end_frame']
            )
            
            if metrics['temporal_iou'] is not None:
                temporal_ious.append(metrics['temporal_iou'])
            if metrics['start_error'] is not None:
                start_errors.append(metrics['start_error'])
            if metrics['end_error'] is not None:
                end_errors.append(metrics['end_error'])
    
    import numpy as np
    
    return {
        'mean_temporal_iou': np.mean(temporal_ious) if temporal_ious else None,
        'median_temporal_iou': np.median(temporal_ious) if temporal_ious else None,
        'mean_start_error': np.mean(start_errors) if start_errors else None,
        'mean_end_error': np.mean(end_errors) if end_errors else None,
        'n_evaluated': len(temporal_ious)
    }
