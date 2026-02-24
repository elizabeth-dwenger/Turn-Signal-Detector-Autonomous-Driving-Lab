"""
Abstract base class for turn signal detection models.
All model implementations must inherit from this.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
import numpy as np
import logging
from pathlib import Path
import time


logger = logging.getLogger(__name__)


class TurnSignalDetector(ABC):
    """
    Abstract base class for all turn signal detection models.
    Provides common interface for video and single-image modes.
    """
    
    def __init__(self, model_config):
        """
        Initialize model with configuration.
        """
        self.config = model_config
        self.model_name = model_config.model_name_or_path
        self.device = model_config.device
        self.inference_mode = model_config.inference_mode
        
        # Model components (to be set by subclasses)
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Prompt
        self.prompt = self._load_prompt(model_config.prompt_template_path)
        
        # Metrics tracking
        self.metrics = {
            'total_inferences': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_latency_ms': 0.0,
            'total_frames_processed': 0,
        }
    
    @abstractmethod
    def warmup(self):
        """
        Load model and initialize.
        Must be called before inference.
        """
        pass
    
    @abstractmethod
    def predict_video(self, video: np.ndarray) -> Dict:
        """
        Predict turn signal state from video sequence.
        """
        pass
    
    @abstractmethod
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict turn signal state from single image.
        
        Args:
            image: Image tensor (H, W, C) in [0, 1] range
        
        Returns:
            Same format as predict_video
        """
        pass
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict turn signal state for batch of images.
        Default implementation calls predict_single sequentially.
        Subclasses can override for true batch processing.
        """
        return [self.predict_single(img) for img in images]
    
    def predict(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> Union[Dict, List[Dict]]:
        """
        Unified prediction interface.
        Automatically determines if input is video or images.
        """
        if isinstance(input_data, np.ndarray):
            # Single tensor - assume video
            return self.predict_video(input_data)
        elif isinstance(input_data, list):
            # List of images
            return self.predict_batch(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load prompt template from file"""
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        
        with open(path, 'r') as f:
            prompt = f.read().strip()
        
        logger.info(f"Loaded prompt template from {prompt_path} ({len(prompt)} chars)")
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse model response to extract label.
        """
        import json
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Extract fields
                label = parsed.get('label', 'none').lower()
                if label == 'both':
                    label = 'hazard'
                reasoning = parsed.get('reasoning', '')
                
                # Validate label
                valid_labels = {'left', 'right', 'none', 'both', 'hazard'}
                if label not in valid_labels:
                    logger.warning(f"Invalid label '{label}', defaulting to 'none'")
                    label = 'none'
                
                self.metrics['successful_parses'] += 1
                
                return {
                    'label': label,
                    'reasoning': reasoning
                }
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
                logger.debug(f"Response: {response[:200]}...")
        
        # Fallback: try to extract label from text
        response_lower = response.lower()
        
        for label in ['left', 'right', 'hazard', 'both', 'none']:
            if label in response_lower:
                if label == 'both':
                    label = 'hazard'
                logger.info(f"Extracted label '{label}' from text (no JSON)")
                self.metrics['failed_parses'] += 1
                return {
                    'label': label,
                    'reasoning': ''
                }
        
        # Complete failure
        logger.error(f"Could not parse response: {response[:200]}...")
        self.metrics['failed_parses'] += 1
        
        return {
            'label': 'none',
            'reasoning': 'Parse failed'
        }
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        """
        metrics = self.metrics.copy()
        
        if metrics['total_inferences'] > 0:
            metrics['avg_latency_ms'] = metrics['total_latency_ms'] / metrics['total_inferences']
            metrics['parse_success_rate'] = metrics['successful_parses'] / metrics['total_inferences']
        else:
            metrics['avg_latency_ms'] = 0.0
            metrics['parse_success_rate'] = 0.0
        
        if metrics['total_frames_processed'] > 0:
            metrics['avg_latency_per_frame_ms'] = metrics['total_latency_ms'] / metrics['total_frames_processed']
        else:
            metrics['avg_latency_per_frame_ms'] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.metrics = {
            'total_inferences': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_latency_ms': 0.0,
            'total_frames_processed': 0,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}', mode={self.inference_mode.value})"
