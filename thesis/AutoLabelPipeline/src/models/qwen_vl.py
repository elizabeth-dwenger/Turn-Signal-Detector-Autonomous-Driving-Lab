"""
Qwen VL model implementation for turn signal detection.
Supports both Qwen3-VL and Qwen2.5-VL models.
Handles both video (multi-image) and single-image modes.
"""
import torch
import numpy as np
from typing import List, Dict
import time
import logging
from PIL import Image

from .base import TurnSignalDetector


logger = logging.getLogger(__name__)


class QwenVLDetector(TurnSignalDetector):
    """
    Qwen VL model for turn signal detection.
    Works with Qwen3-VL and Qwen2.5-VL.
    """
    
    def warmup(self):
        """Load Qwen VL model"""
        logger.info(f"Loading Qwen VL model: {self.model_name}")
        
        # Check CUDA availability and adjust device if needed
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Import AutoProcessor first (needed for all)
        from transformers import AutoProcessor
        
        # Determine which model class to use based on model name
        # Both Qwen2.5-VL and Qwen3-VL use the newer AutoModelForImageTextToText class
        # Only Qwen2-VL (2.0) uses the old Qwen2VLForConditionalGeneration class
        
        model_name_lower = self.model_name.lower()
        
        if "qwen3" in model_name_lower or "qwen2.5" in model_name_lower or "qwen-2.5" in model_name_lower:
            # Qwen3-VL and Qwen2.5-VL both use the newer class
            from transformers import AutoModelForImageTextToText
            logger.info(f"Detected Qwen3-VL or Qwen2.5-VL model, using AutoModelForImageTextToText")
            model_class = AutoModelForImageTextToText
        else:
            # Qwen2-VL (2.0) uses the old class
            from transformers import Qwen2VLForConditionalGeneration
            logger.info("Detected Qwen2-VL model, using Qwen2VLForConditionalGeneration")
            model_class = Qwen2VLForConditionalGeneration
        
        # Load processor (handles images + text)
        processor_kwargs = {}
        if 'trust_remote_code' in self.config.model_kwargs:
            processor_kwargs['trust_remote_code'] = self.config.model_kwargs['trust_remote_code']
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs
        )
        
        # Prepare model loading kwargs
        model_load_kwargs = self.config.model_kwargs.copy()
        
        # Convert torch_dtype to dtype (new API)
        if 'torch_dtype' in model_load_kwargs:
            dtype_value = model_load_kwargs.pop('torch_dtype')
            if isinstance(dtype_value, str):
                dtype_map = {
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16,
                    'float32': torch.float32,
                }
                dtype_value = dtype_map.get(dtype_value, torch.bfloat16)
            model_load_kwargs['dtype'] = dtype_value
        
        # Handle device_map
        if 'device_map' not in model_load_kwargs:
            if self.device == "cpu":
                model_load_kwargs['device_map'] = "cpu"
            else:
                model_load_kwargs['device_map'] = self.device
        
        # For CPU, use float32 instead of bfloat16
        if self.device == "cpu" and 'dtype' in model_load_kwargs:
            if model_load_kwargs['dtype'] == torch.bfloat16:
                logger.warning("BFloat16 not well supported on CPU, using float32 instead")
                model_load_kwargs['dtype'] = torch.float32
        
        logger.info(f"Loading model with device={self.device}, dtype={model_load_kwargs.get('dtype', 'default')}")
        
        # Load the model with the appropriate class
        self.model = model_class.from_pretrained(
            self.model_name,
            **model_load_kwargs
        )
        
        self.model.eval()
        
        logger.info(f"  Qwen VL model loaded on {self.device}")
        logger.info(f"  Model type: {self.model.config.model_type}")
        logger.info(f"  Dtype: {self.model.dtype}")
    
    def predict_video(self, video: np.ndarray) -> Dict:
        """
        Predict from video sequence (multi-image input).
        
        Args:
            video: (T, H, W, C) in [0, 1] range
        
        Returns:
            Prediction dict
        """
        start_time = time.time()
        
        # Convert video frames to PIL Images
        images = self._video_to_pil_images(video)
        
        # Create multi-image message
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        
        # Add text prompt
        content.append({"type": "text", "text": self.prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse response
        parsed = self._parse_response(response)
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        return {
            'label': parsed['label'],
            'confidence': parsed['confidence'],
            'reasoning': parsed.get('reasoning', ''),
            'raw_output': response,
            'latency_ms': latency_ms
        }
    
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict from single image.
        """
        start_time = time.time()
        
        # Convert to PIL
        pil_image = self._array_to_pil(image)
        
        # Create message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse
        parsed = self._parse_response(response)
        
        # Metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        return {
            'label': parsed['label'],
            'confidence': parsed['confidence'],
            'reasoning': parsed.get('reasoning', ''),
            'raw_output': response,
            'latency_ms': latency_ms
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Batch prediction for multiple single images.
        """
        start_time = time.time()
        
        # Convert all to PIL
        pil_images = [self._array_to_pil(img) for img in images]
        
        # Create messages for each image
        all_messages = []
        for pil_img in pil_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
            all_messages.append(messages)
        
        # Process each (Qwen VL doesn't support true batching easily)
        # Fall back to sequential for now
        results = []
        for img in images:
            pred = self.predict_single(img)
            results.append(pred)
        
        return results
    
    def _video_to_pil_images(self, video: np.ndarray) -> List[Image.Image]:
        """Convert video tensor to list of PIL Images"""
        T, H, W, C = video.shape
        images = []
        
        for t in range(T):
            frame = video[t]
            pil_img = self._array_to_pil(frame)
            images.append(pil_img)
        
        return images
    
    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        # Assume array is in [0, 1] range, float32
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array * 255).astype(np.uint8)
        
        return Image.fromarray(array, mode='RGB')
