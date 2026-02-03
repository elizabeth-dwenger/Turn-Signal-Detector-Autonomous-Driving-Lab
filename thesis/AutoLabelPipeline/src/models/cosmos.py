import torch
import numpy as np
import logging
import time
import re
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import TurnSignalDetector

logger = logging.getLogger(__name__)

class CosmosDetector(TurnSignalDetector):
    """
    NVIDIA Cosmos model implementation for turn signal detection.
    Optimized for Cosmos-1.0-Reason-7B and 8B.
    """
    
    def warmup(self):
        """Load Cosmos model and processor"""
        logger.info(f"Loading Cosmos model: {self.model_name}")
        
        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Load processor - extract trust_remote_code from model_kwargs if present
        processor_kwargs = {}
        if 'trust_remote_code' in self.config.model_kwargs:
            processor_kwargs['trust_remote_code'] = self.config.model_kwargs['trust_remote_code']
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs
        )
        
        # Prepare model loading kwargs
        model_load_kwargs = self.config.model_kwargs.copy()
        
        # Convert torch_dtype to dtype if needed
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
        
        # For CPU, use float32
        if self.device == "cpu" and 'dtype' in model_load_kwargs:
            if model_load_kwargs['dtype'] == torch.bfloat16:
                logger.warning("BFloat16 not well supported on CPU, using float32 instead")
                model_load_kwargs['dtype'] = torch.float32
        
        logger.info(f"Loading model with device={self.device}, dtype={model_load_kwargs.get('dtype', 'default')}")
        
        # Load model with all kwargs from config
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_load_kwargs
        )
        
        self.model.eval()
        
        logger.info(f" Cosmos model loaded on {self.device}")
        logger.info(f"  Model type: {self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'N/A'}")
        logger.info(f"  Dtype: {self.model.dtype}")

    def predict_video(self, video: np.ndarray) -> Dict:
        """Predict from video sequence (T, H, W, C)"""
        start_time = time.time()
        images = self._video_to_pil_images(video)
        
        # Construct the multimodal message
        # IMPORTANT: For Cosmos Reason models, keep prompt simple and direct
        # The model will use <think> tags internally
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": images, "fps": 4.0},
                {"type": "text", "text": self.prompt}
            ],
        }]
        
        return self._run_inference(messages, start_time)

    def predict_single(self, image: np.ndarray) -> Dict:
        """Predict from single image (H, W, C)"""
        start_time = time.time()
        pil_image = self._array_to_pil(image)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": self.prompt}
            ],
        }]
        
        return self._run_inference(messages, start_time)

    def _run_inference(self, messages: List[Dict], start_time: float) -> Dict:
        """Shared inference logic for both video and image"""
        # 1. Prepare inputs using chat template and processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 2. Generate response
        # Cosmos Reason models need more tokens for <think> + <answer>
        # Ensure enough tokens
        max_tokens = max(self.config.max_new_tokens, 1000)
        
        logger.debug(f"Generating with max_new_tokens={max_tokens}")
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                # Important: Set EOS token properly
                pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # 3. Decode
        generated_ids = [
            out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Log for debugging
        logger.debug(f"Raw response length: {len(response)} chars")
        if len(response) > 100:
            logger.debug(f"Response starts: {response[:100]}")
            logger.debug(f"Response ends: ...{response[-100:]}")
        
        # 4. Parse response
        parsed = self._parse_response_cosmos(response)
        
        # 5. Update Metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        return {
            'label': parsed.get('label', 'none'),
            'confidence': parsed.get('confidence', 0.0),
            'reasoning': parsed.get('reasoning', ''),
            'start_frame': parsed.get('start_frame'),
            'end_frame': parsed.get('end_frame'),
            'start_time_seconds': parsed.get('start_time_seconds'),
            'end_time_seconds': parsed.get('end_time_seconds'),
            'raw_output': response,
            'latency_ms': latency_ms
        }

    def _parse_response_cosmos(self, text: str) -> Dict:
        """
        Parse Cosmos Reason model output.
        Handles <think> and <answer> tags properly.
        """
        import json
        
        # Extract reasoning from <think> tags
        reasoning_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract answer from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # No answer tags - try to find JSON in the whole response
            # This handles cases where the model didn't use tags
            answer_text = text
        
        # Try to parse JSON from answer
        json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', answer_text, re.DOTALL)
        
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                
                label = parsed_json.get('label', 'none').lower()
                confidence = float(parsed_json.get('confidence', 0.5))
                
                # Extract temporal fields
                start_frame = parsed_json.get('start_frame')
                end_frame = parsed_json.get('end_frame')
                start_time = parsed_json.get('start_time_seconds')
                end_time = parsed_json.get('end_time_seconds')
                
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
                
                # Validate
                valid_labels = {'left', 'right', 'none', 'both'}
                if label not in valid_labels:
                    logger.warning(f"Invalid label '{label}', defaulting to 'none'")
                    label = 'none'
                    confidence = 0.3
                
                confidence = max(0.0, min(1.0, confidence))
                
                # Validate temporal consistency
                if start_frame is not None and end_frame is not None:
                    if end_frame < start_frame:
                        logger.warning(f"Invalid temporal range: end ({end_frame}) < start ({start_frame})")
                        start_frame = None
                        end_frame = None
                        start_time = None
                        end_time = None
                
                self.metrics['successful_parses'] += 1
                
                return {
                    'label': label,
                    'confidence': confidence,
                    'reasoning': reasoning[:500] if reasoning else parsed_json.get('reasoning', ''),  # Limit reasoning length
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_seconds': start_time,
                    'end_time_seconds': end_time,
                }
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # Fallback: extract label from text
        answer_lower = answer_text.lower()
        
        for label in ['left', 'right', 'both', 'none']:
            if label in answer_lower:
                logger.info(f"Extracted label '{label}' from text (no valid JSON)")
                self.metrics['failed_parses'] += 1
                return {
                    'label': label,
                    'confidence': 0.5,
                    'reasoning': reasoning[:200] if reasoning else '',
                    'start_frame': None,
                    'end_frame': None,
                    'start_time_seconds': None,
                    'end_time_seconds': None,
                }
        
        # Complete failure
        logger.error(f"Could not parse response (length: {len(text)})")
        logger.error(f"Response preview: {text[:200]}...")
        self.metrics['failed_parses'] += 1
        
        return {
            'label': 'none',
            'confidence': 0.0,
            'reasoning': 'Parse failed',
            'start_frame': None,
            'end_frame': None,
            'start_time_seconds': None,
            'end_time_seconds': None,
        }

    def _video_to_pil_images(self, video: np.ndarray) -> List[Image.Image]:
        return [self._array_to_pil(frame) for frame in video]

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')
