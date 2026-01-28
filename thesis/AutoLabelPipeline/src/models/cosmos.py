import torch
import numpy as np
import logging
import time
import re
from typing import List, Dict
from PIL import Image

from transformers import AutoModelForVision2Seq, AutoProcessor
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
        
        # Processor replaces both Tokenizer and manual Image processing
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            **self.config.model_kwargs
        )
        self.model.eval()
        
        logger.info(f"Cosmos model loaded on {self.device} in bfloat16")

    def predict_video(self, video: np.ndarray) -> Dict:
        """Predict from video sequence (T, H, W, C)"""
        start_time = time.time()
        images = self._video_to_pil_images(video)
        
        # Construct the multimodal message
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": images, "fps": 4.0},
                {"type": "text", "text": self._get_reasoning_prompt()}
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
                {"type": "text", "text": self._get_reasoning_prompt()}
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
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.get('top_p', 0.9)
            )
        
        # 3. Decode and Parse
        generated_ids = [
            out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        parsed = self._parse_response(response)
        
        # 4. Update Metrics (from your original code)
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        return {
            'label': parsed.get('label'),
            'confidence': parsed.get('confidence'),
            'reasoning': parsed.get('reasoning', ''),
            'raw_output': response,
            'latency_ms': latency_ms
        }

    def _get_reasoning_prompt(self) -> str:
        """Encourages the model to use its internal COT reasoning tags"""
        return f"{self.prompt}\nStructure your thoughts within <think> tags and provide the final answer within <answer> tags."

    def _parse_response(self, text: str) -> Dict:
        """Extracts reasoning and answer from the model's output"""
        reasoning = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        
        # Fallback if model doesn't use tags correctly
        final_answer = answer.group(1).strip() if answer else text.strip()
        
        return {
            "reasoning": reasoning.group(1).strip() if reasoning else "",
            "label": final_answer, # Logic to map string to label goes here
            "confidence": 1.0       # Cosmos-Reason doesn't give native logprobs easily
        }

    def _video_to_pil_images(self, video: np.ndarray) -> List[Image.Image]:
        return [self._array_to_pil(frame) for frame in video]

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')
