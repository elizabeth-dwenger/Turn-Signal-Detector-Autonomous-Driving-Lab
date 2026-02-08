"""
Qwen VL model implementation for turn signal detection.
Supports both Qwen3-VL and Qwen2.5-VL models.
Handles both video (multi-image) and single-image modes.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
from PIL import Image
import json
import re

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
        # Runtime-only args (not for from_pretrained)
        model_load_kwargs.pop('video_fps', None)
        model_load_kwargs.pop('target_video_fps', None)
        
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

    def _predict_video_core(self, video: np.ndarray) -> Tuple[Dict, str]:
        """
        Core video inference without metrics bookkeeping.
        Returns (parsed_dict, raw_response).
        """
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
        
        # Parse response (support segments if provided)
        parsed = self._parse_response_qwen(response)
        
        return parsed, response

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract a JSON object from text.
        """
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _parse_response_qwen(self, response: str) -> Dict:
        """
        Parse Qwen response.
        Supports both segment-based and single-label JSON outputs.
        """
        parsed_json = self._extract_json(response)
        if isinstance(parsed_json, dict) and "segments" in parsed_json:
            segments = []
            for seg in parsed_json.get("segments", []):
                if not isinstance(seg, dict):
                    continue
                label = str(seg.get("label", "none")).lower()
                if label == "both":
                    label = "hazard"
                if label not in {"left", "right", "both", "none", "hazard"}:
                    label = "none"
                try:
                    start = int(seg.get("start_frame", 0))
                except (ValueError, TypeError):
                    start = 0
                try:
                    end = int(seg.get("end_frame", start))
                except (ValueError, TypeError):
                    end = start
                segments.append({
                    "label": label,
                    "start_frame": start,
                    "end_frame": end
                })
            # Provide a fallback label for downstream use
            primary = segments[0] if segments else {"label": "none"}
            return {
                "segments": segments,
                "label": primary.get("label", "none"),
                "reasoning": parsed_json.get("reasoning", "")
            }
        
        # Fallback: use base JSON parser (label)
        return self._parse_response(response)
    
    def predict_video(self, video: np.ndarray = None, chunks: List[Tuple[np.ndarray, int, int]] = None) -> Dict:
        """
        Predict from video sequence (multi-image input) or list of chunks.
        """
        start_time = time.time()
        
        # Handle chunked inference
        if chunks is not None:
            return self._predict_video_chunked(chunks, start_time)
        
        if video is None:
            raise ValueError("Either video or chunks must be provided to predict_video()")
        
        parsed, response = self._predict_video_core(video)
        
        # Track metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        result = {
            'label': parsed['label'],
            'reasoning': parsed.get('reasoning', ''),
            'raw_output': response,
            'latency_ms': latency_ms
        }
        if 'segments' in parsed:
            result['segments'] = parsed['segments']
        return result
    
    def _predict_video_chunked(self, chunks: List[Tuple[np.ndarray, int, int]], 
                               start_time: float) -> Dict:
        """
        Process video in chunks and merge results.
        """
        all_segments = []
        total_frames = max(end_idx for _, _, end_idx in chunks) + 1
        fps = self.config.model_kwargs.get('video_fps', 10.0)
        
        logger.info(f"Processing {len(chunks)} chunks for sequence with {total_frames} frames")
        
        for chunk_idx, (chunk_video, start_idx, end_idx) in enumerate(chunks):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: frames {start_idx}–{end_idx}")
            
            # Process this chunk as a video
            try:
                parsed, response = self._predict_video_core(chunk_video)
                
                # If result has segments, offset them
                if 'segments' in parsed:
                    for seg in parsed['segments']:
                        seg['start_frame'] = seg.get('start_frame', 0) + start_idx
                        seg['end_frame'] = min(seg.get('end_frame', 0) + start_idx, total_frames - 1)
                        fps = self.config.model_kwargs.get('video_fps', 10.0)
                        seg['start_time_seconds'] = round(seg['start_frame'] / fps, 2)
                        seg['end_time_seconds'] = round(seg['end_frame'] / fps, 2)
                        all_segments.append(seg)
                else:
                    # Single label result - convert to segment
                    all_segments.append({
                        'label': parsed.get('label', 'none'),
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'start_time_seconds': round(start_idx / fps, 2),
                        'end_time_seconds': round(end_idx / fps, 2)
                    })
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                # Add a fallback segment for this chunk
                all_segments.append({
                    'label': 'none',
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'start_time_seconds': round(start_idx / fps, 2),
                    'end_time_seconds': round(end_idx / fps, 2)
                })
        
        # Merge overlapping segments
        merged_segments = self._merge_chunk_segments(all_segments, total_frames)
        
        latency_ms = (time.time() - start_time) * 1000
        self.metrics['total_inferences'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        
        return {
            'segments': merged_segments,
            'reasoning': f'Chunked inference across {len(chunks)} windows',
            'latency_ms': latency_ms,
            'label': merged_segments[0]['label'] if merged_segments else 'none',
            'raw_output': f'Processed {len(chunks)} chunks',
        }
    
    def _merge_chunk_segments(self, segments: List[Dict], total_frames: int) -> List[Dict]:
        """Merge overlapping segments from chunks."""
        if not segments:
            return [{
                'label': 'none',
                'start_frame': 0,
                'end_frame': total_frames - 1,
                'start_time_seconds': 0.0,
                'end_time_seconds': round((total_frames - 1) / self.config.model_kwargs.get('video_fps', 10.0), 2)
            }]
        
        # Sort by start frame
        segments = sorted(segments, key=lambda s: s['start_frame'])
        
        merged = [segments[0].copy()]
        for seg in segments[1:]:
            last = merged[-1]
            # Merge if same label and overlapping/adjacent
            if (seg['label'] == last['label'] and 
                seg['start_frame'] <= last['end_frame'] + 1):
                last['end_frame'] = max(last['end_frame'], seg['end_frame'])
                last['end_time_seconds'] = round(last['end_frame'] / self.config.model_kwargs.get('video_fps', 10.0), 2)
            else:
                merged.append(seg.copy())
        
        return merged
    
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
