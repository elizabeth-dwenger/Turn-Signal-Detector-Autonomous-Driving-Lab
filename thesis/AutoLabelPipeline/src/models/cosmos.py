import torch
import numpy as np
import logging
import time
import re
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor
from .qwen_vl_utils import process_vision_info

from .base import TurnSignalDetector

logger = logging.getLogger(__name__)

class CosmosDetector(TurnSignalDetector):
    """
    NVIDIA Cosmos model implementation for turn signal detection.
    Optimized for Cosmos-1.0-Reason-7B and 8B.

    IMPORTANT: Cosmos Reason models use igid and <answer> tags.
    This class handles that format properly.

    Supports both single-segment (legacy) and multi-segment output formats.
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
        # Runtime-only args (not for from_pretrained)
        model_load_kwargs.pop('video_fps', None)

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

    def predict_video(self, video: np.ndarray = None, chunks: List[Tuple[np.ndarray, int, int]] = None) -> Dict:
        """
        Predict from video sequence (T, H, W, C) or list of chunks.
        
        Args:
            video: Full video tensor (T, H, W, C) for standard inference
            chunks: List of (chunk_video, start_idx, end_idx) tuples for chunked inference
        """
        start_time = time.time()
        
        # Handle chunked inference
        if chunks is not None:
            return self._predict_video_chunked(chunks, start_time)
        
        if video is None:
            raise ValueError("Either video or chunks must be provided to predict_video()")
        
        # Original single-video inference
        images = self._video_to_pil_images(video)
        num_frames = len(images)
        
        fps = self.config.model_kwargs.get('video_fps', 10.0)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": images, "fps": fps},
                {"type": "text", "text": self.prompt}
            ],
        }]
        
        return self._run_inference(messages, start_time, num_frames=num_frames)
    
    def _predict_video_chunked(self, chunks: List[Tuple[np.ndarray, int, int]], 
                               start_time: float) -> Dict:
        """
        Process video in chunks and merge results.
        """
        all_segments = []
        total_frames = max(end_idx for _, _, end_idx in chunks) + 1
        
        logger.info(f"Processing {len(chunks)} chunks for sequence with {total_frames} frames")
        
        for chunk_idx, (chunk_video, start_idx, end_idx) in enumerate(chunks):
            logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}: frames {start_idx}–{end_idx}")
            
            images = self._video_to_pil_images(chunk_video)
            chunk_num_frames = end_idx - start_idx + 1
            
            fps = self.config.model_kwargs.get('video_fps', 10.0)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": images, "fps": fps},
                    {"type": "text", "text": self.prompt}
                ],
            }]
            
            try:
                chunk_result = self._run_inference(messages, time.time(), num_frames=chunk_num_frames, update_metrics=False)
                
                # Offset segments to global frame indices
                for seg in chunk_result.get('segments', []):
                    seg['start_frame'] = seg.get('start_frame', 0) + start_idx
                    seg['end_frame'] = min(seg.get('end_frame', 0) + start_idx, total_frames - 1)
                    fps = self.config.model_kwargs.get('video_fps', 10.0)
                    seg['start_time_seconds'] = round(seg['start_frame'] / fps, 2)
                    seg['end_time_seconds'] = round(seg['end_frame'] / fps, 2)
                    all_segments.append(seg)
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                # Add a fallback segment for this chunk
                all_segments.append({
                    'label': 'none',
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'confidence': 0.0,
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
            'confidence': max((s['confidence'] for s in merged_segments), default=0.0),
            'raw_output': f'Processed {len(chunks)} chunks',
        }
    
    def _merge_chunk_segments(self, segments: List[Dict], total_frames: int) -> List[Dict]:
        """Merge overlapping segments from chunks."""
        if not segments:
            return [{
                'label': 'none',
                'start_frame': 0,
                'end_frame': total_frames - 1,
                'confidence': 0.5,
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
                last['confidence'] = max(last['confidence'], seg['confidence'])
            else:
                merged.append(seg.copy())
        
        return merged

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

        return self._run_inference(messages, start_time, num_frames=1)

    def _run_inference(self, messages: List[Dict], start_time: float, num_frames: int = 1, update_metrics: bool = True) -> Dict:
        """Shared inference logic for both video and image"""
        # 1. Prepare inputs using chat template and processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        if not image_inputs:
            image_inputs = None
        if not video_inputs:
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 2. Generate response
        # Multi-segment output needs more tokens than single-segment
        max_tokens = max(self.config.max_new_tokens, 1500)

        logger.debug(f"Generating with max_new_tokens={max_tokens}")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
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

        # 4. Parse response — tries multi-segment first, falls back to single-segment
        parsed = self._parse_response_cosmos(response, num_frames=num_frames)

        # 5. Update Metrics
        latency_ms = (time.time() - start_time) * 1000
        if update_metrics:
            self.metrics['total_inferences'] += 1
            self.metrics['total_latency_ms'] += latency_ms

        # 6. Build return value — always include segments array
        result = {
            'segments': parsed.get('segments', []),
            'reasoning': parsed.get('reasoning', ''),
            'raw_output': response,
            'latency_ms': latency_ms,
        }

        # Also populate legacy top-level fields from the first non-none segment
        # (or the first segment if everything is none) so downstream code that
        # only looks at 'label' still works.
        primary = self._get_primary_segment(parsed.get('segments', []))
        result['label'] = primary.get('label', 'none')
        result['confidence'] = primary.get('confidence', 0.0)
        result['start_frame'] = primary.get('start_frame')
        result['end_frame'] = primary.get('end_frame')
        result['start_time_seconds'] = primary.get('start_time_seconds')
        result['end_time_seconds'] = primary.get('end_time_seconds')

        return result

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def _parse_response_cosmos(self, text: str, num_frames: int = 1) -> Dict:
        """
        Parse Cosmos Reason model output.
        Handles igid and <answer> tags.
        Supports both multi-segment (new) and single-segment (legacy) formats.
        """
        # Extract reasoning from igid tags
        reasoning_match = re.search(r"igid(.*?)igid", text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract answer from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else text

        # --- Try to extract a JSON object (possibly nested with "segments" array) ---
        parsed_json = self._extract_json(answer_text)

        if parsed_json is None:
            # Total parse failure — return safe default
            logger.error(f"Could not parse any JSON from response (length: {len(text)})")
            self.metrics['failed_parses'] += 1
            return self._make_none_segments(num_frames, reasoning="Parse failed")

        # --- Determine format and normalise to segments ---
        if 'segments' in parsed_json and isinstance(parsed_json['segments'], list):
            #  New multi-segment format
            segments = self._validate_segments(parsed_json['segments'], num_frames)
            if not reasoning:
                reasoning = parsed_json.get('reasoning', '')
        elif 'label' in parsed_json:
            # Legacy single-segment format — wrap into segments array
            segments = self._single_to_segments(parsed_json, num_frames)
            if not reasoning:
                reasoning = parsed_json.get('reasoning', '')
        else:
            logger.warning("JSON found but has neither 'segments' nor 'label' key")
            segments = self._make_none_segments(num_frames)['segments']

        self.metrics['successful_parses'] += 1

        return {
            'segments': segments,
            'reasoning': reasoning[:500] if reasoning else '',
        }

    # -------------------------------------------------------------------------
    # JSON extraction helpers
    # -------------------------------------------------------------------------

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Pull the first valid JSON object out of text.
        Handles nested braces (e.g. objects containing arrays of objects).
        """
        # Find the first '{' and then match braces
        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Brace-matched but not valid JSON; keep scanning
                        break
        # Fallback: try rfind for closing brace
        end = text.rfind('}')
        if end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        return None

    # -------------------------------------------------------------------------
    # Segment validation & conversion
    # -------------------------------------------------------------------------

    def _validate_segments(self, raw_segments: list, num_frames: int) -> List[Dict]:
        """
        Validate and sanitise a segments array from the model.
        - Clamps frame numbers to [0, num_frames-1].
        - Ensures labels are valid.
        - Fills gaps / fixes overlaps by trusting the ordering and
          simply reassigning boundaries sequentially.
        - If validation fails badly, returns a safe single-segment fallback.
        """
        if not raw_segments:
            return self._make_none_segments(num_frames)['segments']

        valid_labels = {'left', 'right', 'none', 'both'}
        last_frame = num_frames - 1
        fps = self.config.model_kwargs.get('video_fps', 10.0)
        cleaned = []

        for seg in raw_segments:
            label = str(seg.get('label', 'none')).lower().strip()
            if label not in valid_labels:
                label = 'none'

            confidence = float(seg.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))

            start = seg.get('start_frame')
            end = seg.get('end_frame')

            # Coerce to int; if missing use None (will be fixed below)
            try:
                start = int(start)
            except (TypeError, ValueError):
                start = None
            try:
                end = int(end)
            except (TypeError, ValueError):
                end = None

            cleaned.append({
                'label': label,
                'start_frame': start,
                'end_frame': end,
                'confidence': confidence,
            })

        # --- Sequential boundary repair ---
        # Walk through and assign contiguous ranges based on the order the
        # model produced.  If start/end are missing or inconsistent we
        # distribute frames as evenly as possible.
        repaired = []
        cursor = 0  # next frame to assign

        for i, seg in enumerate(cleaned):
            if cursor > last_frame:
                break  # video is fully covered

            s = seg['start_frame'] if seg['start_frame'] is not None else cursor
            # Clamp start to cursor (no going backward / overlapping)
            s = max(s, cursor)
            s = min(s, last_frame)

            if i == len(cleaned) - 1:
                # Last segment always extends to the end
                e = last_frame
            else:
                e = seg['end_frame'] if seg['end_frame'] is not None else s
                e = max(e, s)          # end >= start
                e = min(e, last_frame) # don't exceed video length

            fps = self.config.model_kwargs.get('video_fps', 10.0)
            repaired.append({
                'label': seg['label'],
                'start_frame': s,
                'end_frame': e,
                'confidence': seg['confidence'],
                'start_time_seconds': round(s / fps, 2),
                'end_time_seconds': round(e / fps, 2),
            })
            cursor = e + 1

        # If we didn't reach the end, extend the last segment
        if repaired and repaired[-1]['end_frame'] < last_frame:
            repaired[-1]['end_frame'] = last_frame
            repaired[-1]['end_time_seconds'] = round(last_frame / 10.0, 2)

        # If nothing survived, return safe default
        if not repaired:
            return self._make_none_segments(num_frames)['segments']

        return repaired

    def _single_to_segments(self, parsed_json: Dict, num_frames: int) -> List[Dict]:
        """
        Convert a legacy single-label prediction into a segments array.
        If temporal info is present, creates up to 3 segments:
            none (before signal) → signal → none (after signal).
        If no temporal info, the whole video gets the single label.
        """
        label = str(parsed_json.get('label', 'none')).lower().strip()
        valid_labels = {'left', 'right', 'none', 'both'}
        if label not in valid_labels:
            label = 'none'

        confidence = float(parsed_json.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))

        last_frame = num_frames - 1

        start_frame = parsed_json.get('start_frame')
        end_frame = parsed_json.get('end_frame')

        # Coerce
        try:
            start_frame = int(start_frame)
        except (TypeError, ValueError):
            start_frame = None
        try:
            end_frame = int(end_frame)
        except (TypeError, ValueError):
            end_frame = None

        # If no temporal info OR label is none, single segment covers everything
        if label == 'none' or start_frame is None or end_frame is None:
            return [{
                'label': label,
                'start_frame': 0,
                'end_frame': last_frame,
                'confidence': confidence,
                'start_time_seconds': 0.0,
                'end_time_seconds': round(last_frame / fps, 2),
            }]

        # Clamp
        start_frame = max(0, min(start_frame, last_frame))
        end_frame = max(start_frame, min(end_frame, last_frame))

        segments = []

        # Leading "none" segment (if signal doesn't start at frame 0)
        if start_frame > 0:
            segments.append({
                'label': 'none',
                'start_frame': 0,
                'end_frame': start_frame - 1,
                'confidence': 0.85,
                'start_time_seconds': 0.0,
                'end_time_seconds': round((start_frame - 1) / fps, 2),
            })

        # The signal segment itself
        segments.append({
            'label': label,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'confidence': confidence,
            'start_time_seconds': round(start_frame / fps, 2),
            'end_time_seconds': round(end_frame / fps, 2),
        })

        # Trailing "none" segment (if signal ends before last frame)
        if end_frame < last_frame:
            segments.append({
                'label': 'none',
                'start_frame': end_frame + 1,
                'end_frame': last_frame,
                'confidence': 0.85,
                'start_time_seconds': round((end_frame + 1) / fps, 2),
                'end_time_seconds': round(last_frame / fps, 2),
            })

        return segments

    def _make_none_segments(self, num_frames: int, reasoning: str = "") -> Dict:
        """Safe fallback: entire video is 'none'."""
        last_frame = max(num_frames - 1, 0)
        fps = self.config.model_kwargs.get('video_fps', 10.0)
        return {
            'segments': [{
                'label': 'none',
                'start_frame': 0,
                'end_frame': last_frame,
                'confidence': 0.0,
                'start_time_seconds': 0.0,
                'end_time_seconds': round(last_frame / fps, 2),
            }],
            'reasoning': reasoning,
        }

    # -------------------------------------------------------------------------
    # Legacy field helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_primary_segment(segments: List[Dict]) -> Dict:
        """
        Pick the "primary" segment for legacy top-level fields.
        Returns the first segment whose label is not 'none'.
        If all are 'none', returns the first segment.
        """
        for seg in segments:
            if seg.get('label', 'none') != 'none':
                return seg
        return segments[0] if segments else {'label': 'none', 'confidence': 0.0}

    # -------------------------------------------------------------------------
    # Image helpers (unchanged)
    # -------------------------------------------------------------------------

    def _video_to_pil_images(self, video: np.ndarray) -> List[Image.Image]:
        return [self._array_to_pil(frame) for frame in video]

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')
