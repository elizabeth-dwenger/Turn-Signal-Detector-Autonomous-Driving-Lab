"""
Visualization generation for predictions.
Creates annotated images and videos showing predictions.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging


logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Create visualizations of predictions"""
    
    # Minimal color scheme (RGB format)
    COLORS = {
        'label_text': (240, 240, 240),  # Light gray (nearly white) - for frame labels
        'title_text': (0, 0, 0),        # Black - for headers/titles
        'match': (0, 255, 0),           # Green - for correct predictions
        'mismatch': (255, 0, 0),        # Red - for incorrect predictions
        'muted': (120, 120, 120),       # Gray
    }
    
    def __init__(self, output_config):
        """
        Args:
            output_config: OutputConfig from configuration
        """
        self.config = output_config
        self.output_dir = Path(output_config.visualization_output_dir or 'visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_label(self, label: str) -> str:
        if label is None:
            return "none"
        label = str(label).strip().lower()
        return label if label in {"left", "right", "both", "none"} else "none"

    def _compute_frame_metrics(self, y_true, y_pred, labels=None):
        if labels is None:
            labels = ["left", "right", "both", "none"]
        if not y_true:
            return {"accuracy": None, "macro_f1": None}
        total = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / total if total else 0.0
        f1s = []
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return {"accuracy": accuracy, "macro_f1": macro_f1}
    
    def visualize_frame(self, image: np.ndarray, prediction: Dict,
                       ground_truth: str = None) -> np.ndarray:
        """
        Annotate a single frame with prediction.
        Returns RGB image.
        """
        # Ensure image is uint8 RGB
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Work on a copy in RGB
        if len(image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()
        elif image.shape[2] == 4:  # RGBA
            vis_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB).copy()
        else:  # Already RGB
            vis_image = image.copy()
        
        h, w = vis_image.shape[:2]
        
        # Get prediction info
        label = prediction['label']
        confidence = prediction['confidence']
        
        font = cv2.FONT_HERSHEY_COMPLEX
        
        # Convert to BGR for OpenCV text drawing, then back to RGB
        bgr_temp = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Draw prediction text (top-left)
        label_text = f"Pred: {label.upper()}"
        cv2.putText(bgr_temp, label_text, (10, 25),
                   font, 0.7, self.COLORS['label_text'][::-1], 2)  # Reverse RGB to BGR
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(bgr_temp, conf_text, (10, 45),
                   font, 0.6, self.COLORS['label_text'][::-1], 2)
        
        # Draw ground truth below prediction (no overlap)
        if ground_truth:
            gt_text = f"GT: {ground_truth.upper()}"
            color_key = 'match' if label == ground_truth else 'mismatch'
            cv2.putText(bgr_temp, gt_text, (10, 65),
                       font, 0.6, self.COLORS[color_key][::-1], 2)
        
        # Convert back to RGB for return
        return cv2.cvtColor(bgr_temp, cv2.COLOR_BGR2RGB)
    
    def _draw_signal_indicator(self, image: np.ndarray, label: str, color: tuple):
        """Deprecated: kept for compatibility."""
        return
    
    def visualize_sequence(self, images: List[np.ndarray],
                          predictions: List[Dict],
                          output_path: str,
                          ground_truth: List[str] = None,
                          fps: int = 10):
        """
        Create video visualization of sequence.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not images:
            logger.warning("No images to visualize")
            return
        
        # Get dimensions from first image
        h, w = images[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Annotate and write frames
        for i, (image, pred) in enumerate(zip(images, predictions)):
            gt = ground_truth[i] if ground_truth else None
            annotated = self.visualize_frame(image, pred, gt)
            
            # Convert RGB to BGR for video writer
            bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        
        writer.release()
        logger.info(f"Saved visualization video: {output_path}")

    def _resize_with_padding(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize image to fit target size with padding. Works with RGB images."""
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((target_h, target_w, 3), 255, dtype=resized.dtype)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    def create_contact_sheet(self, images: List[np.ndarray],
                             frame_ids: List[int],
                             predictions: List[Dict],
                             output_path: Path,
                             sequence_name: str,
                             ground_truth: List[str] = None):
        """
        Create a single contact sheet containing all frames.
        If a frame has no prediction, it will be left blank.
        """
        if not images:
            logger.warning("No images to visualize")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_frames = len(images)
        if not frame_ids or len(frame_ids) != num_frames:
            frame_ids = list(range(num_frames))
        
        # Map predictions by frame_id for alignment
        pred_by_id = {p.get('frame_id'): p for p in predictions if p.get('frame_id') is not None}
        
        # Compute per-sequence metrics if GT available
        metrics_text = ""
        if ground_truth and len(ground_truth) == num_frames:
            y_true = []
            y_pred = []
            for fid, gt in zip(frame_ids, ground_truth):
                if fid in pred_by_id:
                    y_true.append(self._normalize_label(gt))
                    y_pred.append(self._normalize_label(pred_by_id[fid].get('label', 'none')))
            metrics = self._compute_frame_metrics(y_true, y_pred)
            if metrics["accuracy"] is not None:
                metrics_text = f"acc={metrics['accuracy']:.3f} | macroF1={metrics['macro_f1']:.3f}"
        
        # Layout
        cols = 6
        rows = int(np.ceil(num_frames / cols))
        tile_w = 224
        tile_h = 224
        header_h = 70
        sheet_w = cols * tile_w
        sheet_h = rows * tile_h + header_h
        
        # Build contact sheet in RGB
        sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
        
        # Convert to BGR temporarily for text drawing
        sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
        
        header_text = f"{sequence_name} | frames={num_frames}"
        metrics_line = metrics_text.strip()
        cv2.putText(sheet_bgr, header_text, (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, self.COLORS['title_text'][::-1], 2)
        if metrics_line:
            cv2.putText(sheet_bgr, metrics_line, (10, 55),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, self.COLORS['title_text'][::-1], 2)
        
        # Convert back to RGB
        sheet = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2RGB)
        
        for idx, (image, fid) in enumerate(zip(images, frame_ids)):
            r = idx // cols
            c = idx % cols
            y0 = header_h + r * tile_h
            x0 = c * tile_w
            
            if fid in pred_by_id:
                pred = pred_by_id[fid]
                gt = ground_truth[idx] if ground_truth and idx < len(ground_truth) else None
                annotated = self.visualize_frame(image, pred, gt)  # Returns RGB
                tile = self._resize_with_padding(annotated, tile_w, tile_h)
            else:
                # No prediction: show image with a subtle note
                # Ensure image is RGB
                if len(image.shape) == 2:
                    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).copy()
                elif image.shape[2] == 4:
                    annotated = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB).copy()
                else:
                    annotated = image.copy()
                
                font = cv2.FONT_HERSHEY_COMPLEX
                # Convert to BGR for text, then back to RGB
                tmp = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.putText(tmp, "No pred", (10, 25),
                            font, 0.6, self.COLORS['muted'][::-1], 1)
                annotated = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                tile = self._resize_with_padding(annotated, tile_w, tile_h)
            
            sheet[y0:y0+tile_h, x0:x0+tile_w] = tile
        
        # Write with OpenCV (expects BGR)
        cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    
    def create_sample_visualizations(self, sequences_data: Dict,
                                    sample_rate: float = 0.01):
        """
        Create visualizations for sample of sequences.
        """
        num_to_sample = max(1, int(len(sequences_data) * sample_rate))
        
        # Sample sequences
        import random
        sampled = random.sample(list(sequences_data.items()), num_to_sample)
        
        for sequence_id, data in sampled:
            try:
                # Create safe filename
                safe_id = sequence_id.replace('/', '_').replace('\\', '_')
                if 'images' in data and data['images']:
                    frame_ids = data.get('frame_ids', [])
                    out_path = self.output_dir / f"{safe_id}_contact_sheet.png"
                    self.create_contact_sheet(
                        data['images'],
                        frame_ids,
                        data['predictions'],
                        out_path,
                        safe_id,
                        data.get('ground_truth')
                    )
            
            except Exception as e:
                logger.error(f"Error visualizing {sequence_id}: {e}")
        
        logger.info(f"Created visualizations for {len(sampled)} sequences in {self.output_dir}")


def create_visualizer(output_config):
    """
    Factory function to create visualizer.
    """
    return PredictionVisualizer(output_config)
