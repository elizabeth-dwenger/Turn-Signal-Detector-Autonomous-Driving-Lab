"""
Visualization generation for turn signal predictions.
Creates annotated images and videos for spot-checking results.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class FrameVisualizer:
    """
    Creates visual annotations on frames showing predictions.
    """
    
    def __init__(self, visualization_config=None):
        """
        Args:
            visualization_config: Configuration for visualization settings
        """
        self.config = visualization_config or {}
        
        # Colors for labels (BGR format for OpenCV)
        self.label_colors = {
            'none': (200, 200, 200),  # Gray
            'left': (0, 165, 255),     # Orange
            'right': (0, 255, 255),    # Yellow
            'both': (0, 0, 255),       # Red
        }
        
        # Fonts and sizes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
    def annotate_frame(self,
                      image: np.ndarray,
                      prediction: Dict,
                      sequence_id: str = "",
                      frame_id: int = 0,
                      show_timeline: bool = True,
                      timeline_predictions: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Annotate a single frame with prediction information.
        
        Args:
            image: Input image (H, W, 3) BGR format
            prediction: Prediction dict with label, confidence, etc.
            sequence_id: Sequence identifier
            frame_id: Frame number
            show_timeline: Whether to show temporal context
            timeline_predictions: List of predictions for timeline visualization
        
        Returns:
            Annotated image
        """
        # Create copy to avoid modifying original
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Extract prediction info
        label = prediction['label']
        confidence = prediction.get('confidence', 0.0)
        
        # Get label color
        color = self.label_colors.get(label, (255, 255, 255))
        
        # 1. Draw header with sequence and frame info
        header_text = f"{sequence_id[:40]} | Frame {frame_id}"
        cv2.putText(annotated, header_text, (10, 30),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # 2. Draw prediction box (top-left corner)
        box_height = 120
        box_width = 250
        
        # Semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 50), (10 + box_width, 50 + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw border
        cv2.rectangle(annotated, (10, 50), (10 + box_width, 50 + box_height),
                     color, 2)
        
        # Label text
        label_text = f"Label: {label.upper()}"
        cv2.putText(annotated, label_text, (20, 80),
                   self.font, self.font_scale, color, self.thickness)
        
        # Confidence text
        conf_text = f"Conf: {confidence:.2%}"
        conf_color = (0, 255, 0) if confidence >= 0.7 else (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
        cv2.putText(annotated, conf_text, (20, 110),
                   self.font, self.font_scale, conf_color, self.thickness)
        
        # Status indicator
        status = "✓ High conf" if confidence >= 0.7 else "⚠ Low conf" if confidence >= 0.5 else "✗ Very low"
        status_color = (0, 255, 0) if confidence >= 0.7 else (0, 165, 255) if confidence >= 0.5 else (0, 0, 255)
        cv2.putText(annotated, status, (20, 140),
                   self.font, 0.5, status_color, 1)
        
        # 3. Show processing flags if any
        flags = []
        if prediction.get('smoothed'):
            flags.append('SMOOTHED')
        if prediction.get('reconstructed'):
            flags.append('RECONSTRUCTED')
        if prediction.get('constraint_enforced'):
            flags.append('CONSTRAINT')
        if prediction.get('flagged'):
            flags.append('FLAGGED')
        
        if flags:
            flag_text = " | ".join(flags)
            cv2.putText(annotated, flag_text, (20, 160),
                       self.font, 0.4, (255, 255, 0), 1)
        
        # 4. Draw timeline if requested
        if show_timeline and timeline_predictions:
            self._draw_timeline(annotated, timeline_predictions, frame_id, w, h)
        
        return annotated
    
    def _draw_timeline(self,
                      image: np.ndarray,
                      predictions: List[Dict],
                      current_frame_id: int,
                      width: int,
                      height: int):
        """
        Draw temporal context timeline at bottom of frame.
        
        Args:
            image: Image to draw on (modified in-place)
            predictions: List of all predictions in sequence
            current_frame_id: Current frame ID
            width: Image width
            height: Image height
        """
        # Timeline parameters
        timeline_height = 60
        timeline_y = height - timeline_height - 10
        timeline_x_start = 10
        timeline_width = width - 20
        
        # Background
        cv2.rectangle(image,
                     (timeline_x_start, timeline_y),
                     (timeline_x_start + timeline_width, timeline_y + timeline_height),
                     (0, 0, 0), -1)
        cv2.rectangle(image,
                     (timeline_x_start, timeline_y),
                     (timeline_x_start + timeline_width, timeline_y + timeline_height),
                     (255, 255, 255), 1)
        
        # Title
        cv2.putText(image, "Timeline:", (timeline_x_start + 5, timeline_y + 20),
                   self.font, 0.5, (255, 255, 255), 1)
        
        # Draw timeline segments
        if len(predictions) > 0:
            segment_width = max(2, timeline_width // len(predictions))
            
            for i, pred in enumerate(predictions):
                x = timeline_x_start + int(i * timeline_width / len(predictions))
                label = pred['label']
                color = self.label_colors.get(label, (255, 255, 255))
                
                # Draw segment
                cv2.rectangle(image,
                            (x, timeline_y + 25),
                            (x + segment_width, timeline_y + 50),
                            color, -1)
                
                # Highlight current frame
                if pred.get('frame_id') == current_frame_id:
                    cv2.rectangle(image,
                                (x, timeline_y + 25),
                                (x + segment_width, timeline_y + 50),
                                (255, 255, 255), 2)
                    # Draw pointer
                    cv2.arrowedLine(image,
                                  (x + segment_width // 2, timeline_y + 55),
                                  (x + segment_width // 2, timeline_y + 23),
                                  (255, 255, 255), 2)
    
    def create_comparison_grid(self,
                               images: List[np.ndarray],
                               predictions: List[Dict],
                               grid_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Create a grid of annotated frames for comparison.
        
        Args:
            images: List of images
            predictions: List of predictions (same length as images)
            grid_size: (rows, cols) or None for auto
        
        Returns:
            Grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        n_images = len(images)
        
        # Auto-determine grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            grid_size = (rows, cols)
        
        rows, cols = grid_size
        
        # Get image dimensions (assuming all same size)
        h, w = images[0].shape[:2]
        
        # Create grid
        grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        for idx, (img, pred) in enumerate(zip(images, predictions)):
            if idx >= rows * cols:
                break
            
            row = idx // cols
            col = idx % cols
            
            # Annotate image
            annotated = self.annotate_frame(img, pred, show_timeline=False)
            
            # Place in grid
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = annotated
        
        return grid


class VideoVisualizer:
    """
    Creates annotated videos from frame sequences.
    """
    
    def __init__(self, fps: int = 10):
        """
        Args:
            fps: Output video frame rate
        """
        self.fps = fps
        self.frame_visualizer = FrameVisualizer()
    
    def create_annotated_video(self,
                              images: List[np.ndarray],
                              predictions: List[Dict],
                              output_path: str,
                              sequence_id: str = "",
                              show_timeline: bool = True,
                              codec: str = 'mp4v') -> str:
        """
        Create annotated video from image sequence.
        
        Args:
            images: List of frame images
            predictions: List of predictions (same length as images)
            output_path: Output video file path
            sequence_id: Sequence identifier
            show_timeline: Show temporal context timeline
            codec: Video codec (e.g., 'mp4v', 'avc1', 'XVID')
        
        Returns:
            Path to created video file
        """
        if not images:
            raise ValueError("No images provided")
        
        if len(images) != len(predictions):
            raise ValueError(f"Image count ({len(images)}) != prediction count ({len(predictions)})")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video dimensions
        h, w = images[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
        
        logger.info(f"Creating annotated video: {output_path}")
        
        # Process each frame
        for i, (img, pred) in enumerate(tqdm(zip(images, predictions),
                                             total=len(images),
                                             desc="Creating video")):
            # Annotate frame
            annotated = self.frame_visualizer.annotate_frame(
                img, pred,
                sequence_id=sequence_id,
                frame_id=pred.get('frame_id', i),
                show_timeline=show_timeline,
                timeline_predictions=predictions if show_timeline else None
            )
            
            # Write frame
            out.write(annotated)
        
        out.release()
        logger.info(f"Saved annotated video to {output_path}")
        
        return str(output_path)


def visualize_samples(predictions_by_sequence: Dict[str, List[Dict]],
                     images_by_sequence: Dict[str, List[np.ndarray]],
                     output_dir: str,
                     sample_rate: float = 0.01,
                     format: str = 'images',
                     **kwargs) -> List[str]:
    """
    Create visualizations for a sample of sequences.
    
    Args:
        predictions_by_sequence: Dict mapping sequence_id to predictions
        images_by_sequence: Dict mapping sequence_id to frame images
        output_dir: Output directory for visualizations
        sample_rate: Fraction of frames to visualize
        format: 'images' or 'video'
        **kwargs: Additional arguments for visualizers
    
    Returns:
        List of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    frame_viz = FrameVisualizer()
    video_viz = VideoVisualizer(fps=kwargs.get('fps', 10))
    
    for sequence_id, predictions in predictions_by_sequence.items():
        images = images_by_sequence.get(sequence_id)
        
        if images is None or len(images) != len(predictions):
            logger.warning(f"Skipping {sequence_id}: images not available or length mismatch")
            continue
        
        if format == 'video':
            # Create full video for sequence
            video_path = output_dir / f"{sequence_id.replace('/', '_')}_annotated.mp4"
            output_file = video_viz.create_annotated_video(
                images, predictions, str(video_path), sequence_id
            )
            output_files.append(output_file)
        
        else:  # images
            # Sample frames
            n_frames = len(predictions)
            n_samples = max(1, int(n_frames * sample_rate))
            sample_indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)
            
            for idx in sample_indices:
                img = images[idx]
                pred = predictions[idx]
                
                # Annotate frame
                annotated = frame_viz.annotate_frame(
                    img, pred,
                    sequence_id=sequence_id,
                    frame_id=pred.get('frame_id', idx),
                    show_timeline=True,
                    timeline_predictions=predictions
                )
                
                # Save frame
                frame_path = output_dir / f"{sequence_id.replace('/', '_')}_frame_{idx:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated)
                output_files.append(str(frame_path))
    
    logger.info(f"Created {len(output_files)} visualization files in {output_dir}")
    return output_files

