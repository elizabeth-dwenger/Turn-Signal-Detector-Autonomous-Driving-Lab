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
    
    # Color scheme
    COLORS = {
        'none': (200, 200, 200),   # Gray
        'left': (0, 255, 255),      # Yellow
        'right': (255, 165, 0),     # Orange
        'both': (0, 0, 255),        # Red
        'flagged': (255, 0, 255)    # Magenta
    }
    
    def __init__(self, output_config):
        """
        Args:
            output_config: OutputConfig from configuration
        """
        self.config = output_config
        self.output_dir = Path(output_config.visualization_output_dir or 'visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_frame(self, image: np.ndarray, prediction: Dict,
                       ground_truth: str = None) -> np.ndarray:
        """
        Annotate a single frame with prediction.
        """
        # Convert to BGR for OpenCV
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        h, w = vis_image.shape[:2]
        
        # Get prediction info
        label = prediction['label']
        confidence = prediction['confidence']
        flagged = prediction.get('flagged', False)
        
        # Choose color
        color = self.COLORS.get(label, (255, 255, 255))
        if flagged:
            color = self.COLORS['flagged']
        
        # Draw label box
        box_height = 80
        cv2.rectangle(vis_image, (0, 0), (w, box_height), (0, 0, 0), -1)
        
        # Draw label text
        label_text = f"Pred: {label.upper()}"
        cv2.putText(vis_image, label_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw confidence
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(vis_image, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw ground truth if available
        if ground_truth:
            gt_text = f"GT: {ground_truth.upper()}"
            match = "✓" if label == ground_truth else "✗"
            gt_color = (0, 255, 0) if label == ground_truth else (0, 0, 255)
            
            cv2.putText(vis_image, f"{gt_text} {match}", (w - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2)
        
        # Draw flags if any
        if flagged:
            flags_text = f"FLAGGED"
            cv2.putText(vis_image, flags_text, (w - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLORS['flagged'], 2)
        
        # Draw indicator arrow/box for signal
        if label in ['left', 'right', 'both']:
            self._draw_signal_indicator(vis_image, label, color)
        
        return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    def _draw_signal_indicator(self, image: np.ndarray, label: str, color: tuple):
        """Draw arrow indicating signal direction"""
        h, w = image.shape[:2]
        arrow_y = h - 50
        
        if label == 'left' or label == 'both':
            # Left arrow
            cv2.arrowedLine(image, (w // 4, arrow_y), (50, arrow_y),
                          color, thickness=5, tipLength=0.3)
        
        if label == 'right' or label == 'both':
            # Right arrow
            cv2.arrowedLine(image, (3 * w // 4, arrow_y), (w - 50, arrow_y),
                          color, thickness=5, tipLength=0.3)
    
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
            
            # Convert back to BGR for video writer
            bgr_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        
        writer.release()
        logger.info(f"Saved visualization video: {output_path}")
    
    def visualize_timeline(self, predictions: List[Dict],
                          output_path: str,
                          ground_truth: List[str] = None):
        """
        Create timeline visualization showing predictions over time.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Extract data
        frames = [p.get('frame_id', i) for i, p in enumerate(predictions)]
        labels = [p['label'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # Map labels to numbers
        label_to_num = {'none': 0, 'left': 1, 'right': 2, 'both': 3}
        label_nums = [label_to_num.get(l, 0) for l in labels]
        
        # Plot predictions
        ax.plot(frames, label_nums, 'o-', label='Prediction',
               color='blue', linewidth=2, markersize=4)
        
        # Plot ground truth if available
        if ground_truth:
            gt_nums = [label_to_num.get(gt, 0) for gt in ground_truth]
            ax.plot(frames, gt_nums, 's-', label='Ground Truth',
                   color='green', linewidth=1, markersize=3, alpha=0.7)
        
        # Plot confidence as background
        ax2 = ax.twinx()
        ax2.fill_between(frames, confidences, alpha=0.3, color='gray', label='Confidence')
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_ylim([0, 1])
        
        # Formatting
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Signal State', fontsize=12)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['None', 'Left', 'Right', 'Both'])
        ax.set_title('Turn Signal Detection Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved timeline visualization: {output_path}")
    
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
                
                # Timeline visualization
                timeline_path = self.output_dir / f"{safe_id}_timeline.png"
                self.visualize_timeline(
                    data['predictions'],
                    str(timeline_path),
                    data.get('ground_truth')
                )
                
                # Video visualization (if images available)
                if 'images' in data and data['images']:
                    video_path = self.output_dir / f"{safe_id}_annotated.mp4"
                    self.visualize_sequence(
                        data['images'],
                        data['predictions'],
                        str(video_path),
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
