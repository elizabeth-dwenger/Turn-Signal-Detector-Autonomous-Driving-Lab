"""
Visualization utilities for turn signal detection results.
Import this in the notebook for enhanced plotting functions.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_sequence_timeline_with_temporal(predictions, ground_truth=None, fps=10.0,
                                         sequence_length=None):
    """
    Plot timeline of predictions with temporal localization markers.
    """
    # Handle video mode (single prediction)
    if isinstance(predictions, dict):
        predictions = [predictions]
        is_video_mode = True
        if sequence_length is None:
            sequence_length = predictions[0].get('end_frame', 50) + 1
    else:
        is_video_mode = False
        sequence_length = len(predictions)
    
    fig = plt.figure(figsize=(16, 8))
    
    if is_video_mode:
        # Video mode: show temporal boundaries
        pred = predictions[0]
        
        # Create subplot layout
        ax1 = plt.subplot(3, 1, 1)  # Signal visualization
        ax2 = plt.subplot(3, 1, 2)  # Temporal info
        ax3 = plt.subplot(3, 1, 3)  # Metadata
        
        # Main signal visualization
        frames = list(range(sequence_length))
        label = pred.get('label', 'none')
        
        label_map = {'none': 0, 'left': 1, 'right': 2, 'both': 3}
        label_num = label_map.get(label, 0)
        
        # Plot as a bar showing the prediction
        ax1.barh(0, sequence_length, height=0.5, color='lightgray', alpha=0.3)
        ax1.text(sequence_length/2, 0, f'{label.upper()}',
                ha='center', va='center', fontsize=14, weight='bold')
        
        ax1.set_xlim([0, sequence_length])
        ax1.set_ylim([-1, 1])
        ax1.set_yticks([])
        ax1.set_xlabel('Frame Index', fontsize=12)
        ax1.set_title('Video-Mode Prediction', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Temporal boundaries
        start_f = pred.get('start_frame')
        end_f = pred.get('end_frame')
        start_t = pred.get('start_time_seconds')
        end_t = pred.get('end_time_seconds')
        
        if start_f is not None and end_f is not None:
            # Highlight the signal region
            signal_region = ax2.barh(0, end_f - start_f, left=start_f, height=0.8,
                                     color='orange', alpha=0.5, label='Signal Active')
            
            # Mark boundaries
            ax2.axvline(start_f, color='green', linestyle='--', linewidth=3,
                       label=f'Start: frame {start_f}')
            ax2.axvline(end_f, color='red', linestyle='--', linewidth=3,
                       label=f'End: frame {end_f}')
            
            # Add time annotations
            if start_t is not None:
                ax2.text(start_f, 0.5, f'{start_t:.2f}s', fontsize=11, ha='center',
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
            if end_t is not None:
                ax2.text(end_f, 0.5, f'{end_t:.2f}s', fontsize=11, ha='center',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
            
            # Calculate duration
            duration_frames = end_f - start_f + 1
            duration_seconds = (end_t - start_t) if (end_t and start_t) else duration_frames / fps
            
            ax2.text((start_f + end_f) / 2, -0.5,
                    f'Duration: {duration_frames} frames ({duration_seconds:.2f}s)',
                    ha='center', fontsize=10, style='italic')
        
        ax2.set_xlim([0, sequence_length])
        ax2.set_ylim([-1, 1])
        ax2.set_yticks([])
        ax2.set_xlabel('Frame Index', fontsize=12)
        ax2.set_title('Temporal Localization', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add secondary time axis
        ax2_time = ax2.twiny()
        ax2_time.set_xlim([0, sequence_length / fps])
        ax2_time.set_xlabel('Time (seconds)', fontsize=12)
        
        # Metadata and reasoning
        ax3.axis('off')
        reasoning = pred.get('reasoning', 'No reasoning provided')
        
        # Wrap text
        import textwrap
        wrapped = textwrap.fill(f"Reasoning: {reasoning}", width=100)
        
        ax3.text(0.05, 0.8, wrapped, fontsize=10, verticalalignment='top',
                family='monospace', wrap=True)
        
        # Add temporal summary
        if start_f is not None and end_f is not None:
            summary = f"Signal: {label.upper()} | Frames {start_f}-{end_f}"
            if start_t is not None and end_t is not None:
                summary += f" | Time {start_t:.2f}s-{end_t:.2f}s"
            ax3.text(0.05, 0.5, summary, fontsize=11, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
    else:
        # Single-image mode: per-frame predictions
        ax1 = plt.subplot(1, 1, 1)
        
        frames = list(range(len(predictions)))
        labels = [p.get('label', 'none') for p in predictions]
        
        label_map = {'none': 0, 'left': 1, 'right': 2, 'both': 3}
        label_nums = [label_map.get(l, 0) for l in labels]
        
        # Plot labels
        ax1.plot(frames, label_nums, 'o-', linewidth=2, markersize=6, label='Prediction')
        
        # Plot ground truth if available
        if ground_truth:
            gt_nums = [label_map.get(gt, 0) for gt in ground_truth]
            ax1.plot(frames, gt_nums, 's-', linewidth=1, markersize=4,
                    alpha=0.7, label='Ground Truth', color='green')
        
        ax1.set_ylabel('Signal State', fontsize=12)
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(['None', 'Left', 'Right', 'Both'])
        ax1.set_title('Frame-by-Frame Predictions', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlabel('Frame Index', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_temporal_comparison(results_dict, fps=10.0):
    """
    Compare temporal predictions from multiple models/prompts.
    """
    fig, axes = plt.subplots(len(results_dict), 1, figsize=(16, 3*len(results_dict)))
    
    if len(results_dict) == 1:
        axes = [axes]
    
    for ax, (model_name, pred) in zip(axes, results_dict.items()):
        start_f = pred.get('start_frame')
        end_f = pred.get('end_frame')
        label = pred.get('label', 'none')
        
        # Determine sequence length
        seq_len = end_f + 10 if end_f else 50
        
        # Draw timeline
        ax.barh(0, seq_len, height=0.5, color='lightgray', alpha=0.3)
        
        if start_f is not None and end_f is not None:
            # Signal region
            ax.barh(0, end_f - start_f, left=start_f, height=0.5,
                   color='orange', alpha=0.7)
            
            # Boundaries
            ax.axvline(start_f, color='green', linestyle='--', linewidth=2)
            ax.axvline(end_f, color='red', linestyle='--', linewidth=2)
            
            # Times
            start_t = pred.get('start_time_seconds', start_f / fps)
            end_t = pred.get('end_time_seconds', end_f / fps)
            
            ax.text(start_f, 0.3, f'{start_t:.2f}s', fontsize=9, ha='center')
            ax.text(end_f, 0.3, f'{end_t:.2f}s', fontsize=9, ha='center')
        
        # Model info
        ax.text(0, 0, f'{model_name}: {label.upper()}',
               fontsize=11, va='center', ha='left', weight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim([0, seq_len])
        ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([])
        ax.set_xlabel('Frame Index', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Temporal Localization Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    return fig


def print_temporal_summary(prediction):
    """
    Print a nice summary of temporal information.
    """
    print("="*80)
    print("TEMPORAL LOCALIZATION SUMMARY")
    print("="*80)
    
    label = prediction.get('label', 'none')
    print(f"\nSignal Type: {label.upper()}")
    
    start_f = prediction.get('start_frame')
    end_f = prediction.get('end_frame')
    start_t = prediction.get('start_time_seconds')
    end_t = prediction.get('end_time_seconds')
    
    if start_f is not None and end_f is not None:
        print(f"\nTemporal Boundaries:")
        print(f"  Start: Frame {start_f:4d} ({start_t:.2f}s)" if start_t else f"  Start: Frame {start_f:4d}")
        print(f"  End:   Frame {end_f:4d} ({end_t:.2f}s)" if end_t else f"  End:   Frame {end_f:4d}")
        
        duration_frames = end_f - start_f + 1
        duration_seconds = (end_t - start_t) if (end_t and start_t) else None
        
        print(f"  Duration: {duration_frames} frames", end="")
        if duration_seconds:
            print(f" ({duration_seconds:.2f}s)")
        else:
            print()
    else:
        print(f"\nNo temporal boundaries (signal: {label})")
    
    reasoning = prediction.get('reasoning', '')
    if reasoning:
        print(f"\nReasoning:")
        print(f"  {reasoning}")
    
    print("="*80)


# Export functions for easy import
__all__ = [
    'plot_sequence_timeline_with_temporal',
    'plot_temporal_comparison',
    'print_temporal_summary'
]
