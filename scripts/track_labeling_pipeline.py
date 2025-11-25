import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import argparse
import sys

from turn_signal_analyzer import analyze_sequence_computational


# ============================================================================
class Config:
    """Configuration for the labeling pipeline."""
    
    # Sampling strategy: [0, 4, 8, 12, ...]
    SAMPLE_INTERVAL = 4
    SAMPLE_START_INDEX = 0
    
    # Minimum frames needed
    MIN_SAMPLED_FRAMES = 4  # Need at least 4 sampled frames
    BATCH_SIZE = 4  # Analyze 4 sampled frames at a time
    
    # Paths
    IMAGE_BASE_PATH = '/gpfs/space/projects/ml2024'
    OUTPUT_DIR = 'labeled_dataset'
    
    # ResNet integration - START SEQUENCES AT FIRST LIGHT
    USE_RESNET_FILTERING = False  # Set True to trim sequences to start at first light
    RESNET_WINDOW_FRAMES = 200  # Extended window around ResNet detections
    
    # Advanced: Dynamic window sizing based on detection pattern
    USE_DYNAMIC_WINDOW = False  # Adjust window based on blink frequency
    MIN_WINDOW_SIZE = 12
    MAX_WINDOW_SIZE = 32


# ============================================================================
def load_resnet_predictions(resnet_file: str) -> Set[str]:
    """
    Load ResNet predictions (paths where lights are supposedly on).
    """
    with open(resnet_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    # Keep full relative paths to handle duplicates in different folders
    # Normalize to match crop_path format
    normalized_paths = set()
    for p in paths:
        # Extract everything after '/gpfs/space/projects/ml2024/' to match crop_path format
        if '/gpfs/space/projects/ml2024/' in p:
            relative_path = p.split('/gpfs/space/projects/ml2024/')[-1]
            normalized_paths.add(relative_path)
        else:
            # Fallback: use last 2 path components
            parts = Path(p).parts
            if len(parts) >= 2:
                normalized_paths.add(str(Path(parts[-2]) / parts[-1]))
            else:
                normalized_paths.add(Path(p).name)
    
    print(f"Loaded {len(normalized_paths):,} ResNet predictions")
    return normalized_paths


def find_resnet_windows(sequence_df: pd.DataFrame,
                       resnet_predictions: Set[str],
                       window_size: int = 12) -> List[Tuple[int, int]]:
    """
    Find windows around ResNet detections for focused analysis.
    """
    # Normalize crop_path to match ResNet format
    def normalize_path(crop_path):
        # Extract relative path after /gpfs/space/projects/ml2024/
        if '/gpfs/space/projects/ml2024/' in crop_path:
            return crop_path.split('/gpfs/space/projects/ml2024/')[-1]
        # Fallback: use last 2 components
        parts = Path(crop_path).parts
        if len(parts) >= 2:
            return str(Path(parts[-2]) / parts[-1])
        return Path(crop_path).name
    
    sequence_df['normalized_path'] = sequence_df['crop_path'].apply(normalize_path)
    
    # Find frames with ResNet detections
    detection_indices = sequence_df[
        sequence_df['normalized_path'].isin(resnet_predictions)
    ].index.tolist()
    
    if not detection_indices:
        return []
    
    # Create windows around detections
    windows = []
    for idx in detection_indices:
        start = max(0, idx - window_size // 2)
        end = min(len(sequence_df), idx + window_size // 2)
        windows.append((start, end))
    
    # Merge overlapping windows
    if not windows:
        return []
    
    windows.sort()
    merged = [windows[0]]
    
    for start, end in windows[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    return merged


# ============================================================================
def sample_frames(sequence_df: pd.DataFrame,
                 interval: int = 4,
                 start_idx: int = 0) -> pd.DataFrame:
    """
    Sample frames at indices [0, 4, 8, 12, ...].
    """
    sequence_df = sequence_df.sort_values('frame_id').reset_index(drop=True)
    
    # Sample starting from start_idx, then every interval
    sampled_indices = range(start_idx, len(sequence_df), interval)
    sampled = sequence_df.iloc[list(sampled_indices)].copy()
    
    return sampled


def create_batches(sampled_df: pd.DataFrame, batch_size: int = 4) -> List[pd.DataFrame]:
    """Split sampled frames into batches for analysis."""
    batches = []
    n_frames = len(sampled_df)
    
    for i in range(0, n_frames, batch_size):
        batch = sampled_df.iloc[i:i+batch_size]
        batches.append(batch)
    
    return batches


def should_process_sequence(sequence_df: pd.DataFrame,
                           sample_interval: int = 4,
                           min_sampled: int = 4) -> bool:
    """Check if sequence has enough frames to be processed."""
    n_frames = len(sequence_df)
    n_sampled = len(range(0, n_frames, sample_interval))
    return n_sampled >= min_sampled


# ============================================================================
# LABELING
# ============================================================================

def label_batch(batch_df: pd.DataFrame,
                image_base_path: str) -> Dict:
    """Run computational analysis on a batch of frames."""
    image_paths = [str(Path(image_base_path) / p) for p in batch_df['crop_path']]
    
    # Check if images exist
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        return {
            'predicted_signal': 'error',
            'error': f'Missing {len(missing)} images'
        }
    
    # Run analysis
    try:
        result = analyze_sequence_computational(image_paths, fps=5.0)
        return result
    except Exception as e:
        return {
            'predicted_signal': 'error',
            'error': str(e)
        }


def back_label_frames(full_sequence_df: pd.DataFrame,
                     batch_labels: List[Dict]) -> pd.DataFrame:
    """
    Apply same label to ALL frames in sequence (from first batch's label).
    
    Since turn signals are usually consistent throughout a sequence,
    we use the first batch's prediction for the entire sequence.
    """
    result_df = full_sequence_df.copy()
    
    # Use first non-error batch label
    sequence_label = 'none'
    for batch_label in batch_labels:
        label = batch_label.get('predicted_signal', 'none')
        if label != 'error':
            sequence_label = label
            break
    
    # Apply to ALL frames
    result_df['predicted_label'] = sequence_label
    
    return result_df


# ============================================================================
def find_signal_start(sequence_df: pd.DataFrame,
                     resnet_predictions: Set[str]) -> Optional[int]:
    """
    Find the first frame where a turn signal is detected by ResNet.
    """
    def normalize_path(crop_path):
        if '/gpfs/space/projects/ml2024/' in crop_path:
            return crop_path.split('/gpfs/space/projects/ml2024/')[-1]
        parts = Path(crop_path).parts
        if len(parts) >= 2:
            return str(Path(parts[-2]) / parts[-1])
        return Path(crop_path).name
    
    sequence_df['normalized_path'] = sequence_df['crop_path'].apply(normalize_path)
    
    # Find first frame with ResNet detection
    with_light = sequence_df[sequence_df['normalized_path'].isin(resnet_predictions)]
    
    if len(with_light) == 0:
        return None
    
    return with_light.index[0]


def process_single_sequence(sequence: str,
                           sequence_df: pd.DataFrame,
                           image_base_path: str,
                           resnet_predictions: Optional[Set[str]] = None) -> Optional[pd.DataFrame]:
    """Process a single vehicle track sequence."""
    
    # Optional: Start sequence at first light detection
    if resnet_predictions and Config.USE_RESNET_FILTERING:
        signal_start_idx = find_signal_start(sequence_df, resnet_predictions)
        
        if signal_start_idx is None:
            # No lights detected - label entire sequence as 'none'
            sequence_df['predicted_label'] = 'none'
            sequence_df['sequence'] = sequence
            sequence_df['sampled_frame_id'] = -1
            sequence_df['signal_start_frame'] = -1
            return sequence_df
        
        # Trim sequence to start at first light
        sequence_df = sequence_df.iloc[signal_start_idx:].reset_index(drop=True)
        signal_start_frame = signal_start_idx
    else:
        signal_start_frame = -1
    
    # Check if valid after trimming
    if not should_process_sequence(sequence_df, Config.SAMPLE_INTERVAL, Config.MIN_SAMPLED_FRAMES):
        return None
    
    # Sample frames starting from index 0 (which is now the first light frame)
    sampled_df = sample_frames(sequence_df, Config.SAMPLE_INTERVAL, Config.SAMPLE_START_INDEX)
    
    # Create batches
    batches = create_batches(sampled_df, Config.BATCH_SIZE)
    
    # Label each batch
    batch_labels = []
    for batch in batches:
        label = label_batch(batch, image_base_path)
        batch_labels.append(label)
    
    # Back-label all frames with same label
    labeled_df = back_label_frames(sequence_df, batch_labels)
    labeled_df['sequence'] = sequence
    
    # Store which frames were sampled (for review)
    labeled_df['sampled_frame_id'] = -1
    labeled_df.loc[sampled_df.index, 'sampled_frame_id'] = sampled_df['frame_id']
    
    # Store signal start info
    labeled_df['signal_start_frame'] = signal_start_frame
    
    return labeled_df


def process_all_tracks(tracks_csv: str,
                       image_base_path: str,
                       output_csv: str,
                       resnet_predictions: Optional[str] = None,
                       max_sequences: Optional[int] = None) -> pd.DataFrame:
    """Process all vehicle tracks and generate labeled dataset."""
    
    # Load tracks
    print(f"Loading tracks from {tracks_csv}...")
    df = pd.read_csv(tracks_csv)
    df['sequence_id'] = df['sequence'].astype(str) + "_" + df['track_id'].astype(str)
    
    print(f"Total frames: {len(df):,}")
    print(f"Total sequences: {df['sequence_id'].nunique():,}")
    
    # Load ResNet predictions if provided
    resnet_set = None
    if resnet_predictions:
        resnet_set = load_resnet_predictions(resnet_predictions)
        if Config.USE_RESNET_FILTERING:
            print("ResNet filtering enabled")
    
    # Process sequences
    all_results = []
    sequences = df.groupby('sequence')
    
    if max_sequences:
        sequences = list(sequences)[:max_sequences]
    
    total = len(sequences) if isinstance(sequences, list) else df['sequence_id'].nunique()
    skipped = 0
    processed = 0
    
    print(f"\nProcessing {total:,} sequences...")
    print(f"Sampling: indices [0, 4, 8, 12, ...]")
    print(f"Min sampled frames: {Config.MIN_SAMPLED_FRAMES}")
    print("-" * 60)
    
    for seq_id, seq_df in tqdm(sequences if isinstance(sequences, list) else sequences,
                               desc="Processing"):
        
        result = process_single_sequence(
            seq_id,
            seq_df.sort_values('frame_id').reset_index(drop=True),
            image_base_path,
            resnet_set
        )
        
        if result is None:
            skipped += 1
            continue
        
        all_results.append(result)
        processed += 1
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed:,}")
    print(f"Skipped: {skipped:,}")
    
    if not all_results:
        print("\nWARNING: No sequences were processed!")
        return pd.DataFrame()
    
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"\nSaved to: {output_csv}")
    print(f"  Total labeled frames: {len(final_df):,}")
    
    # Label distribution
    print("\nLabel distribution (by sequence):")
    label_counts = final_df.groupby('predicted_label')['sequence'].nunique()
    for label, count in label_counts.items():
        pct = count / processed * 100
        print(f"  {label:8s}: {count:5,} sequences ({pct:5.1f}%)")
    
    return final_df


# ============================================================================
class InteractiveReviewer:
    """Interactive tool for reviewing and correcting labels."""
    
    def __init__(self, labeled_csv: str, image_base_path: str):
        self.df = pd.read_csv(labeled_csv)
        self.image_base_path = image_base_path
        
        # Get unique sequences with their sampled frames
        self.sequences = []
        for seq_id, group in self.df.groupby('sequence'):
            # Get sampled frames (indices [0, 4, 8, 12])
            sampled = group[group['sampled_frame_id'] >= 0].sort_values('frame_id')
            
            if len(sampled) >= 4:
                # Take first 4 sampled frames
                first_four = sampled.head(4)
                label = group['predicted_label'].iloc[0]
                
                self.sequences.append({
                    'sequence': seq_id,
                    'frames': first_four,
                    'all_sampled': sampled,  # All sampled frames
                    'all_frames': group,
                    'label': label,
                    'current_start_idx': 0  # Which frame to start showing from
                })
        
        self.current_idx = 0
        self.corrections = {}
        self.history = []
        self.fig = None
        
        print(f"Loaded {len(self.sequences):,} sequences for review")
    
    def run(self):
        """Start interactive review."""
        print("\n" + "=" * 60)
        print("INTERACTIVE REVIEW MODE")
        print("=" * 60)
        print("Controls:")
        print("  ENTER    - Accept current label and continue")
        print("  x        - Go back to previous sequence")
        print("  c        - Change label (then enter: l/r/h/n)")
        print("  s        - Shift frames forward (show next 4 frames)")
        print("  b        - Shift frames back (show previous 4 frames)")
        print("  f <num>  - Start from specific frame number")
        print("  q        - Quit and save corrections")
        print("=" * 60 + "\n")
        
        while self.current_idx < len(self.sequences):
            self.show_sequence()
            
            action = input("\nAction [ENTER/x/c/s/b/f/q]: ").strip().lower()
            
            if action == 'q':
                break
            elif action == 'x':
                self.go_back()
            elif action == 'c':
                self.change_label()
            elif action == 's':
                self.shift_frames_forward()
            elif action == 'b':
                self.shift_frames_back()
            elif action.startswith('f'):
                self.jump_to_frame(action)
            else:  # Enter or empty
                self.accept_and_next()
        
        self.save_corrections()
    
    def show_sequence(self):
        """Display current sequence."""
        if self.current_idx >= len(self.sequences):
            print("Review complete!")
            return
        
        seq = self.sequences[self.current_idx]
        
        # Get frames to show based on current_start_idx
        all_sampled = seq['all_sampled'].reset_index(drop=True)
        start = seq['current_start_idx']
        end = min(start + 4, len(all_sampled))
        frames_to_show = all_sampled.iloc[start:end]
        
        print(f"\n[{self.current_idx + 1}/{len(self.sequences)}] Sequence: {seq['sequence']}")
        print(f"Current label: {seq['label'].upper()}")
        print(f"Showing frames {start}-{end-1} of {len(all_sampled)} sampled frames")
        
        if 'signal_start_frame' in seq['all_frames'].columns:
            sig_start = seq['all_frames']['signal_start_frame'].iloc[0]
            if sig_start >= 0:
                print(f"Signal detected starting at frame {sig_start}")
        
        # Close previous figure if exists
        if self.fig is not None:
            plt.close(self.fig)
        
        # Show images
        n_frames = len(frames_to_show)
        self.fig, axes = plt.subplots(1, max(4, n_frames), figsize=(16, 4))
        
        if n_frames == 1:
            axes = [axes]
        
        for i, (_, row) in enumerate(frames_to_show.iterrows()):
            img_path = Path(self.image_base_path) / row['crop_path']
            
            if img_path.exists():
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Frame {row['frame_id']}", fontsize=12)
            else:
                axes[i].text(0.5, 0.5, 'Image\nNot Found',
                           ha='center', va='center')
                axes[i].set_title(f"Frame {row['frame_id']}", fontsize=12)
            
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')
        
        # Add label text
        label_colors = {'left': 'blue', 'right': 'orange', 'hazard': 'red', 'none': 'green'}
        color = label_colors.get(seq['label'], 'black')
        
        self.fig.suptitle(f"Label: {seq['label'].upper()}",
                         fontsize=16, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def shift_frames_forward(self):
        """Show next 4 frames in sequence."""
        seq = self.sequences[self.current_idx]
        all_sampled = seq['all_sampled']
        
        new_start = seq['current_start_idx'] + 4
        if new_start >= len(all_sampled):
            print("Already at end of sequence!")
            return
        
        seq['current_start_idx'] = new_start
        if self.fig:
            plt.close(self.fig)
        print("Showing next 4 frames...")
    
    def shift_frames_back(self):
        """Show previous 4 frames in sequence."""
        seq = self.sequences[self.current_idx]
        
        new_start = max(0, seq['current_start_idx'] - 4)
        if new_start == seq['current_start_idx']:
            print("⚠ Already at start of sequence!")
            return
        
        seq['current_start_idx'] = new_start
        if self.fig:
            plt.close(self.fig)
        print("Showing previous 4 frames...")
    
    def jump_to_frame(self, action: str):
        """Jump to specific frame number."""
        try:
            parts = action.split()
            if len(parts) < 2:
                print("Usage: f <frame_number>")
                return
            
            target_frame = int(parts[1])
            seq = self.sequences[self.current_idx]
            all_sampled = seq['all_sampled'].reset_index(drop=True)
            
            # Find which sampled frame batch contains this frame
            frame_indices = all_sampled['frame_id'].values
            
            if target_frame < frame_indices[0] or target_frame > frame_indices[-1]:
                print(f"Frame {target_frame} not in sampled frames [{frame_indices[0]}-{frame_indices[-1]}]")
                return
            
            # Find closest sampled frame
            closest_idx = np.argmin(np.abs(frame_indices - target_frame))
            
            # Start showing from this frame (aligned to batch of 4)
            new_start = (closest_idx // 4) * 4
            seq['current_start_idx'] = new_start
            
            if self.fig:
                plt.close(self.fig)
            print(f"Jumping to frame ~{target_frame}...")
            
        except (ValueError, IndexError) as e:
            print(f"Invalid frame number: {e}")
    
    def accept_and_next(self):
        """Accept current label and move to next."""
        # Reset frame position for next sequence
        seq = self.sequences[self.current_idx]
        seq['current_start_idx'] = 0
        
        self.history.append(self.current_idx)
        self.current_idx += 1
        
        if self.fig:
            plt.close(self.fig)
    
    def go_back(self):
        """Go back to previous sequence."""
        if self.history:
            self.current_idx = self.history.pop()
            
            # Reset frame position
            seq = self.sequences[self.current_idx]
            seq['current_start_idx'] = 0
            
            if self.fig:
                plt.close(self.fig)
            print("Going back...")
        else:
            print("Already at first sequence!")
    
    def change_label(self):
        """Change label for current sequence."""
        seq = self.sequences[self.current_idx]
        
        print(f"\nCurrent label: {seq['label']}")
        new_label = input("New label [l/r/h/n]: ").strip().lower()
        
        label_map = {
            'l': 'left',
            'r': 'right',
            'h': 'hazard',
            'n': 'none'
        }
        
        if new_label in label_map:
            new_label_full = label_map[new_label]
            seq['label'] = new_label_full
            self.corrections[seq['sequence']] = new_label_full
            print(f"Changed to: {new_label_full}")
            
            # Update figure title
            if self.fig:
                label_colors = {'left': 'blue', 'right': 'orange', 'hazard': 'red', 'none': 'green'}
                color = label_colors.get(new_label_full, 'black')
                self.fig.suptitle(f"Label: {new_label_full.upper()}",
                                 fontsize=16, fontweight='bold', color=color)
                plt.draw()
        else:
            print("Invalid label. Keeping original.")
    
    def save_corrections(self):
        """Save corrected labels back to CSV."""
        if not self.corrections:
            print("No corrections made")
            return
        
        print(f"Saving {len(self.corrections)} corrections...")
        
        # Apply corrections
        for seq_id, new_label in self.corrections.items():
            self.df.loc[self.df['sequence_id'] == seq_id, 'predicted_label'] = new_label
        
        # Save with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"labeled_tracks_reviewed_{timestamp}.csv"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        # Save corrections log
        corrections_log = {
            'timestamp': timestamp,
            'num_corrections': len(self.corrections),
            'corrections': self.corrections
        }
        
        log_path = f"corrections_log_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(corrections_log, f, indent=2)
        
        print(f"Corrections log: {log_path}")


# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Label and review vehicle turn signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label all tracks
  python track_labeling_pipeline.py label --tracks tracks.csv --output labeled.csv
  
  # Label with ResNet filtering
  python track_labeling_pipeline.py label --tracks tracks.csv --resnet lights.txt --use-resnet
  
  # Interactive review
  python track_labeling_pipeline.py review --labeled labeled.csv --image-base ../seq_img
  
  # Test on 100 sequences
  python track_labeling_pipeline.py label --tracks tracks.csv --output test.csv --max 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Label command
    label_parser = subparsers.add_parser('label', help='Label sequences')
    label_parser.add_argument('--tracks', required=True, help='Input tracks CSV')
    label_parser.add_argument('--image-base', default='../seq_img', help='Image base path')
    label_parser.add_argument('--output', default='labeled_tracks.csv', help='Output CSV')
    label_parser.add_argument('--resnet', help='ResNet predictions file (one path per line)')
    label_parser.add_argument('--use-resnet', action='store_true', help='Enable ResNet filtering')
    label_parser.add_argument('--max', type=int, help='Max sequences (for testing)')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Interactive review')
    review_parser.add_argument('--labeled', required=True, help='Labeled tracks CSV')
    review_parser.add_argument('--image-base', default='../seq_img', help='Image base path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'label':
        # Update config
        Config.IMAGE_BASE_PATH = args.image_base
        if args.use_resnet:
            Config.USE_RESNET_FILTERING = True
        
        # Process tracks
        process_all_tracks(
            args.tracks,
            args.image_base,
            args.output,
            args.resnet,
            args.max
        )
    
    elif args.command == 'review':
        # Start interactive review
        reviewer = InteractiveReviewer(args.labeled, args.image_base)
        reviewer.run()


if __name__ == '__main__':
    main()
