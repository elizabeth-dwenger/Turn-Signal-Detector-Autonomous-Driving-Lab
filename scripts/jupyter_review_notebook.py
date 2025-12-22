import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from datetime import datetime


class ProgressTracker:
    """Tracks which sequences were reviewed, corrections made, and flagged sequences."""
    
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, "r") as f:
                data = json.load(f)
            self.reviewed = set(data.get("reviewed", []))
            self.corrections = data.get("corrections", {})
            self.image_flags = data.get("image_flags", {})
        else:
            self.reviewed = set()
            self.corrections = {}
            self.image_flags = {}

    def mark_reviewed(self, seq_id):
        self.reviewed.add(str(seq_id))
        self._save()

    def save_correction(self, seq_id, frame_id, new_label, custom_key=None):
        """Save correction. Use custom_key (crop_path) if provided."""
        if custom_key:
            key = custom_key
        else:
            key = f"{seq_id}_{frame_id}"
        self.corrections[key] = new_label
        self._save()
    
    def flag_bad_images(self, seq_id, flag=True):
        """Set or clear flag for incorrectly sequenced images."""
        if flag:
            self.image_flags[str(seq_id)] = True
        else:
            self.image_flags.pop(str(seq_id), None)
        self._save()

    def get_unreviewed_sequences(self, all_sequences):
        return [s for s in all_sequences if str(s) not in self.reviewed]

    def _save(self):
        data = {
            "reviewed": sorted(self.reviewed),
            "corrections": self.corrections,
            "image_flags": self.image_flags
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

class JupyterReviewer:
    """Web-based review tool for per-frame labeling with corrected back-filling."""
    
    def __init__(self, labeled_csv: str, image_base_path: str, progress: Optional['ProgressTracker'] = None):
        self.df = pd.read_csv(labeled_csv)
        self.image_base_path = image_base_path
        self.progress = progress
        
        # Map crop_path -> label (Unique identifier)
        self.correction_labels = {}
        
        # Load existing corrections from progress
        if self.progress:
            # We assume progress saves keys as strings, we load them back
            for key, label in self.progress.corrections.items():
                self.correction_labels[key] = label
        
        # Get sequences
        self.sequences = []
        for seq_id, group in self.df.groupby('sequence_id'):
            # Sampled frames are those with sampled_frame_id >= 0
            sampled = group[group['sampled_frame_id'] >= 0].sort_values('frame_id')
            
            if len(sampled) >= 4:
                self.sequences.append({
                    'sequence_id': seq_id,
                    'all_sampled': sampled,
                    'all_frames': group,
                    'label': group['predicted_label'].iloc[0],
                    'original_label': group['predicted_label'].iloc[0]
                })
        
        self.current_idx = 0
        self.frame_offset = 0
        
        # Create widgets
        self._create_widgets()
        
        print(f"Loaded {len(self.sequences):,} sequences for review")
        if self.correction_labels:
            print(f"Loaded {len(self.correction_labels)} existing corrections")
    
    def _create_widgets(self):
        """Create interactive widgets."""
        
        # Navigation buttons
        self.btn_prev = widgets.Button(description='◀ Previous (p)', button_style='info', layout=widgets.Layout(width='150px'))
        self.btn_next = widgets.Button(description='Next ▶ (n)', button_style='success', layout=widgets.Layout(width='150px'))
        
        # Frame navigation
        self.btn_frames_back = widgets.Button(description='◀◀ Back 4 (b)', layout=widgets.Layout(width='120px'))
        self.btn_frames_forward = widgets.Button(description='Forward 4 ▶▶ (f)', layout=widgets.Layout(width='120px'))

        self.chk_bad_images = widgets.Checkbox(value=False, description='Bad Image Sequence?', layout=widgets.Layout(width='200px'))
        self.chk_bad_images.observe(self._on_flag_change, names='value')
        
        self.txt_jump_frame = widgets.IntText(value=0, description='Jump:', layout=widgets.Layout(width='150px'))
        self.btn_jump = widgets.Button(description='Go', button_style='warning', layout=widgets.Layout(width='60px'))
        
        # Label selection for CURRENT 4 FRAMES
        self.radio_label = widgets.RadioButtons(
            options=['left', 'right', 'hazard', 'none'],
            description='Label these 4 frames:',
            layout=widgets.Layout(width='250px')
        )
        
        self.label_progress = widgets.Label(value='')
        self.label_info = widgets.HTML(value='')
        self.output_images = widgets.Output()
        
        # Bind events
        self.btn_prev.on_click(lambda b: self.go_previous())
        self.btn_next.on_click(lambda b: self.go_next())
        self.btn_frames_back.on_click(lambda b: self.shift_frames(-4))
        self.btn_frames_forward.on_click(lambda b: self.shift_frames(4))
        self.btn_jump.on_click(lambda b: self.jump_to_frame())
        self.radio_label.observe(self._on_label_change, names='value')
    
    def _on_label_change(self, change):
        """Handle label change for CURRENT 4 displayed frames."""
        new_label = change['new']
        seq = self.sequences[self.current_idx]
        all_sampled = seq['all_sampled'].sort_values('frame_id').reset_index(drop=True)
        
        # Get the 4 frames currently displayed
        start = self.frame_offset
        end = min(start + 4, len(all_sampled))
        displayed_frames = all_sampled.iloc[start:end]
        
        # Label these specific frames using CROP_PATH as key
        current_frame_ids = []
        for _, row in displayed_frames.iterrows():
            crop_path = row['crop_path']
            
            # Save to internal dict
            self.correction_labels[crop_path] = new_label
            
            # Save to progress tracker immediately
            if self.progress:
                # We use the sequence_id only for context, but store by unique key in corrections
                self.progress.save_correction(seq['sequence_id'], "crop_key", new_label, custom_key=crop_path)
            
            current_frame_ids.append(row['frame_id'])
        
        print(f"Labeled frames {current_frame_ids} as '{new_label}'")
        self._update_info()
    
    def _on_flag_change(self, change):
        if self.progress:
            seq_id = self.sequences[self.current_idx]['sequence_id']
            self.progress.flag_bad_images(seq_id, flag=change['new'])
    
    def display(self):
        """Display the review interface."""
        nav_box = widgets.HBox([self.btn_prev, self.btn_next, self.label_progress])
        frame_nav_box = widgets.HBox([self.btn_frames_back, self.btn_frames_forward, self.txt_jump_frame, self.btn_jump])
        
        control_box = widgets.VBox([
            widgets.HTML("<h3>Navigation</h3>"),
            nav_box,
            widgets.HTML("<h4>Frame Navigation</h4>"),
            frame_nav_box,
            widgets.HTML("<h4>Change Label (applies to currently displayed 4 frames + backfill)</h4>"),
            self.radio_label,
            self.chk_bad_images,
            widgets.HTML("<br>"),
            self.label_info
        ])
        
        main_box = widgets.VBox([self.output_images, control_box])
        display(main_box)
        self.show_current_sequence()
    
    def show_current_sequence(self):
        if self.current_idx >= len(self.sequences):
            with self.output_images:
                clear_output(wait=True)
                print("Review complete!")
            return
        
        seq = self.sequences[self.current_idx]
        
        # Update progress
        self.label_progress.value = f"Sequence {self.current_idx + 1} / {len(self.sequences)}"
        
        if self.progress:
            is_flagged = seq['sequence_id'] in self.progress.image_flags
            self.chk_bad_images.value = is_flagged
        
        # Get frames to show
        all_sampled = seq['all_sampled'].sort_values('frame_id').reset_index(drop=True)
        start = self.frame_offset
        end = min(start + 4, len(all_sampled))
        frames_to_show = all_sampled.iloc[start:end]
        
        if frames_to_show.empty:
             with self.output_images:
                print("No frames to show in this range.")
             return

        # Determine the radio button state based on the FIRST frame displayed
        # We prefer the correction if it exists, otherwise the original prediction
        first_frame_row = frames_to_show.iloc[0]
        first_crop_path = first_frame_row['crop_path']
        
        current_label = self.correction_labels.get(first_crop_path, first_frame_row['predicted_label'])
        
        # Set value without triggering event listener loop if possible,
        # but here we want it to reflect state.
        self.radio_label.unobserve(self._on_label_change, names='value')
        self.radio_label.value = current_label
        self.radio_label.observe(self._on_label_change, names='value')
        
        self._update_info()
        
        # Display images
        with self.output_images:
            clear_output(wait=True)
            
            n_frames = len(frames_to_show)
            fig, axes = plt.subplots(1, max(4, n_frames), figsize=(16, 4))
            if n_frames == 1: axes = [axes]
            
            for i, (_, row) in enumerate(frames_to_show.iterrows()):
                img_path = Path(self.image_base_path) / row['crop_path']
                frame_id = row['frame_id']
                crop_path = row['crop_path']
                
                if img_path.exists():
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    
                    # Check for correction
                    if crop_path in self.correction_labels:
                        lbl = self.correction_labels[crop_path]
                        title = f"Frame {frame_id}\n({lbl})"
                        color = 'blue' # Blue indicates user correction
                    else:
                        title = f"Frame {frame_id}"
                        color = 'black'
                    
                    axes[i].set_title(title, fontsize=12, color=color)
                else:
                    axes[i].text(0.5, 0.5, 'Not Found', ha='center', va='center')
                    axes[i].set_title(f"Frame {frame_id}", fontsize=12)
                
                axes[i].axis('off')
            
            # Hide unused axes
            for i in range(n_frames, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def _update_info(self):
        seq = self.sequences[self.current_idx]
        all_sampled = seq['all_sampled'].sort_values('frame_id')
        start = self.frame_offset
        end = min(start + 4, len(all_sampled))
        
        # Count corrections
        seq_crop_paths = set(seq['all_frames']['crop_path'].unique())
        corrections_in_seq = sum(1 for cp in seq_crop_paths if cp in self.correction_labels)
        
        info_html = f"""
        <div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>
            <h4>Sequence: {seq['sequence_id']}</h4>
            <p><b>Original label:</b> {seq['original_label']}</p>
            <p><b>Showing frames:</b> {start}-{end-1} (Sampled indices)</p>
            <p><b>Corrections in this seq:</b> {corrections_in_seq}</p>
        </div>
        """
        self.label_info.value = info_html

    def go_next(self):
        seq = self.sequences[self.current_idx]
        if self.progress:
            self.progress.mark_reviewed(seq["sequence_id"])
            
        if self.current_idx < len(self.sequences) - 1:
            self.current_idx += 1
            self.frame_offset = 0
            self.show_current_sequence()
        else:
            print("Last sequence reached!")

    def go_previous(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.frame_offset = 0
            self.show_current_sequence()

    def shift_frames(self, delta: int):
        new_offset = self.frame_offset + delta
        seq = self.sequences[self.current_idx]
        if 0 <= new_offset < len(seq['all_sampled']):
            self.frame_offset = new_offset
            self.show_current_sequence()

    def jump_to_frame(self):
        target = self.txt_jump_frame.value
        seq = self.sequences[self.current_idx]
        all_sampled = seq['all_sampled'].reset_index(drop=True)
        frame_indices = all_sampled['frame_id'].values
        
        if target < frame_indices[0] or target > frame_indices[-1]:
            print(f"Frame {target} out of range")
            return
            
        closest_idx = np.argmin(np.abs(frame_indices - target))
        self.frame_offset = (closest_idx // 4) * 4
        self.show_current_sequence()

    def save_corrections(self, output_path: Optional[str] = None):
        """
        Save corrections with strict sequential back-filling.
        1. Identify sampled frames (the anchors).
        2. Determine the intended label for each anchor (User correction OR Original).
        3. Iterate row by row: update active label ONLY when hitting an anchor.
        """
        if not self.correction_labels:
            print("No corrections to save")
            return
        
        print(f"Applying corrections based on {len(self.correction_labels)} unique crop paths...")
        
        # We need to process every sequence that has at least one correction
        # But to be safe, we should process all sequences to ensure consistency
        # (or just those touched). Let's do all to be safe.
        
        # Group entire dataframe by sequence to process sequentially
        grouped = self.df.groupby('sequence_id')
        
        processed_frames = 0
        
        for seq_id, seq_data in grouped:
            # Sort strictly by frame_id
            seq_data = seq_data.sort_values('frame_id')
            
            # This variable tracks the label to apply to "in-between" frames
            # Initialize with the label of the very first frame in the sequence
            first_row = seq_data.iloc[0]
            current_active_label = self.correction_labels.get(
                first_row['crop_path'],
                first_row['predicted_label']
            )
            
            # Iterate through EVERY frame in the sequence
            for idx, row in seq_data.iterrows():
                crop_path = row['crop_path']
                is_sampled = row['sampled_frame_id'] >= 0
                
                if is_sampled:
                    # When we hit a sampled frame, we reset the active label
                    # Check if user corrected THIS specific frame
                    if crop_path in self.correction_labels:
                        current_active_label = self.correction_labels[crop_path]
                    else:
                        # User did NOT correct this frame.
                        # We must revert to the ORIGINAL prediction for this frame.
                        # This prevents "Car" from bleeding into the rest of the sequence
                        # if the user only labeled the first 4 frames.
                        current_active_label = row['predicted_label']
                
                # Apply the active label (whether it was just updated or carried over)
                self.df.at[idx, 'predicted_label'] = current_active_label
                processed_frames += 1

        # Save to CSV
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"labeled_tracks_reviewed_{timestamp}.csv"
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved processed dataframe to: {output_path}")
        return output_path


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def start_review(labeled_csv: str, image_base_path: str = '/gpfs/space/projects/ml2024/'):
    """Start interactive review in Jupyter notebook."""
    reviewer = JupyterReviewer(labeled_csv, image_base_path)
    reviewer.display()
    return reviewer


def review_filtered(labeled_csv: str,
                   image_base_path: str = '/gpfs/space/projects/ml2024/',
                   filter_label: Optional[str] = None,
                   max_sequences: Optional[int] = None,
                   random_sample: bool = False,
                   progress_file: Optional[str] = None,
                   resume: bool = True):
    """Start review with filtering options and resume capability."""
    df = pd.read_csv(labeled_csv)
    
    # Generate progress file name
    if progress_file is None:
        if filter_label:
            progress_file = f'review_progress_{filter_label}.json'
        else:
            progress_file = 'review_progress_all.json'
    
    # Load progress
    progress = ProgressTracker(progress_file) if resume else None
    
    if progress and resume and Path(progress_file).exists():
        print(f"Resuming from {progress_file}")
        
        # Get unreviewed sequences
        all_seq_ids = df['sequence_id'].unique()
        unreviewed = progress.get_unreviewed_sequences(all_seq_ids.tolist())
        
        if len(unreviewed) == 0:
            print("All sequences already reviewed!")
            return None
        
        print(f"   {len(all_seq_ids) - len(unreviewed)} already reviewed")
        print(f"   {len(unreviewed)} remaining")
        
        # Filter to unreviewed
        df = df[df['sequence_id'].isin(unreviewed)]
    
    # Filter by label
    if filter_label:
        sequences_with_label = df[df['predicted_label'] == filter_label]['sequence_id'].unique()
        df = df[df['sequence_id'].isin(sequences_with_label)]
        print(f"Filtered to {len(sequences_with_label)} sequences with label '{filter_label}'")
    
    # Sample sequences
    if max_sequences:
        all_seq_ids = df['sequence_id'].unique()
        
        if random_sample:
            selected = np.random.choice(all_seq_ids, min(max_sequences, len(all_seq_ids)), replace=False)
        else:
            selected = all_seq_ids[:max_sequences]
        
        df = df[df['sequence_id'].isin(selected)]
        print(f"Reviewing {len(selected)} sequences")
    
    # Save filtered dataset temporarily
    temp_path = f'temp_filtered_for_review_{filter_label or "all"}.csv'
    df.to_csv(temp_path, index=False)
    
    # Start review
    if not progress:
        progress = ProgressTracker(progress_file)
    
    reviewer = JupyterReviewer(temp_path, image_base_path, progress=progress)
    reviewer.display()
    return reviewer


def quick_stats(labeled_csv: str):
    """Print quick statistics about labeled dataset."""
    df = pd.read_csv(labeled_csv)
    
    print("DATASET STATISTICS")
    print(f"Total frames: {len(df):,}")
    print(f"Total sequences: {df['sequence_id'].nunique():,}")
    print(f"\nLabel distribution (by sequence):")
    
    label_counts = df.groupby('predicted_label')['sequence_id'].nunique()
    total = label_counts.sum()
    
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label:8s}: {count:6,} ({pct:5.1f}%)")
    
