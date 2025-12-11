import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import json
from datetime import datetime
from collections import deque
import io
from PIL import Image

class SequenceCleaningTool:
    def __init__(self, csv_path, base_path="", max_sequences=5000, autosave_file="autosave_state.json"):
        self.csv_path = csv_path
        self.base_path = Path(base_path)
        self.max_sequences = max_sequences
        self.autosave_file = autosave_file
        
        # Load and Preprocess Data
        self.df = pd.read_csv(csv_path)
        self._filter_sequences()
        
        # State Management
        # removed_images: set of crop_paths
        self.removed_images = set()
        # label_changes: dict {crop_path: new_label}
        self.label_changes = {}
        # sequence_merges: dict {original_seq_id: target_seq_id}
        self.sequence_merges = {}
        # Undo stack: list of dicts {'action': str, 'data': ...}
        self.undo_stack = deque(maxlen=50)
        
        # Pagination
        self.items_per_page = 5
        self.current_page = 0
        self.total_pages = (len(self.selected_sequences) + self.items_per_page - 1) // self.items_per_page
        
        # UI Elements
        self.output_area = widgets.Output()
        self.msg_output = widgets.Output()
        self._setup_ui()
        
        # Load previous state if exists
        self._load_autosave()

    def _filter_sequences(self):
        """
        Rule A: Include all sequences with left/right/hazard.
        Rule B: Fill remaining up to max_sequences with longest non-signal sequences.
        """
        print("Filtering sequences based on rules...")
        
        # Ensure predicted_label is string
        self.df['predicted_label'] = self.df['predicted_label'].astype(str)
        
        # Group
        groups = self.df.groupby('sequence_id')
        
        priority_seqs = []
        other_seqs = []
        
        # Analyze groups (this might take a moment)
        seq_metrics = []
        
        for seq_id, group in groups:
            labels = group['predicted_label'].unique()
            has_signal = any(l in ['left', 'right', 'hazard'] for l in labels)
            length = len(group)
            
            seq_metrics.append({
                'sequence_id': seq_id,
                'has_signal': has_signal,
                'length': length
            })
            
        metrics_df = pd.DataFrame(seq_metrics)
        
        # Rule A
        rule_a_df = metrics_df[metrics_df['has_signal'] == True]
        selected_ids = rule_a_df['sequence_id'].tolist()
        
        # Rule B
        remaining_slots = self.max_sequences - len(selected_ids)
        if remaining_slots > 0:
            rule_b_df = metrics_df[metrics_df['has_signal'] == False].sort_values('length', ascending=False)
            top_b = rule_b_df.head(remaining_slots)['sequence_id'].tolist()
            selected_ids.extend(top_b)
            
        self.selected_seq_ids = selected_ids
        
        # Filter main DF to just these sequences for performance in the tool
        self.selected_sequences = [
            (seq_id, self.df[self.df['sequence_id'] == seq_id].sort_values('frame_id'))
            for seq_id in selected_ids
        ]
        
        print(f"Selected {len(self.selected_sequences)} sequences (Rule A: {len(rule_a_df)}, Rule B: {len(selected_ids)-len(rule_a_df)})")

    def _setup_ui(self):
        # Header Controls
        self.btn_prev = widgets.Button(description="< Prev Page", icon='arrow-left')
        self.btn_next = widgets.Button(description="Next Page >", icon='arrow-right')
        self.lbl_page = widgets.Label(value=f"Page 1 / {self.total_pages}")
        
        self.btn_save = widgets.Button(description="Save Final CSV", button_style='success', icon='save')
        self.btn_undo = widgets.Button(description="Undo", button_style='warning', icon='undo')
        
        self.btn_prev.on_click(self.prev_page)
        self.btn_next.on_click(self.next_page)
        self.btn_save.on_click(self.save_final)
        self.btn_undo.on_click(self.undo_action)
        
        self.header = widgets.VBox([
            widgets.HBox([self.btn_prev, self.lbl_page, self.btn_next]),
            widgets.HBox([self.btn_save, self.btn_undo])
        ])

    def display(self):
        display(self.header)
        display(self.msg_output)
        display(self.output_area)
        self.render_page()

    def render_page(self):
        self.output_area.clear_output(wait=True)
        
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_seqs = self.selected_sequences[start_idx:end_idx]
        
        seq_widgets = []
        
        # Options for merge dropdown (all available sequence IDs)
        # For performance, we might limit this list or allow free text,
        # but dropdown is requested.
        all_ids = self.selected_seq_ids
        
        with self.output_area:
            for seq_id, group in page_seqs:
                # Check if merged
                target_merge = self.sequence_merges.get(seq_id, None)
                
                # --- Sequence Header ---
                title = widgets.HTML(f"<h3>Sequence: {seq_id}</h3>")
                stats = widgets.Label(value=f"Frames: {len(group)}")
                
                # Merge Control
                dd_merge = widgets.Combobox(
                    placeholder='Type target Sequence ID...',
                    options=all_ids,
                    description='Merge into:',
                    value=target_merge if target_merge else '',
                    ensure_option=False, # Allow typing new IDs
                    layout=widgets.Layout(width='400px')
                )
                dd_merge.observe(lambda c, sid=seq_id: self._on_merge_change(c, sid), names='value')
                
                if target_merge:
                    # Visual indicator of merge
                    container_style = "border: 2px solid orange; padding: 10px; margin-bottom: 20px; background-color: #fff3e0;"
                    status_lbl = widgets.HTML(f"<b style='color:orange'>MERGED INTO: {target_merge}</b>")
                else:
                    container_style = "border: 1px solid #ccc; padding: 10px; margin-bottom: 20px;"
                    status_lbl = widgets.HTML("")

                # Bulk Label
                btn_bulk_label = widgets.Dropdown(
                    options=['Select to apply all...', 'left', 'right', 'hazard', 'none'],
                    value='Select to apply all...',
                    layout=widgets.Layout(width='200px')
                )
                btn_bulk_label.observe(lambda c, g=group: self._on_bulk_label(c, g), names='value')

                header_box = widgets.HBox([title, stats, dd_merge, status_lbl, btn_bulk_label])
                
                # --- Image Grid ---
                image_cards = []
                for _, row in group.iterrows():
                    crop_path = row['crop_path']
                    img_id = row.get('track_id', '?') # Just a visual ID
                    
                    # Check if removed
                    is_removed = crop_path in self.removed_images
                    if is_removed:
                        continue # Don't render if removed (or render grayed out)
                    
                    # Check current label
                    current_lbl = self.label_changes.get(crop_path, row['predicted_label'])
                    
                    # Image Widget
                    # Load image efficiently
                    full_path = self.base_path / crop_path.lstrip('/')
                    try:
                        # Resize for thumbnail to save memory
                        with Image.open(full_path) as pil_img:
                            pil_img.thumbnail((200, 200))
                            b_io = io.BytesIO()
                            pil_img.save(b_io, format='JPEG')
                            img_widget = widgets.Image(value=b_io.getvalue(), format='jpg', width=200, height=150)
                    except Exception:
                        img_widget = widgets.Button(description="Image Not Found", disabled=True, layout=widgets.Layout(width='200px', height='150px'))
                    
                    # Controls
                    dd_label = widgets.Dropdown(
                        options=['left', 'right', 'hazard', 'none'],
                        value=current_lbl,
                        layout=widgets.Layout(width='100px')
                    )
                    dd_label.observe(lambda c, cp=crop_path: self._on_single_label(c, cp), names='value')
                    
                    btn_remove = widgets.Button(
                        description='X',
                        button_style='danger',
                        layout=widgets.Layout(width='40px')
                    )
                    btn_remove.on_click(lambda b, cp=crop_path: self._on_remove_image(b, cp))
                    
                    card = widgets.VBox([
                        img_widget,
                        widgets.HBox([dd_label, btn_remove])
                    ], layout=widgets.Layout(align_items='center', margin='5px'))
                    
                    image_cards.append(card)
                
                # CSS Grid Layout: 4 columns
                grid = widgets.GridBox(
                    children=image_cards,
                    layout=widgets.Layout(
                        grid_template_columns='repeat(4, 1fr)',
                        grid_gap='20px',
                        width='100%'
                    )
                )
                
                seq_container = widgets.VBox([header_box, grid])
                # Apply inline style wrapper
                seq_wrapper = widgets.VBox([seq_container], layout=widgets.Layout(border='1px solid #ccc', margin='10px 0px 30px 0px', padding='10px'))
                
                display(seq_wrapper)

    # --- Actions ---

    def _on_single_label(self, change, crop_path):
        if change['type'] != 'change' or change['name'] != 'value': return
        new_val = change['new']
        
        # Save undo
        old_val = self.label_changes.get(crop_path, "original") # Simple sentinel
        self.undo_stack.append({'type': 'label', 'path': crop_path, 'prev': old_val})
        
        self.label_changes[crop_path] = new_val
        self._autosave()

    def _on_bulk_label(self, change, group_df):
        if change['type'] != 'change' or change['name'] != 'value': return
        new_val = change['new']
        if new_val == 'Select to apply all...': return
        
        # Apply to all in group
        paths = []
        prev_states = []
        
        for _, row in group_df.iterrows():
            cp = row['crop_path']
            if cp not in self.removed_images:
                paths.append(cp)
                prev_states.append(self.label_changes.get(cp, "original"))
                self.label_changes[cp] = new_val
                
        # Group undo
        self.undo_stack.append({'type': 'bulk_label', 'paths': paths, 'prevs': prev_states})
        self._autosave()
        
        # Refresh to show changes
        self.render_page()

    def _on_remove_image(self, b, crop_path):
        self.removed_images.add(crop_path)
        self.undo_stack.append({'type': 'remove', 'path': crop_path})
        self._autosave()
        self.render_page() # Re-render to hide

    def _on_merge_change(self, change, seq_id):
        if change['type'] != 'change' or change['name'] != 'value': return
        target_id = change['new']
        
        if not target_id:
            # Cleared
            if seq_id in self.sequence_merges:
                del self.sequence_merges[seq_id]
        elif target_id == seq_id:
            with self.msg_output:
                print(f"Warning: Cannot merge sequence {seq_id} into itself.")
            return
        else:
            self.sequence_merges[seq_id] = target_id
            
        self._autosave()
        self.render_page() # Update visual indicator

    def undo_action(self, b):
        if not self.undo_stack:
            with self.msg_output:
                clear_output(wait=True)
                print("Nothing to undo.")
            return
            
        action = self.undo_stack.pop()
        
        if action['type'] == 'remove':
            self.removed_images.remove(action['path'])
        elif action['type'] == 'label':
            if action['prev'] == "original":
                del self.label_changes[action['path']]
            else:
                self.label_changes[action['path']] = action['prev']
        elif action['type'] == 'bulk_label':
            for path, prev in zip(action['paths'], action['prevs']):
                if prev == "original":
                    if path in self.label_changes: del self.label_changes[path]
                else:
                    self.label_changes[path] = prev
                    
        self._autosave()
        self.render_page()
        with self.msg_output:
            clear_output(wait=True)
            print(f"Undid action: {action['type']}")

    # --- Persistence ---

    def _autosave(self):
        state = {
            'removed': list(self.removed_images),
            'labels': self.label_changes,
            'merges': self.sequence_merges
        }
        with open(self.autosave_file, 'w') as f:
            json.dump(state, f)
            
    def _load_autosave(self):
        if Path(self.autosave_file).exists():
            with open(self.autosave_file, 'r') as f:
                state = json.load(f)
                self.removed_images = set(state.get('removed', []))
                self.label_changes = state.get('labels', {})
                self.sequence_merges = state.get('merges', {})
                print("Loaded autosave state.")

    def prev_page(self, b):
        if self.current_page > 0:
            self.current_page -= 1
            self.lbl_page.value = f"Page {self.current_page + 1} / {self.total_pages}"
            self.render_page()

    def next_page(self, b):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.lbl_page.value = f"Page {self.current_page + 1} / {self.total_pages}"
            self.render_page()

    def save_final(self, b):
        """Export updated labels, removed images, and merged mappings."""
        with self.msg_output:
            print("Saving...")
            
        # 1. Apply removals
        final_df = self.df[~self.df['crop_path'].isin(self.removed_images)].copy()
        
        # 2. Apply Label Changes
        for crop_path, new_lbl in self.label_changes.items():
            final_df.loc[final_df['crop_path'] == crop_path, 'predicted_label'] = new_lbl
            
        # 3. Apply Merges
        # If A merges into B, any row with sequence_id A becomes B
        for old_seq, new_seq in self.sequence_merges.items():
            final_df.loc[final_df['sequence_id'] == old_seq, 'sequence_id'] = new_seq
            
        # 4. Check for empty sequences
        final_counts = final_df.groupby('sequence_id').size()
        empty_seqs = final_counts[final_counts == 0].index.tolist()
        
        # Generate Filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"final_dataset_{timestamp}.csv"
        final_df.to_csv(out_name, index=False)
        
        # Save Metadata log
        meta = {
            'removed_count': len(self.removed_images),
            'merged_count': len(self.sequence_merges),
            'label_changes_count': len(self.label_changes),
            'empty_sequences_created': empty_seqs
        }
        with open(f"final_dataset_{timestamp}_log.json", 'w') as f:
            json.dump(meta, f, indent=2)
            
        with self.msg_output:
            clear_output(wait=True)
            print(f"Saved {out_name}")
            print(f"Removed {len(self.removed_images)} images")
            print(f"Merged {len(self.sequence_merges)} sequences")
            if empty_seqs:
                print(f"Warning: The following sequences became empty: {empty_seqs}")
