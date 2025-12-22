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
        
        # UI Elements
        self.output_area = widgets.Output()
        self.msg_output = widgets.Output()
        
        # 1. Load Data
        self.df = pd.read_csv(csv_path)
        
        # 2. Normalize Data & Sort
        self._normalize_data()
        
        # State Data
        self.selected_seq_ids = []
        self.selected_sequences = [] # List of tuples: (seq_id, dataframe_group)
        
        self.removed_images = set()        # Committed removals
        self.pending_removals = set()      # Removals waiting for page turn
        self.label_changes = {}
        self.sequence_merges = {}
        self.undo_stack = deque(maxlen=50)
        
        # PERFORMANCE: Set to 1 because long sequences are heavy to render
        self.items_per_page = 1
        self.current_page = 0
        
        # Load state OR run fast filter
        if Path(self.autosave_file).exists():
            self._load_autosave()
            self._reconstruct_sequences_from_ids()
        else:
            self._filter_sequences_fast()
            self._reconstruct_sequences_from_ids()

        self.total_pages = (len(self.selected_sequences) + self.items_per_page - 1) // self.items_per_page
        
        self._setup_ui()

    def _normalize_data(self):
        """Adapts new CSV formats and Sorts by Length."""
        # Rename column if needed
        if 'sequence' in self.df.columns and 'sequence_id' not in self.df.columns:
            self.df.rename(columns={'sequence': 'sequence_id'}, inplace=True)
            
        # Ensure predicted_label exists
        if 'predicted_label' not in self.df.columns:
            self.df['predicted_label'] = 'none'
            
        # Ensure frame_id is numeric
        if 'frame_id' in self.df.columns:
            self.df['frame_id'] = pd.to_numeric(self.df['frame_id'], errors='coerce').fillna(0).astype(int)

        # SORTING: Longest sequences first
        print("Sorting dataset by sequence length (longest first)...")
        # We calculate length and map it back
        seq_lengths = self.df['sequence_id'].value_counts()
        self.df['seq_len'] = self.df['sequence_id'].map(seq_lengths)
        
        # Sort by Length (Desc), then ID, then Frame
        self.df.sort_values(by=['seq_len', 'sequence_id', 'frame_id'],
                          ascending=[False, True, True],
                          inplace=True)
        self.df.drop(columns=['seq_len'], inplace=True)
        print("Data normalized and sorted.")

    def _filter_sequences_fast(self):
        """
        Since data is already sorted by length in _normalize_data,
        just blindly take the top N sequence IDs.
        No iterating through groups to check for labels.
        """
        with self.msg_output:
            print("Selecting longest sequences...")
        
        # unique() preserves order of appearance.
        # Since DF is sorted by length, the first IDs are the longest ones.
        unique_ids = self.df['sequence_id'].unique()
        
        # Slice the top N
        self.selected_seq_ids = unique_ids[:self.max_sequences].tolist()
        
        print(f"Initialized with {len(self.selected_seq_ids)} sequences.")

    def _reconstruct_sequences_from_ids(self):
        target_ids = set(self.selected_seq_ids)
        subset_df = self.df[self.df['sequence_id'].isin(target_ids)]
        grouped = {k: v for k, v in subset_df.groupby('sequence_id')}
        
        self.selected_sequences = []
        for seq_id in self.selected_seq_ids:
            if seq_id in grouped:
                self.selected_sequences.append((seq_id, grouped[seq_id].sort_values('frame_id')))

    def _setup_ui(self):
        # --- Top Bar (Actions) ---
        self.btn_save = widgets.Button(description="Save Progress & CSV", button_style='success', icon='save')
        self.btn_undo = widgets.Button(description="Undo Last Action", button_style='warning', icon='undo')
        
        self.btn_save.on_click(self.save_final)
        self.btn_undo.on_click(self.undo_action)
        
        self.top_bar = widgets.HBox([self.btn_save, self.btn_undo], layout=widgets.Layout(margin='0px 0px 20px 0px'))

        # --- Bottom Bar (Navigation) ---
        self.btn_prev = widgets.Button(description="< Prev Sequence", icon='arrow-left', layout=widgets.Layout(width='150px'))
        self.btn_next = widgets.Button(description="Next Sequence >", icon='arrow-right', layout=widgets.Layout(width='150px'))
        self.lbl_page = widgets.Label(value=f"Sequence {self.current_page + 1} / {self.total_pages}")
        
        self.btn_prev.on_click(self.prev_page)
        self.btn_next.on_click(self.next_page)
        
        self.bottom_bar = widgets.HBox([self.btn_prev, self.lbl_page, self.btn_next], layout=widgets.Layout(justify_content='center', margin='20px 0px 0px 0px'))

    def display(self):
        display(self.top_bar)
        display(self.msg_output)
        display(self.output_area)
        display(self.bottom_bar)
        self.render_page()

    def render_page(self):
        self.output_area.clear_output(wait=True)
        
        start_idx = self.current_page * self.items_per_page
        end_idx = start_idx + self.items_per_page
        
        if start_idx >= len(self.selected_sequences):
            start_idx = 0
            self.current_page = 0
        
        page_seqs = self.selected_sequences[start_idx:end_idx]
        all_ids = self.selected_seq_ids
        
        with self.output_area:
            for seq_index, (seq_id, group) in enumerate(page_seqs):
                global_index = start_idx + seq_index
                target_merge = self.sequence_merges.get(seq_id, None)
                
                # --- Header ---
                title = widgets.HTML(f"<h3>Sequence: {seq_id}</h3>")
                stats = widgets.Label(value=f"Frames: {len(group)}")
                
                dd_merge = widgets.Combobox(
                    placeholder='Merge into ID...',
                    options=all_ids,
                    description='Merge:',
                    value=target_merge if target_merge else '',
                    ensure_option=False,
                    layout=widgets.Layout(width='300px')
                )
                dd_merge.observe(lambda c, sid=seq_id: self._on_merge_change(c, sid), names='value')
                
                if target_merge:
                    status_lbl = widgets.HTML(f"<b style='color:orange; margin-left:10px'>⚠ MERGED INTO: {target_merge}</b>")
                    container_style = "border: 2px solid orange; padding: 10px; margin-bottom: 20px; background-color: #fff3e0;"
                else:
                    status_lbl = widgets.HTML("")
                    container_style = "border: 1px solid #ccc; padding: 10px; margin-bottom: 20px;"

                btn_bulk_label = widgets.Dropdown(
                    options=['Set all...', 'left', 'right', 'hazard', 'none'],
                    value='Set all...',
                    layout=widgets.Layout(width='150px')
                )
                btn_bulk_label.observe(lambda c, g=group: self._on_bulk_label(c, g), names='value')

                header_box = widgets.HBox([title, stats, dd_merge, status_lbl, btn_bulk_label])
                
                # --- Images ---
                image_cards = []
                for idx, row in group.iterrows():
                    crop_path = row['crop_path']
                    
                    if crop_path in self.removed_images:
                        continue
                        
                    is_pending = crop_path in self.pending_removals
                    current_lbl = self.label_changes.get(crop_path, row['predicted_label'])
                    
                    # Smart Path Handling
                    try:
                        p = Path(crop_path)
                        if p.exists():
                            full_path = p
                        else:
                            rel_path = crop_path.lstrip('/')
                            full_path = self.base_path / rel_path
                            
                        with Image.open(full_path) as pil_img:
                            pil_img.thumbnail((180, 180))
                            b_io = io.BytesIO()
                            pil_img.save(b_io, format='JPEG')
                            img_widget = widgets.Image(value=b_io.getvalue(), format='jpg', width=180, height=135)
                    except Exception as e:
                        img_widget = widgets.Button(description="Missing", disabled=True, layout=widgets.Layout(width='180px', height='135px'))
                    
                    # Controls
                    dd_label = widgets.Dropdown(
                        options=['left', 'right', 'hazard', 'none'],
                        value=current_lbl,
                        layout=widgets.Layout(width='90px')
                    )
                    dd_label.observe(lambda c, cp=crop_path: self._on_single_label(c, cp), names='value')
                    
                    btn_remove = widgets.Button(
                        description='Pending...' if is_pending else 'X',
                        button_style='' if is_pending else 'danger',
                        disabled=is_pending,
                        layout=widgets.Layout(width='40px')
                    )
                    btn_remove.on_click(lambda b, cp=crop_path: self._on_remove_click(b, cp))
                    
                    btn_split = widgets.Button(
                        description='Split',
                        button_style='info',
                        icon='cut',
                        layout=widgets.Layout(width='70px')
                    )
                    btn_split.on_click(lambda b, g_idx=global_index, split_val=row['frame_id']: self._on_split_sequence(g_idx, split_val))

                    ctrl_row = widgets.HBox([dd_label, btn_remove, btn_split])
                    card = widgets.VBox([img_widget, ctrl_row], layout=widgets.Layout(align_items='center', margin='5px', border='1px solid #eee'))
                    image_cards.append(card)
                
                grid = widgets.GridBox(
                    children=image_cards,
                    layout=widgets.Layout(
                        grid_template_columns='repeat(4, 1fr)',
                        grid_gap='10px',
                        width='100%'
                    )
                )
                
                styled_box = widgets.VBox([widgets.HTML(f"<div style='{container_style}'>"), widgets.VBox([header_box, grid]), widgets.HTML("</div>")])
                display(styled_box)

    # --- Interaction Handlers ---

    def _on_remove_click(self, b, crop_path):
        self.pending_removals.add(crop_path)
        b.description = "Pending"
        b.button_style = ""
        b.disabled = True
        
    def _commit_removals(self):
        if not self.pending_removals:
            return
        count = len(self.pending_removals)
        self.removed_images.update(self.pending_removals)
        self.undo_stack.append({
            'type': 'batch_remove',
            'paths': list(self.pending_removals)
        })
        self.pending_removals.clear()
        self._autosave()

    def _on_split_sequence(self, global_seq_index, split_frame_id):
        seq_id, df_group = self.selected_sequences[global_seq_index]
        df_top = df_group[df_group['frame_id'] < split_frame_id]
        df_bottom = df_group[df_group['frame_id'] >= split_frame_id]
        
        if len(df_top) == 0 or len(df_bottom) == 0:
            with self.msg_output:
                print("Cannot split at edge.")
            return

        new_id = f"{seq_id}_split_{int(datetime.now().timestamp())}"
        
        self.selected_sequences[global_seq_index] = (seq_id, df_top)
        self.selected_sequences.insert(global_seq_index + 1, (new_id, df_bottom))
        self.selected_seq_ids.insert(global_seq_index + 1, new_id)
        
        self.total_pages = (len(self.selected_sequences) + self.items_per_page - 1) // self.items_per_page
        self.lbl_page.value = f"Sequence {self.current_page + 1} / {self.total_pages}"
        
        self._autosave()
        self.render_page()
        with self.msg_output:
            clear_output(wait=True)
            print(f"Split sequence {seq_id}. New sequence {new_id} created.")

    def _on_single_label(self, change, crop_path):
        if change['type'] == 'change' and change['name'] == 'value':
            old_val = self.label_changes.get(crop_path, "original")
            self.undo_stack.append({'type': 'label', 'path': crop_path, 'prev': old_val})
            self.label_changes[crop_path] = change['new']
            self._autosave()

    def _on_bulk_label(self, change, group_df):
        if change['type'] == 'change' and change['name'] == 'value':
            new_val = change['new']
            if new_val == 'Set all...': return
            
            paths = []
            prevs = []
            for _, row in group_df.iterrows():
                cp = row['crop_path']
                if cp not in self.removed_images and cp not in self.pending_removals:
                    paths.append(cp)
                    prevs.append(self.label_changes.get(cp, "original"))
                    self.label_changes[cp] = new_val
            
            self.undo_stack.append({'type': 'bulk_label', 'paths': paths, 'prevs': prevs})
            self._autosave()
            self.render_page()

    def _on_merge_change(self, change, seq_id):
        if change['type'] == 'change' and change['name'] == 'value':
            val = change['new']
            if not val:
                if seq_id in self.sequence_merges: del self.sequence_merges[seq_id]
            elif val == seq_id:
                return
            else:
                self.sequence_merges[seq_id] = val
            self._autosave()
            self.render_page()

    def prev_page(self, b):
        self._commit_removals()
        if self.current_page > 0:
            self.current_page -= 1
            self.lbl_page.value = f"Sequence {self.current_page + 1} / {self.total_pages}"
            self._autosave()
            self.render_page()

    def next_page(self, b):
        self._commit_removals()
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.lbl_page.value = f"Sequence {self.current_page + 1} / {self.total_pages}"
            self._autosave()
            self.render_page()

    def undo_action(self, b):
        if not self.undo_stack: return
        action = self.undo_stack.pop()
        
        if action['type'] == 'label':
            if action['prev'] == "original":
                del self.label_changes[action['path']]
            else:
                self.label_changes[action['path']] = action['prev']
        elif action['type'] == 'bulk_label':
            for p, prev in zip(action['paths'], action['prevs']):
                if prev == "original":
                    if p in self.label_changes: del self.label_changes[p]
                else:
                    self.label_changes[p] = prev
        elif action['type'] == 'batch_remove':
            for p in action['paths']:
                self.removed_images.discard(p)
        
        self._autosave()
        self.render_page()

    def _autosave(self):
        state = {
            'removed': list(self.removed_images),
            'labels': self.label_changes,
            'merges': self.sequence_merges,
            'selected_seq_ids': self.selected_seq_ids,
            'current_page': self.current_page
        }
        with open(self.autosave_file, 'w') as f:
            json.dump(state, f)

    def _load_autosave(self):
        with open(self.autosave_file, 'r') as f:
            state = json.load(f)
            self.removed_images = set(state.get('removed', []))
            self.label_changes = state.get('labels', {})
            self.sequence_merges = state.get('merges', {})
            self.selected_seq_ids = state.get('selected_seq_ids', [])
            self.current_page = state.get('current_page', 0)
            with self.msg_output:
                print("Loaded previous session state.")

    def save_final(self, b):
        self._commit_removals()
        with self.msg_output:
            print("Preparing export...")
        
        final_df = self.df[~self.df['crop_path'].isin(self.removed_images)].copy()
        
        for cp, lbl in self.label_changes.items():
            final_df.loc[final_df['crop_path'] == cp, 'predicted_label'] = lbl
            
        for old_s, new_s in self.sequence_merges.items():
            final_df.loc[final_df['sequence_id'] == old_s, 'sequence_id'] = new_s
            
        path_to_seq_map = {}
        for seq_id, group in self.selected_sequences:
            for _, row in group.iterrows():
                path_to_seq_map[row['crop_path']] = seq_id
        
        final_df['sequence_id'] = final_df['crop_path'].map(path_to_seq_map).fillna(final_df['sequence_id'])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"cleaned_data_{timestamp}.csv"
        final_df.to_csv(out_name, index=False)
        
        with self.msg_output:
            clear_output(wait=True)
            print(f"Saved {out_name} successfully.")
