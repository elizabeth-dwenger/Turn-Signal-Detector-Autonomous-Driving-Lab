import os
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import torch

# pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

DET_CSV   = "/gpfs/helios/home/dwenger/front_detections_with_crop_path.csv"
TRACK_OUT = "/gpfs/helios/home/dwenger/front_tracks_deepsort.csv"

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

def load_frame_cache(img_path, cache={}):
    """Load frame with caching to avoid reading same frame multiple times"""
    if img_path not in cache:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        cache[img_path] = img
    return cache[img_path]

def df_to_deepsort_detections(df_frame):
    """
    Convert DataFrame detections to DeepSORT format
    Returns detections list and mapping to original dataframe indices
    """
    detections = []
    det_indices = []
    
    for idx, row in df_frame.iterrows():
        # Keep coordinates as floats for precision
        x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
        w = x2 - x1
        h = y2 - y1
        
        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue
            
        conf = float(row['score'])
        class_id = int(row['class_id'])
        
        # DeepSORT format: ([left, top, width, height], confidence, detection_class)
        detections.append(([x1, y1, w, h], conf, class_id))
        det_indices.append(idx)
    
    return detections, det_indices

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def main():
    df = pd.read_csv(DET_CSV)
    print(f"Loaded {len(df)} detections from {DET_CSV}")
    
    # Use correct column name
    sequence_col = 'sequence' if 'sequence' in df.columns else 'sequence_id'
    print(f"Using sequence column: '{sequence_col}'")
    
    df["frame_id_int"] = df["frame_id"].astype(int)
    df = df.sort_values([sequence_col, "frame_id_int"]).reset_index(drop=True)
    
    # Initialize tracking columns
    df["track_id"] = -1
    df["track_confidence"] = 0.0
    
    # Track global sequence counter for composite IDs
    sequence_counter = {}
    
    for seq, df_seq in df.groupby(sequence_col):
        print(f"\nProcessing sequence: {seq}")
        print(f"  Detections: {len(df_seq)}")
        print(f"  Frames: {df_seq['frame_id_int'].nunique()}")
        print(f"  Frame range: {df_seq['frame_id_int'].min()} - {df_seq['frame_id_int'].max()}")
        
        # Initialize DeepSORT tracker
        print(f"  Initializing DeepSORT tracker...")
        try:
            tracker = DeepSort(
                max_age=30,                    # Frames to keep track alive without updates
                n_init=3,                      # Frames needed to confirm a track
                nms_max_overlap=0.7,           # NMS IoU threshold
                max_cosine_distance=0.3,       # Appearance similarity (lower = stricter)
                nn_budget=100,                 # Max samples per class for appearance
                embedder="mobilenet",          # Feature extractor
                half=torch.cuda.is_available(),  # Use FP16 only if GPU available
                embedder_gpu=torch.cuda.is_available(),
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )
            print(f"  ✓ DeepSORT initialized {'with GPU' if torch.cuda.is_available() else 'on CPU'}")
        except Exception as e:
            print(f"  ⚠ Error initializing DeepSORT: {e}")
            print(f"  Trying without half precision...")
            tracker = DeepSort(
                max_age=30,
                n_init=3,
                nms_max_overlap=0.7,
                max_cosine_distance=0.3,
                nn_budget=100,
                embedder="mobilenet",
                half=False,
                embedder_gpu=torch.cuda.is_available(),
                polygon=False
            )
        
        frame_cache = {}
        frame_count = 0
        
        # Process frames in order
        sorted_frames = sorted(df_seq['frame_id_int'].unique())
        
        for frame_id in tqdm(sorted_frames, desc=f"  Tracking"):
            frame_count += 1
            df_frame = df_seq[df_seq['frame_id_int'] == frame_id]
            
            # Load the full frame image
            img_path = df_frame.iloc[0]["frame_path"]
            
            try:
                frame = load_frame_cache(img_path, frame_cache)
            except FileNotFoundError as e:
                print(f"\n[WARN] {e}")
                continue
            
            # Convert detections to DeepSORT format
            detections, det_indices = df_to_deepsort_detections(df_frame)
            
            if len(detections) == 0:
                # Still update tracker with empty detections to age out old tracks
                tracker.update_tracks([], frame=frame)
                continue
            
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Match tracks back to detections
            confirmed_tracks = [t for t in tracks if t.is_confirmed()]
            
            for det_idx, orig_idx in enumerate(det_indices):
                row = df.loc[orig_idx]
                det_box = [row['x1'], row['y1'], row['x2'], row['y2']]
                
                # Find best matching track
                best_track_id = -1
                best_iou = 0
                
                for track in confirmed_tracks:
                    ltrb = track.to_ltrb()
                    track_box = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                    
                    iou = calculate_iou(track_box, det_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track.track_id
                
                if best_iou >= 0.8:
                    df.at[orig_idx, "track_id"] = best_track_id
                    df.at[orig_idx, "track_confidence"] = best_iou
            
            # Clear frame cache more aggressively for long sequences
            if frame_count % 20 == 0:
                frame_cache.clear()
        
        # Clear cache after each sequence
        frame_cache.clear()
        
        # Statistics for this sequence
        seq_tracked = df_seq[df['track_id'] != -1]
        if len(seq_tracked) > 0:
            n_tracks = seq_tracked['track_id'].nunique()
            print(f"  ✓ Assigned {len(seq_tracked)}/{len(df_seq)} detections to {n_tracks} tracks")
    
    # Assignment results
    print(f"\n{'='*60}")
    print(f"TRACKING RESULTS")
    print(f"{'='*60}")
    print(f"Total detections: {len(df):,}")
    print(f"Assigned to tracks: {(df['track_id'] != -1).sum():,}")
    print(f"Unassigned: {(df['track_id'] == -1).sum():,}")
    print(f"Assignment rate: {(df['track_id'] != -1).sum()/len(df)*100:.1f}%")
    
    # Keep only successfully tracked detections
    tracked_df = df[df['track_id'] != -1].copy()
    
    # Create composite sequence_id from sequence + track_id
    # This ensures each track is globally unique
    print(f"\n{'='*60}")
    print(f"CREATING COMPOSITE TRACK IDs")
    print(f"{'='*60}")
    
    tracked_df['original_track_id'] = tracked_df['track_id']
    
    # Create unique sequence_id for each (sequence, track_id) pair
    composite_ids = []
    for _, row in tracked_df.iterrows():
        seq = row[sequence_col]
        track = row['track_id']
        composite_id = f"{seq}__track_{track}"
        composite_ids.append(composite_id)
    
    tracked_df['sequence_id'] = composite_ids
    
    print(f"Original sequences: {tracked_df[sequence_col].nunique()}")
    print(f"Original tracks: {tracked_df['original_track_id'].nunique()}")
    print(f"Composite sequence_ids: {tracked_df['sequence_id'].nunique()}")
    
    # Filter out tracks that are too short (likely noise)
    print(f"\n{'='*60}")
    print(f"FILTERING SHORT TRACKS")
    print(f"{'='*60}")
    
    track_lengths = tracked_df.groupby('sequence_id').size()
    print(f"Track length distribution:")
    print(f"  Min: {track_lengths.min()}")
    print(f"  Max: {track_lengths.max()}")
    print(f"  Mean: {track_lengths.mean():.1f}")
    print(f"  Median: {track_lengths.median():.1f}")
    
    min_track_length = 3
    valid_tracks = track_lengths[track_lengths >= min_track_length].index
    tracked_df = tracked_df[tracked_df['sequence_id'].isin(valid_tracks)].copy()
    
    print(f"\nAfter filtering tracks < {min_track_length} frames:")
    print(f"  Remaining detections: {len(tracked_df):,}")
    print(f"  Remaining tracks: {tracked_df['sequence_id'].nunique():,}")
    
    # Save results
    tracked_df.to_csv(TRACK_OUT, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Wrote {len(tracked_df):,} rows -> {TRACK_OUT}")
    print(f"{'='*60}")
    
    # Final statistics
    if len(tracked_df) > 0:
        final_track_lengths = tracked_df.groupby('sequence_id').size()
        
        print(f"\nFINAL TRACK STATISTICS:")
        print(f"  Total unique tracks: {len(final_track_lengths):,}")
        print(f"  Mean frames per track: {final_track_lengths.mean():.1f}")
        print(f"  Median frames per track: {final_track_lengths.median():.1f}")
        print(f"  Longest track: {final_track_lengths.max():,} frames")
        print(f"  Shortest track: {final_track_lengths.min():,} frames")
        print(f"  Mean track confidence (IoU): {tracked_df['track_confidence'].mean():.3f}")
        
        # Show sample of longest tracks
        print(f"\nTop 5 longest tracks:")
        for i, (seq_id, length) in enumerate(final_track_lengths.nlargest(5).items(), 1):
            print(f"  {i}. {seq_id[:80]:80s} : {length:,} frames")
    else:
        print("\n[WARNING] No tracks produced after filtering!")

if __name__ == "__main__":
    main()
