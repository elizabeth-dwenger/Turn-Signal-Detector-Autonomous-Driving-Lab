import os
import pandas as pd
from tqdm import tqdm
import supervision as sv
import numpy as np

DET_CSV   = "/gpfs/helios/home/dwenger/detections_with_crop_path.csv"
TRACK_OUT = "/gpfs/helios/home/dwenger/tracks_with_crops_tuned_01.csv"

def df_to_detections(df_frame):
    xyxy = df_frame[["x1","y1","x2","y2"]].to_numpy()
    scores = df_frame["score"].to_numpy()
    classes = df_frame["class_id"].to_numpy()
    det = sv.Detections(xyxy=xyxy)
    det.confidence = scores
    det.class_id = classes
    return det

def filter_tracks_by_size_consistency(tracks_df, size_tolerance=0.3):
    print(f"\n[INFO] Filtering tracks by size consistency (tolerance={size_tolerance})...")
    
    # Calculate bbox sizes
    tracks_df['bbox_w'] = tracks_df['x2'] - tracks_df['x1']
    tracks_df['bbox_h'] = tracks_df['y2'] - tracks_df['y1']
    tracks_df['bbox_area'] = tracks_df['bbox_w'] * tracks_df['bbox_h']
    
    filtered_rows = []
    
    for (seq, tid), group in tracks_df.groupby(['sequence', 'track_id']):
        if len(group) < 2:
            # Keep single-frame tracks as-is
            filtered_rows.append(group)
            continue
        
        # Calculate median size for this track
        median_w = group['bbox_w'].median()
        median_h = group['bbox_h'].median()
        median_area = group['bbox_area'].median()
        
        # Filter detections that deviate too much from median
        valid_mask = (
            (np.abs(group['bbox_w'] - median_w) / median_w <= size_tolerance) &
            (np.abs(group['bbox_h'] - median_h) / median_h <= size_tolerance)
        )
        
        filtered_group = group[valid_mask]
        
        if len(filtered_group) < len(group):
            print(f"  Track {tid} in {seq}: kept {len(filtered_group)}/{len(group)} frames")
        
        if len(filtered_group) > 0:
            filtered_rows.append(filtered_group)
    
    result = pd.concat(filtered_rows, ignore_index=True)
    print(f"[INFO] Filtered: {len(tracks_df)} → {len(result)} rows ({len(tracks_df)-len(result)} removed)")
    
    return result.drop(columns=['bbox_w', 'bbox_h', 'bbox_area'])


def main():
    df = pd.read_csv(DET_CSV)
    df["frame_id_int"] = df["frame_id"].astype(int)
    df = df.sort_values(["sequence", "frame_id_int"]).reset_index(drop=True)
    
    all_tracks = []
    
    for seq, df_seq in df.groupby("sequence"):
        print(f"Processing sequence: {seq}")
        
        # TUNED PARAMETERS - More strict matching
        tracker = sv.ByteTrack(
            track_activation_threshold=0.6,      # Higher: only track high-confidence detections (was 0.5)
            lost_track_buffer=15,                # Lower: drop tracks faster if lost (was 30)
            minimum_matching_threshold=0.85,     # Higher: require better IoU match (was 0.8)
            frame_rate=30
        )
        
        for frame_id, df_frame in df_seq.groupby("frame_id_int"):
            dets = df_to_detections(df_frame)
            
            tracks = tracker.update_with_detections(detections=dets)
            
            width = int(df_frame.iloc[0]["width"])
            height = int(df_frame.iloc[0]["height"])
            
            for i in range(len(tracks)):
                x1, y1, x2, y2 = tracks.xyxy[i].astype(int).tolist()
                tid = int(tracks.tracker_id[i])
                conf = float(tracks.confidence[i])
                cid = int(tracks.class_id[i])
                
                crop_path = df_frame.iloc[i]["crop_path"]
                img_path = df_frame.iloc[i]["img_path"]
                frame_id_str = df_frame.iloc[i]["frame_id"]
                
                all_tracks.append({
                    "sequence": seq,
                    "frame_id": frame_id_str,
                    "track_id": tid,
                    "class_id": cid,
                    "score": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop_path": crop_path,
                    "img_path": img_path,
                    "width": width,
                    "height": height
                })
    
    tracks_df = pd.DataFrame(all_tracks)
    
    # Apply size consistency filter
    tracks_df = filter_tracks_by_size_consistency(tracks_df, size_tolerance=0.3)
    
    # Remove very short tracks
    track_lengths = tracks_df.groupby(['sequence', 'track_id']).size()
    valid_tracks = track_lengths[track_lengths >= 3].index  # At least 3 frames
    tracks_df = tracks_df.set_index(['sequence', 'track_id']).loc[valid_tracks].reset_index()
    
    tracks_df.to_csv(TRACK_OUT, index=False)
    print(f"\n[DONE] wrote {len(tracks_df)} rows → {TRACK_OUT}")
    
    # Print statistics
    final_track_lengths = tracks_df.groupby(['sequence', 'track_id']).size()
    print(f"\nTrack statistics:")
    print(f"  Total tracks: {len(final_track_lengths)}")
    print(f"  Mean length: {final_track_lengths.mean():.1f} frames")
    print(f"  Median length: {final_track_lengths.median():.1f} frames")
    print(f"  Max length: {final_track_lengths.max()} frames")

if __name__ == "__main__":
    main()
