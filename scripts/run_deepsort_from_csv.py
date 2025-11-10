import os
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import torch

# pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort

DET_CSV   = "/gpfs/helios/home/dwenger/detections_with_crop_path.csv"
TRACK_OUT = "/gpfs/helios/home/dwenger/tracks_deepsort.csv"

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

def load_frame_cache(img_path, cache={}):
    # load frame with caching to avoid reading same frame multiple times
    if img_path not in cache:
        cache[img_path] = cv2.imread(img_path)
        if cache[img_path] is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
    return cache[img_path]

def df_to_deepsort_detections(df_frame, frame_img):
    # Convert DataFrame detections to DeepSORT format
    # DeepSORT expects: ([x1, y1, w, h], confidence, class_id)
    detections = []
    for _, row in df_frame.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        w = x2 - x1
        h = y2 - y1
        conf = float(row['score'])
        class_id = int(row['class_id'])
        
        # DeepSORT format: ([left, top, width, height], confidence, detection_class)
        detections.append(([x1, y1, w, h], conf, class_id))
    
    return detections

def main():
    df = pd.read_csv(DET_CSV)
    df["frame_id_int"] = df["frame_id"].astype(int)
    df = df.sort_values(["sequence", "frame_id_int"]).reset_index(drop=True)
    
    all_tracks = []
    
    for seq, df_seq in df.groupby("sequence"):
        print(f"\nProcessing sequence: {seq}")
        
        # DeepSORT tracker with GPU support
        print(f"  Initializing DeepSORT tracker...")
        tracker = DeepSort(
            max_age=30,                    # Frames to keep track alive without updates
            n_init=3,                      # Frames needed to confirm a track
            nms_max_overlap=0.7,           # NMS IoU threshold
            max_cosine_distance=0.3,       # Appearance similarity (lower = stricter)
            nn_budget=100,                 # Max samples per class for appearance
            embedder="mobilenet",          # Feature extractor: 'mobilenet', 'torchreid', or 'clip'
            half=True,                     # Use FP16 for speed (requires GPU)
            embedder_gpu=True,             # Use GPU for feature extraction
            embedder_model_name=None,      # Use default model
            embedder_wts=None,             # Use default weights
            polygon=False,
            today=None
        )
        print(f"DeepSORT initialized with GPU support")
        
        frame_cache = {}
        
        for frame_id, df_frame in tqdm(df_seq.groupby("frame_id_int"),
                                       desc=f"Frames in {seq}"):
            # Load the full frame image
            img_path = df_frame.iloc[0]["img_path"]
            
            try:
                frame = load_frame_cache(img_path, frame_cache)
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue
            
            # convert detections to DeepSORT format
            detections = df_to_deepsort_detections(df_frame, frame)
            
            # update tracker (DeepSORT extracts appearance features internally)
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # match tracks back to original detections
            width = int(df_frame.iloc[0]["width"])
            height = int(df_frame.iloc[0]["height"])
            
            # DeepSORT returns tracks with track_id
            for track in tracks:
                if not track.is_confirmed():
                    continue  # Skip unconfirmed tracks
                
                track_id = track.track_id
                ltrb = track.to_ltrb()  # get bounding box [left, top, right, bottom]
                x1, y1, x2, y2 = map(int, ltrb)
                
                # find matching detection in original df_frame by IoU
                best_match_idx = None
                best_iou = 0
                
                for idx, row in df_frame.iterrows():
                    # calculate IoU
                    det_x1, det_y1, det_x2, det_y2 = row['x1'], row['y1'], row['x2'], row['y2']
                    
                    xi1 = max(x1, det_x1)
                    yi1 = max(y1, det_y1)
                    xi2 = min(x2, det_x2)
                    yi2 = min(y2, det_y2)
                    
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    box1_area = (x2 - x1) * (y2 - y1)
                    box2_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                    union_area = box1_area + box2_area - inter_area
                    
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = idx
                
                if best_match_idx is None or best_iou < 0.5:
                    continue  # no good match found
                
                # get original detection info
                matched_row = df_frame.loc[best_match_idx]
                
                all_tracks.append({
                    "sequence": seq,
                    "frame_id": matched_row["frame_id"],
                    "track_id": track_id,
                    "class_id": int(matched_row["class_id"]),
                    "score": float(matched_row["score"]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "crop_path": matched_row["crop_path"],
                    "img_path": img_path,
                    "width": width,
                    "height": height
                })
            
            # clear frame cache periodically to save memory
            if len(frame_cache) > 50:
                frame_cache.clear()
    
    tracks_df = pd.DataFrame(all_tracks)
    
    # filter very short tracks
    track_lengths = tracks_df.groupby(['sequence', 'track_id']).size()
    valid_tracks = track_lengths[track_lengths >= 3].index
    tracks_df = tracks_df.set_index(['sequence', 'track_id']).loc[valid_tracks].reset_index()
    
    tracks_df.to_csv(TRACK_OUT, index=False)
    print(f"\n[DONE] wrote {len(tracks_df)} rows -> {TRACK_OUT}")
    
    # Print statistics
    final_track_lengths = tracks_df.groupby(['sequence', 'track_id']).size()
    print(f"\nTrack statistics:")
    print(f"Total tracks: {len(final_track_lengths)}")
    print(f"Mean length: {final_track_lengths.mean():.1f} frames")
    print(f"Median length: {final_track_lengths.median():.1f} frames")
    print(f"Max length: {final_track_lengths.max()} frames")

if __name__ == "__main__":
    main()
