import os
import pandas as pd
from tqdm import tqdm
import supervision as sv

DET_CSV   = "/gpfs/helios/home/dwenger/detections_with_crop_path.csv"
TRACK_OUT = "/gpfs/helios/home/dwenger/tracks_with_crops.csv"

def df_to_detections(df_frame):
    xyxy = df_frame[["x1","y1","x2","y2"]].to_numpy()
    scores = df_frame["score"].to_numpy()
    classes = df_frame["class_id"].to_numpy()
    det = sv.Detections(xyxy=xyxy)
    det.confidence = scores
    det.class_id = classes
    return det

def main():
    df = pd.read_csv(DET_CSV)
    # Convert frame_id (string) → int for correct ordering
    df["frame_id_int"] = df["frame_id"].astype(int)
    df = df.sort_values(["sequence", "frame_id_int"]).reset_index(drop=True)
    
    all_tracks = []
    
    for seq, df_seq in df.groupby("sequence"):
        print(f"Processing sequence: {seq}")
        
        # supervision 0.26.1 API
        tracker = sv.ByteTrack(
            track_activation_threshold=0.6,
            lost_track_buffer=20,
            minimum_matching_threshold=0.85,
            frame_rate=30
        )
        
        for frame_id, df_frame in df_seq.groupby("frame_id_int"):
            dets = df_to_detections(df_frame)
            
            # Just pass detections, no frame_resolution_wh
            tracks = tracker.update_with_detections(detections=dets)
            
            width = int(df_frame.iloc[0]["width"])
            height = int(df_frame.iloc[0]["height"])
            
            for i in range(len(tracks)):
                x1, y1, x2, y2 = tracks.xyxy[i].astype(int).tolist()
                tid = int(tracks.tracker_id[i])
                conf = float(tracks.confidence[i])
                cid = int(tracks.class_id[i])
                
                # Keep crop path!
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
    tracks_df.to_csv(TRACK_OUT, index=False)
    print(f"[DONE] wrote {len(tracks_df)} rows → {TRACK_OUT}")

if __name__ == "__main__":
    main()
