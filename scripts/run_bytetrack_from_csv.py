import os
import pandas as pd
from tqdm import tqdm
import supervision as sv

#DET_CSV   = "/gpfs/helios/home/dwenger/detections.csv"
#TRACK_OUT = "/gpfs/helios/home/dwenger/tracks.csv"
DET_CSV   = "/gpfs/helios/home/dwenger/detections_test.csv"
TRACK_OUT = "/gpfs/helios/home/dwenger/tracks_test.csv"

def df_to_detections(df_seq_frame):
    """
    Convert a frame's rows to supervision.Detections
    """
    xyxy = df_seq_frame[["x1","y1","x2","y2"]].to_numpy()
    conf = df_seq_frame["score"].to_numpy() if "score" in df_seq_frame.columns else None
    cls  = df_seq_frame["class_id"].to_numpy() if "class_id" in df_seq_frame.columns else None
    det = sv.Detections(xyxy=xyxy)
    if conf is not None:
        det.confidence = conf
    if cls is not None:
        det.class_id = cls
    return det

def main():
    df = pd.read_csv(DET_CSV)

    # Ensure sorting by sequence and frame_id
    def coerce_int(x):
        try:
            return int(x)
        except:
            return x
    df["frame_id_sort"] = df["frame_id"].apply(coerce_int)
    df = df.sort_values(["sequence", "frame_id_sort"]).reset_index(drop=True)

    all_tracks = []

    for seq, df_seq in df.groupby("sequence"):
        # tracker per sequence
        tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        # Get resolution from first row
        seq_w = int(df_seq.iloc[0]["width"])
        seq_h = int(df_seq.iloc[0]["height"])

        # Iterate frames in order
        for frame_id, df_frame in df_seq.groupby("frame_id_sort"):
            dets = df_to_detections(df_frame)

            # Update tracker with detections for this frame
            tracks = tracker.update_with_detections(dets)

            if len(tracks) == 0:
                continue

            # Export per detection in this frame
            for i in range(len(tracks)):
                x1, y1, x2, y2 = tracks.xyxy[i].astype(int).tolist()
                tid = int(tracks.tracker_id[i]) if tracks.tracker_id is not None else -1
                conf = float(tracks.confidence[i]) if tracks.confidence is not None else 1.0
                cid = int(tracks.class_id[i]) if tracks.class_id is not None else -1

                orig_row = df_frame.iloc[i]
                all_tracks.append({
                    "sequence": seq,
                    "frame_id": orig_row["frame_id"],
                    "img_path": orig_row["img_path"],
                    "track_id": tid,
                    "class_id": cid,
                    "score": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": seq_w, "height": seq_h
                })

    tracks_df = pd.DataFrame(all_tracks)
    tracks_df.to_csv(TRACK_OUT, index=False)
    print(f"[DONE] Wrote {len(tracks_df)} rows to {TRACK_OUT}")

if __name__ == "__main__":
    main()
