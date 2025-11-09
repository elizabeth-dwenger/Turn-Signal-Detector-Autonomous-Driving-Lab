import os
import csv
from PIL import Image
from datetime import datetime
import traceback

LIST_FILE = "/gpfs/helios/home/dwenger/back_of_car_filtered.txt"
OUT_CSV = "/gpfs/helios/home/dwenger/detections.csv"
CHECKPOINT_DIR = "/gpfs/helios/home/dwenger/csv_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#LIST_FILE = "/gpfs/helios/home/dwenger/test_subset.txt"
#OUT_CSV   = "/gpfs/helios/home/dwenger/detections_test.csv"

FRAME_LOCATIONS = [
    ".",
    "predict",
    os.path.join("predict", "images"),
    "images",
    "frames",
]

def split_at(path, token):
    parts = path.split(token)
    if len(parts) < 2:
        return None, None
    left = parts[0]
    right = token + parts[1]
    return left, right

def parse_crop_filename(filename):
    name = filename.replace(".jpg", "").replace(".png", "")
    frame_id = name[:6]
    crop_idx = name[6:] if len(name) > 6 else ""
    # Convert crop index to line number in label file
    # No suffix = line 0, suffix "2" = line 1, suffix "3" = line 2, etc.
    if crop_idx == "":
        line_idx = 0
    else:
        line_idx = int(crop_idx) - 1
    return frame_id, line_idx

def find_session_and_frame(crop_path):
    crop_path = os.path.normpath(crop_path)
    try:
        idx = crop_path.index(os.path.join("predict", "crops"))
    except ValueError:
        return None, None, None
    session_dir = crop_path[:idx].rstrip(os.sep)
    filename = os.path.basename(crop_path)
    frame_id, line_idx = parse_crop_filename(filename)
    return session_dir, frame_id, line_idx

def find_label_path(session_dir, frame_id):
    return os.path.join(session_dir, "predict", "labels", f"{frame_id}.txt")

def find_frame_image_path(session_dir, frame_id):
    candidates = []
    for loc in FRAME_LOCATIONS:
        if loc == ".":
            candidates.append(os.path.join(session_dir, f"{frame_id}.jpg"))
            candidates.append(os.path.join(session_dir, f"{frame_id}.png"))
        else:
            candidates.append(os.path.join(session_dir, loc, f"{frame_id}.jpg"))
            candidates.append(os.path.join(session_dir, loc, f"{frame_id}.png"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x_center = cx * img_w
    y_center = cy * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = max(0, min(img_w - 1, x_center - bw / 2))
    y1 = max(0, min(img_h - 1, y_center - bh / 2))
    x2 = max(0, min(img_w - 1, x_center + bw / 2))
    y2 = max(0, min(img_h - 1, y_center + bh / 2))
    return int(x1), int(y1), int(x2), int(y2)

def find_last_checkpoint():
    """Find the highest checkpoint index."""
    parts = []
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("detections_part_") and f.endswith(".csv"):
            try:
                idx = int(f.replace("detections_part_", "").replace(".csv", ""))
                parts.append(idx)
            except:
                pass
    return max(parts) if parts else 0

def consolidate_checkpoints_to_final(last_checkpoint_idx):
    """
    Consolidate all checkpoint files into the final CSV without loading everything into memory.
    Only call this once at the start if resuming.
    """
    if last_checkpoint_idx == 0:
        return
    
    print(f"[INFO] Consolidating checkpoints into {OUT_CSV}...")
    
    # Get all checkpoint files in order
    checkpoint_files = []
    for i in range(100000, last_checkpoint_idx + 1, 100000):
        cp_file = os.path.join(CHECKPOINT_DIR, f"detections_part_{i}.csv")
        if os.path.exists(cp_file):
            checkpoint_files.append(cp_file)
    
    if not checkpoint_files:
        return
    
    # Write consolidated file by streaming through each checkpoint
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "sequence","frame_id","crop_path","frame_path","width","height",
        "class_id","score","x1","y1","x2","y2"
    ]
    with open(OUT_CSV, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for cp_file in checkpoint_files:
            with open(cp_file, "r") as inf:
                reader = csv.DictReader(inf)
                for row in reader:
                    writer.writerow(row)
    
    print(f"[INFO] Consolidated {len(checkpoint_files)} checkpoint files into {OUT_CSV}")

def main():
    last_idx = find_last_checkpoint()
    print(f"Resuming from line {last_idx + 1}")

    # Consolidate existing checkpoints into final CSV once
    if last_idx > 0 and not os.path.exists(OUT_CSV):
        consolidate_checkpoints_to_final(last_idx)

    # Open CSV in APPEND mode for ongoing writes
    # for fresh start, write header; if resuming, just append
    csv_mode = "a" if os.path.exists(OUT_CSV) else "w"
    csv_file = open(OUT_CSV, csv_mode, newline="")
    fieldnames = [
        "sequence","frame_id","crop_path","frame_path","width","height",
        "class_id","score","x1","y1","x2","y2"
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if csv_mode == "w":
        writer.writeheader()

    # Buffer for batch writing to checkpoint files
    checkpoint_buffer = []
    
    missed = 0
    total_crops = 0
    rows_written = 0

    try:
        with open(LIST_FILE, "r") as f:
            for i, line in enumerate(f, start=1):
                # Skip previously processed lines
                if i <= last_idx:
                    continue

                crop_path = line.strip()
                if not crop_path:
                    continue
                total_crops += 1

                if i % 10000 == 0:
                    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"Processed {i} crops. Missed: {missed}", flush=True)

                try:
                    session_dir, frame_id, line_idx = find_session_and_frame(crop_path)
                    if session_dir is None:
                        print(f"[WARN] Could not parse session_dir/frame_id from: {crop_path}", flush=True)
                        continue

                    label_path = find_label_path(session_dir, frame_id)
                    frame_img_path = find_frame_image_path(session_dir, frame_id)

                    if not os.path.exists(label_path):
                        print(f"[MISS] Label missing: {label_path}", flush=True)
                        missed += 1
                        continue
                    if not frame_img_path or not os.path.exists(frame_img_path):
                        print(f"[MISS] Frame image missing: {session_dir} {frame_id}", flush=True)
                        missed += 1
                        continue

                    with Image.open(frame_img_path) as im:
                        img_w, img_h = im.size

                    sequence = os.path.basename(session_dir)
                    parent = os.path.basename(os.path.dirname(session_dir))
                    sequence = os.path.join(parent, sequence)

                    # Read only the specific line that corresponds to this crop
                    with open(label_path, "r") as lf:
                        lines = lf.readlines()
                        if line_idx >= len(lines):
                            print(f"[WARN] Crop {crop_path} wants line {line_idx} but label only has {len(lines)} lines", flush=True)
                            continue
                        
                        lab = lines[line_idx]
                        parts = lab.strip().split()
                        if len(parts) < 5:
                            continue
                        class_id = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        score = float(parts[5]) if len(parts) >= 6 else 1.0
                        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
                        
                        row = {
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "crop_path": crop_path,
                            "frame_path": frame_img_path,
                            "width": img_w,
                            "height": img_h,
                            "class_id": class_id,
                            "score": score,
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2
                        }
                        
                        # Write directly to main CSV
                        writer.writerow(row)
                        rows_written += 1
                        
                        # buffer for checkpoint
                        checkpoint_buffer.append(row)

                except Exception as e:
                    print(f"[ERR] Exception processing crop {crop_path}: {e}", flush=True)
                    traceback.print_exc()

                # Write checkpoint every 100k lines
                if i % 100000 == 0:
                    csv_file.flush()  # write to main CSV is written
                    
                    # Write checkpoint file
                    temp_csv = os.path.join(CHECKPOINT_DIR, f"detections_part_{i}.csv")
                    os.makedirs(os.path.dirname(temp_csv), exist_ok=True)
                    with open(temp_csv, "w", newline="") as cf:
                        cp_writer = csv.DictWriter(cf, fieldnames=[
                            "sequence","frame_id","crop_path","frame_path","width","height",
                            "class_id","score","x1","y1","x2","y2"
                        ])
                        cp_writer.writeheader()
                        cp_writer.writerows(checkpoint_buffer)
                    
                    checkpoint_buffer.clear()  # Clear buffer to free memory
                    
                    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"Wrote checkpoint CSV: {temp_csv}", flush=True)

    finally:
        csv_file.close()

    print(f"[DONE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Wrote {rows_written} new detections to {OUT_CSV}. Missed {missed} frames "
          f"out of {total_crops} crops.", flush=True)

if __name__ == "__main__":
    main()
