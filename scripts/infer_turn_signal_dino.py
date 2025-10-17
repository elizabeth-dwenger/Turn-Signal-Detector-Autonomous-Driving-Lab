import torch
from groundingdino.util.inference import load_model, load_image, predict
import json
from tqdm import tqdm
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Paths
MODEL_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "groundingdino_swint_ogc.pth"
IMAGE_LIST = "sample_back_cars_10k.txt"
OUTPUT_JSON = "light_states_grounding_dino.json"

# Base path for my local images, should be changed if used
LOCAL_BASE = "/Users/elizabethdwenger/Desktop/MScYr2/ADL/sampled_images"

# Load model
model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)
model.to(DEVICE).eval()

def fix_path(path):
    if path.startswith("/gpfs/"):
        return os.path.join(LOCAL_BASE, path.lstrip("/"))
    return path

def detect_lights(image_path):
    try:
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption="car taillight, left turn signal, right turn signal, brake light",
            box_threshold=0.3,
            text_threshold=0.25,
            device=DEVICE
        )
        phrases = [p.lower() for p in phrases]

        # Basic heuristic rules
        turn_signal = "none"
        if any("left" in p for p in phrases):
            turn_signal = "left"
        if any("right" in p for p in phrases):
            turn_signal = "right" if turn_signal == "none" else "both"

        tail_light = "on" if any("brake" in p or "taillight" in p for p in phrases) else "off"

        return {"turn_signal": turn_signal, "tail_light": tail_light}
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"turn_signal": "none", "tail_light": "not_visible"}

# Process all images
results = []
with open(IMAGE_LIST) as f:
    paths = [fix_path(line.strip()) for line in f if line.strip()]

for path in tqdm(paths):
    result = detect_lights(path)
    results.append({
        "image": path,
        **result
    })

# Save output
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done! Saved results to {OUTPUT_JSON}")
