# Base GroundingDINO didn't work at all for this
import os, json, sys
from tqdm import tqdm
import torch

from groundingdino.util.inference import load_model, load_image, predict

LOCAL_BASES = [
    "sampled_images",
    "../sampled_images",
    "/Users/elizabethdwenger/Desktop/MScYr2/ADL/sampled_images"
]
IMAGE_JSON = "sample_100_for_openai.json"
OUTPUT_JSON = "grounding_dino_results.json"
MODEL_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "/Users/elizabethdwenger/Desktop/MScYr2/ADL/runs/groundingdino/groundingdino_swint_ogc.pth"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------

def resolve_local_path(relative_path):
    rp = relative_path.lstrip("/")
    for base in LOCAL_BASES:
        candidate = os.path.join(base, rp)
        if os.path.exists(candidate):
            return candidate
    # try the raw path if it's actually local already
    if os.path.exists(relative_path):
        return relative_path
    return None

def detect_lights_dino(model, image_path, device):
    try:
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption="car taillight, left turn signal, right turn signal, brake light",
            box_threshold=0.3,
            text_threshold=0.25,
            device=device
        )
        phrases = [p.lower() for p in phrases]

        # heuristics
        turn_signal = "none"
        if any("left" in p for p in phrases): turn_signal = "left"
        if any("right" in p for p in phrases):
            turn_signal = "right" if turn_signal == "none" else "both"

        if any(("brake" in p) or ("taillight" in p) or ("tail light" in p) for p in phrases):
            tail_light = "on"
        else:
            tail_light = "off"

        return {"turn_signal": turn_signal, "tail_light": tail_light, "phrases": phrases}
    except Exception as e:
        return {"turn_signal": "none", "tail_light": "not_visible", "error": str(e)}

def main():
    # load model
    print("DEVICE:", DEVICE)
    model = load_model(MODEL_CONFIG, MODEL_WEIGHTS)
    model.to(DEVICE).eval()
    print("Model loaded.")

    # read 100-sample list
    with open(IMAGE_JSON) as f:
        items = json.load(f)

    results = []
    for entry in tqdm(items, desc="Grounding DINO"):
        img_rel = entry["image"]
        local = resolve_local_path(img_rel)
        if local is None:
            results.append({
                "image": img_rel,
                "hand_label": {"turn_signal": entry.get("turn_signal"), "tail_light": entry.get("tail_light")},
                "grounding_dino_label": {"turn_signal": "none", "tail_light": "not_visible"},
                "note": "image not found locally"
            })
            continue

        res = detect_lights_dino(model, local, DEVICE)
        results.append({
            "image": img_rel,
            "hand_label": {"turn_signal": entry.get("turn_signal"), "tail_light": entry.get("tail_light")},
            "grounding_dino_label": {"turn_signal": res["turn_signal"], "tail_light": res["tail_light"]},
            "phrases": res.get("phrases", []),
            "error": res.get("error", None)
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved Grounding DINO results to", OUTPUT_JSON)

if __name__ == "__main__":
    main()

