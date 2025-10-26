import json

INPUT_JSON = "/Users/elizabethdwenger/Desktop/MScYr2/ADL/json_txt/hand_labeled_annotations_redo.json"
OUTPUT_TXT = "/Users/elizabethdwenger/Desktop/MScYr2/ADL/json_txt/light_on_off_labels.txt"

def is_light_on(entry):
    if entry["turn_signal"] != "none":
        return 1
    if entry["tail_light"] not in ["off", "not_visible"]:
        return 1
    return 0

with open(INPUT_JSON) as f:
    data = json.load(f)

with open(OUTPUT_TXT, "w") as f:
    for d in data:
        img_path = d["image"]
        label = is_light_on(d)
        f.write(f"{img_path} {label}\n")

print(f"Saved {OUTPUT_TXT} with {len(data)} entries.")
