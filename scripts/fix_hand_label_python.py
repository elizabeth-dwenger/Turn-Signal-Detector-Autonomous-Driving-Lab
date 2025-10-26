import os
import json
import subprocess
import platform
import time

# Added "u": "unclear"
TURN_SIGNAL_OPTIONS = {"l": "left", "r": "right", "n": "none", "b": "both", "u": "unclear"}
TAIL_LIGHT_OPTIONS = {"o": "on", "f": "off", "v": "not_visible", "u": "unclear"}

UNDO_COMMANDS = {"x", "undo"}

def open_image(image_path):
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", image_path], check=True)
            time.sleep(0.3)
            subprocess.run(["osascript", "-e", 'tell application "Terminal" to activate'])
        elif platform.system() == "Windows":
            os.startfile(image_path)
        else:  # Linux / WSL
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        print(f"Could not open image: {e}")

def prompt_choice(prompt, options_dict):
    shortcuts = "/".join([f"{k}({v})" for k, v in options_dict.items()])
    while True:
        value = input(f"{prompt} [{shortcuts} or 'undo']: ").strip().lower()
        if value in {"undo", "x"}:
            return "UNDO"
        if value in options_dict:
            return options_dict[value]
        if value in options_dict.values():
            return value
        print(f"Invalid input. Use one of: {', '.join(options_dict.values())}, their shortcuts, or 'undo'.")

def interactive_labeling(base_json, new_json, image_base_dir):
    # Load original annotations
    try:
        with open(base_json, "r") as f:
            original_annotations = json.load(f)
        print(f"Loaded base annotations from {base_json}.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Could not read {base_json}. Exiting.")
        return

    # Separate those to keep and those to re-label
    to_keep = [
        entry for entry in original_annotations
        if entry.get("turn_signal") == "none" and entry.get("tail_light") == "off"
    ]
    to_redo = [
        entry.copy()
        for entry in original_annotations
        if not (entry.get("turn_signal") == "none" and entry.get("tail_light") == "off")
    ]
    print(f"Keeping {len(to_keep)} entries (none/off). Re-labeling {len(to_redo)} entries.")

    # Try to load progress from existing redo file
    existing_map = {}
    try:
        with open(new_json, "r") as f:
            existing = json.load(f)
            print(f"Loaded existing annotations from {new_json} (will resume).")
            existing_map = {e["image"]: e for e in existing}
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid redo annotation file found. Starting fresh.")

    # Merge existing annotations (resume)
    for entry in to_redo:
        if entry["image"] in existing_map:
            entry.update(existing_map[entry["image"]])
        else:
            # Blank out labels so we get prompted
            entry["turn_signal"] = ""
            entry["tail_light"] = ""

    index = 0
    undo_stack = []

    while index < len(to_redo):
        entry = to_redo[index]

        # Skip if already labeled
        if entry.get("turn_signal") and entry.get("tail_light"):
            index += 1
            continue

        image_path = os.path.join(image_base_dir, entry["image"])
        print(f"\nImage: {entry['image']}")
        open_image(image_path)

        turn_signal = prompt_choice("Turn signal", TURN_SIGNAL_OPTIONS)
        if turn_signal == "UNDO":
            if not undo_stack:
                print("Nothing to undo.")
                continue
            last_index = undo_stack.pop()
            to_redo[last_index]["turn_signal"] = ""
            to_redo[last_index]["tail_light"] = ""
            index = last_index
            continue

        tail_light = prompt_choice("Tail light", TAIL_LIGHT_OPTIONS)
        if tail_light == "UNDO":
            if not undo_stack:
                print("Nothing to undo.")
                continue
            last_index = undo_stack.pop()
            to_redo[last_index]["turn_signal"] = ""
            to_redo[last_index]["tail_light"] = ""
            index = last_index
            continue

        # Save the label
        entry["turn_signal"] = turn_signal
        entry["tail_light"] = tail_light
        undo_stack.append(index)

        # Merge and save full annotations
        merged = sorted(to_redo + to_keep, key=lambda x: x["image"])
        with open(new_json, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"Saved: turn_signal={turn_signal}, tail_light={tail_light}")
        index += 1

    # Final save
    merged = sorted(to_redo + to_keep, key=lambda x: x["image"])
    with open(new_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nAll done! Saved complete annotations to {new_json}.")

if __name__ == "__main__":
    interactive_labeling(
        base_json="../hand_labeled_annotations.json",
        new_json="../hand_labeled_annotations_redo.json",
        image_base_dir="../sampled_images"
    )
