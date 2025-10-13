import os
import json
import subprocess
import platform
import time

TURN_SIGNAL_OPTIONS = {"l": "left", "r": "right", "n": "none", "b": "both"}
TAIL_LIGHT_OPTIONS = {"o": "on", "f": "off", "v": "not_visible"}

UNDO_COMMANDS = {"u", "undo"}

def open_image(image_path):
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", image_path], check=True)
            time.sleep(0.3)
            subprocess.run([
                "osascript", "-e",
                'tell application "Terminal" to activate'
            ])
        elif platform.system() == "Windows":
            os.startfile(image_path)
        else:  # Linux / WSL
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        print(f"Could not open image: {e}")

def get_image_paths(base_dir):
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)
                image_paths.append(rel_path)
    return sorted(image_paths)

def prompt_choice(prompt, options_dict):
    shortcuts = "/".join([f"{k}({v})" for k, v in options_dict.items()])
    while True:
        value = input(f"{prompt} [{shortcuts} or 'u' to undo]: ").strip().lower()
        if value in UNDO_COMMANDS:
            return "UNDO"
        if value in options_dict:
            return options_dict[value]
        if value in options_dict.values():
            return value
        print(f"Invalid input. Use one of: {', '.join(options_dict.values())}, their shortcuts, or 'u' to undo.")

def interactive_labeling(json_file, image_base_dir):
    # Load/initialize annotation list
    try:
        with open(json_file, "r") as f:
            annotations = json.load(f)
        print(f"Loaded existing annotations from {json_file}.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid annotation file found. Auto-populating from {image_base_dir}...")
        image_paths = get_image_paths(image_base_dir)
        annotations = [{"image": path, "turn_signal": "", "tail_light": ""} for path in image_paths]
        with open(json_file, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Created a new annotation file with {len(annotations)} images.")

    index = 0
    undo_stack = []

    while index < len(annotations):
        entry = annotations[index]

        # Skip if already labeled
        if entry["turn_signal"] and entry["tail_light"]:
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
            annotations[last_index]["turn_signal"] = ""
            annotations[last_index]["tail_light"] = ""
            index = last_index
            with open(json_file, "w") as f:
                json.dump(annotations, f, indent=2)
            print(f"Undo successful. Returning to image: {annotations[index]['image']}")
            continue

        tail_light = prompt_choice("Tail light", TAIL_LIGHT_OPTIONS)
        if tail_light == "UNDO":
            if not undo_stack:
                print("Nothing to undo.")
                continue
            last_index = undo_stack.pop()
            annotations[last_index]["turn_signal"] = ""
            annotations[last_index]["tail_light"] = ""
            index = last_index
            with open(json_file, "w") as f:
                json.dump(annotations, f, indent=2)
            print(f"Undo successful. Returning to image: {annotations[index]['image']}")
            continue

        # Save the current annotation
        entry["turn_signal"] = turn_signal
        entry["tail_light"] = tail_light
        undo_stack.append(index)

        print(f"Saved: turn_signal={turn_signal}, tail_light={tail_light}")
        with open(json_file, "w") as f:
            json.dump(annotations, f, indent=2)

        index += 1

    print("\n All done! Annotations are up to date.")

if __name__ == "__main__":
    interactive_labeling("hand_labeled_annotations.json", "../sampled_images")



# python hand_label.py
# Opens image, type one of the shortcuts or the full name
