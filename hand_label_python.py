import os
import json
import subprocess
import platform

def open_image(image_path):
    """Opens an image with the system default viewer"""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", image_path], check=True)
        elif platform.system() == "Windows":
            os.startfile(image_path)
        else:  # Linux / WSL
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        print(f"Could not open image: {e}")

def get_image_paths(base_dir):
    """Finds all image files in a directory and its subdirectories"""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def interactive_labeling(json_file, image_base_dir):
    # Try to load existing annotations
    try:
        with open(json_file, "r") as f:
            annotations = json.load(f)
        print(f"Loaded existing annotations from {json_file}.")
    except (FileNotFoundError, json.JSONDecodeError):
        # If file doesn't exist or is empty, auto-populate it
        print(f"No valid annotation file found. Auto-populating from {image_base_dir}...")
        image_paths = get_image_paths(image_base_dir)
        annotations = [{"image": path, "labels": []} for path in image_paths]
        
        with open(json_file, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Created a new annotation file with {len(annotations)} images.")

    # Start the interactive labeling process
    for entry in annotations:
        if entry["labels"]:
            continue
            
        print(f"\nImage: {entry['image']}")
        open_image(entry["image"])

        labels_input = input("Enter labels (comma-separated), '-' for no labels, or Enter to skip: ").strip()
        if labels_input == "":
            print("Skipped for now.")
            continue
        elif labels_input == "-":
            entry["labels"] = []
            print("Marked as no labels.")
        else:
            entry["labels"] = [label.strip() for label in labels_input.split(",")]
            print(f"Saved labels: {entry['labels']}")

        # Save progress after each change
        with open(json_file, "w") as f:
            json.dump(annotations, f, indent=2)

    print("\nAll done! Your annotations are up to date.")


interactive_labeling("hand_labeled_annotations.json", "sampled_images")

# python hand_label.py
# Opens image, type left_turn, right_turn, hazard, no_visible_light, tail_light_on, or - (for no lights no)
