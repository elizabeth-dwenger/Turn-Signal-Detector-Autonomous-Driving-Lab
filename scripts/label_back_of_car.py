import os
import json
import subprocess
import platform

OUTPUT_FILE = "front_of_car_labels.txt"

def open_image(image_path):
    try:
        if platform.system() == "Darwin":  # macOS focus on terminal
            subprocess.run(["open", image_path], check=True)
        elif platform.system() == "Windows":
            os.startfile(image_path)
        else:  # Linux / WSL
            subprocess.run(["xdg-open", image_path], check=True)
    except Exception as e:
        print(f"Could not open image: {e}")

def refocus_terminal_mac():
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "Terminal" to activate'],
            check=True
        )
    except Exception as e:
        print(f"Could not refocus Terminal: {e}")

def get_image_paths(base_dir):
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)
                image_paths.append(rel_path)
    return sorted(image_paths)

def load_existing_labels(file_path):
    labels = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(" ", 1)
                labels[path] = label
        print(f"Loaded {len(labels)} existing labels from {file_path}.")
    return labels

def save_labels(file_path, labels):
    with open(file_path, "w") as f:
        for img, lbl in labels.items():
            f.write(f"{img} {lbl}\n")

def interactive_labeling(image_base_dir):
    image_paths = get_image_paths(image_base_dir)
    labels = load_existing_labels(OUTPUT_FILE)

    i = 0
    while i < len(image_paths):
        img_rel = image_paths[i]
        if img_rel in labels:
            i += 1
            continue

        img_path = os.path.join(image_base_dir, img_rel)
        print(f"\nImage {i+1}/{len(image_paths)}: {img_rel}")
        open_image(img_path)
        refocus_terminal_mac()

        user_input = input("Label as front_of_car? [1=yes / 0=no / u=undo / q=quit]: ").strip().lower()
        if user_input == "q":
            print("Exiting early. Progress saved.")
            break
        elif user_input == "u" and i > 0:
            prev_img = image_paths[i-1]
            print(f"Undoing label for: {prev_img}")
            labels.pop(prev_img, None)
            i -= 1
            save_labels(OUTPUT_FILE, labels)
            continue
        elif user_input in ["1", "0"]:
            labels[img_rel] = user_input
            save_labels(OUTPUT_FILE, labels)
            print(f"Saved: {img_rel} -> {user_input}")
            i += 1
        else:
            print("Invalid input. Please press 1, 0, u, or q.")

    print(f"\nAll done! {len(labels)} images labeled and saved in {OUTPUT_FILE}.")

if __name__ == "__main__":
    interactive_labeling("front_car_images")

