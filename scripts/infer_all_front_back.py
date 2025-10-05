"""
Load a checkpoint saved by train.py (best_model.pth) and run inference over
a list of image paths (one per line). Writes positive predictions (label==1)
to output file.

example:
python scripts/infer_all.py \
  --checkpoint runs/resnet18_expt/best_model.pth \
  --image-list filtered_landscape_images.txt \
  --img-root /gpfs/space/projects/ml2024 \
  --output back_of_car_filtered.txt
"""

import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

def build_model(checkpoint_path, device):
    model = models.resnet18(weights=None)
    
    # Replace final layer for binary classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--image-list", required=True, help="Text file with one image path per line")
    parser.add_argument("--img-root", default=".", help="Prefix for relative paths in list")
    parser.add_argument("--output", default="back_of_car_filtered.txt")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    model = build_model(args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.output, "w") as out_f:
        with open(args.image_list, "r") as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                img_path = path if os.path.isabs(path) else os.path.join(args.img_root, path)
                if not os.path.exists(img_path):
                    print("Missing:", img_path)
                    continue
                
                img = Image.open(img_path).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(device, non_blocking=True)
                
                with torch.no_grad():
                    out = model(img_t)
                    pred = out.argmax(dim=1).item()
                
                if pred == 1:
                    out_f.write(path + "\n")
    
    print("Wrote positive predictions to", args.output)

if __name__ == "__main__":
    main()
