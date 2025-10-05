"""
Load a checkpoint saved by train.py (best_model.pth) and run inference over
a list of image paths (one per line). Writes positive predictions (label==1)
to output file.

example:
python scripts/infer_all.py \
  --model-name resnet18 \
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

def replace_final_layer(model, num_classes):
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            for i in range(len(cls)-1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    in_features = cls[i].in_features
                    cls[i] = nn.Linear(in_features, num_classes)
                    model.classifier = cls
                    return
        elif isinstance(cls, nn.Linear):
            in_features = cls.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Could not find a final classifier layer to replace.")

def build_model(name, checkpoint_path, device):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(pretrained=False)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=False)
    else:
        raise ValueError("Unsupported model.")
    replace_final_layer(model, 2)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=["resnet18","efficientnet_b0","mobilenet_v3_small"])
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--image-list", required=True, help="Text file with one image path per line")
    parser.add_argument("--img-root", default=".", help="Prefix for relative paths in list")
    parser.add_argument("--output", default="back_of_car_filtered.txt")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    out_f = open(args.output, "w")
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
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                pred = out.argmax(dim=1).item()
            if pred == 1:
                out_f.write(path + "\n")
    out_f.close()
    print("Wrote positive predictions to", args.output)

if __name__ == "__main__":
    main()

