"""
train/val files each line is: <path> <label>
paths may be relative to --img-root or absolute.
saves best model to --save-dir/best_model.pth

 example run:
caffeinate python scripts/train_front_back.py \
  --train-file train_front_back.txt \
  --val-file val_front_back.txt \
  --img-root front_back_images \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-4 \
  --save-dir runs/resnet18_expt
"""

import os
import argparse
import random
from pathlib import Path
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    SKL = True
except Exception:
    SKL = False

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SimpleImageDataset(Dataset):
    def __init__(self, txt_file, img_root, transform=None):
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.rsplit(" ", 1)
                self.samples.append((path, int(label)))
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        if os.path.isabs(rel_path):
            img_path = rel_path
        else:
            img_path = os.path.join(self.img_root, rel_path)
        # Ensure file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def build_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            pred = outputs.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            trues.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    if SKL:
        acc = accuracy_score(trues, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="binary", zero_division=0)
        return avg_loss, acc, prec, rec, f1
    else:
        acc = sum(p==t for p,t in zip(preds, trues)) / len(trues)
        return avg_loss, acc, None, None, None

def train(args):
    set_seed(args.seed)
    print("Using device:", device)

    # Transforms - using updated normalization values from ResNet18_Weights.DEFAULT
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = SimpleImageDataset(args.train_file, args.img_root, transform=train_transform)
    val_ds = SimpleImageDataset(args.val_file, args.img_root, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=True if args.num_workers > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           persistent_workers=True if args.num_workers > 0 else False)

    model = build_model(num_classes=2, pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_path = None

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        start = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} | time {elapsed:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        if SKL and val_prec is not None:
            print(f"    precision {val_prec:.4f} recall {val_rec:.4f} f1 {val_f1:.4f}")

        scheduler.step(val_loss)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_dir, exist_ok=True)
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "args": vars(args),
            }, best_path)
            print(f"Saved best model to {best_path}")

    print("Training complete. Best val_acc:", best_val_acc)
    if best_path:
        print("Best model:", best_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--img-root", required=True, help="Root directory to join relative image paths (or '.' if labels contain absolute paths)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
