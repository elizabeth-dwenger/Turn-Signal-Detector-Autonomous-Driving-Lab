"""
Training script for SignalClassifier.

Usage:
    python -m signal_classifier.train --config signal_classifier/config.yaml

Features
--------
* Class-weighted cross-entropy or focal loss (configured via config.yaml).
* Learning-rate cosine annealing with warm restarts.
* Early stopping on validation accuracy.
* Best checkpoint saved to checkpoint_dir/best.pt; every-epoch checkpoint to last.pt.
* Per-class accuracy and macro-F1 logged at the end of each validation epoch.
* Optional TensorBoard logging (silently skipped if tensorboard is not installed).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from .dataset import SlidingWindowDataset, build_datasets, LABEL_NAMES
from .model import SignalClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # per-class weights [num_classes]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)                # [B, C]
        probs     = log_probs.exp()                              # [B, C]

        # Gather probability of the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B]

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt                            # [B]

        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w[targets]

        return loss.mean()


def build_criterion(cfg_training: dict, class_weights: torch.Tensor) -> nn.Module:
    weight = class_weights if cfg_training.get("use_class_weights", True) else None
    loss_type = cfg_training.get("loss", "cross_entropy")
    if loss_type == "focal":
        gamma = cfg_training.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, weight=weight)
    # default: cross-entropy
    return nn.CrossEntropyLoss(weight=weight)


def build_train_sampler(dataset: SlidingWindowDataset, cfg_training: dict):
    """Optionally build an inverse-frequency sampler over training windows."""
    if not cfg_training.get("use_balanced_sampler", False):
        return None

    class_weights = dataset.class_weights()
    sample_weights = torch.tensor(
        [float(class_weights[w.label_int]) for w in dataset.windows],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def apply_same_class_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
    cfg_training: dict,
) -> torch.Tensor:
    """Mix features only with samples from the same class and camera."""
    if not cfg_training.get("mixup_enabled", False):
        return features

    mixup_prob = float(cfg_training.get("mixup_prob", 0.0))
    if mixup_prob <= 0.0 or torch.rand(1).item() >= mixup_prob:
        return features

    alpha = float(cfg_training.get("mixup_alpha", 0.2))
    if alpha <= 0.0:
        return features

    cameras = metadata.get("camera", []) if isinstance(metadata, dict) else []
    same_camera_only = cfg_training.get("mixup_same_camera_only", True)
    same_class_only = cfg_training.get("mixup_same_class_only", True)

    mixed = features.clone()
    batch_size = labels.size(0)
    for i in range(batch_size):
        candidates = []
        for j in range(batch_size):
            if i == j:
                continue
            if same_class_only and int(labels[j].item()) != int(labels[i].item()):
                continue
            if same_camera_only and cameras and cameras[j] != cameras[i]:
                continue
            candidates.append(j)

        if not candidates:
            continue

        partner_idx = candidates[torch.randint(len(candidates), (1,)).item()]
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        mixed[i] = lam * features[i] + (1.0 - lam) * features[partner_idx]

    return mixed


def apply_temporal_augmentation(
    features: torch.Tensor,
    cfg_training: dict,
) -> torch.Tensor:
    """Apply mild, fixed-length temporal augmentation to feature windows."""
    if not cfg_training.get("temporal_aug_enabled", False):
        return features

    augmented = features.clone()
    batch_size, num_frames = augmented.shape[:2]
    if num_frames < 2:
        return augmented

    jitter_prob = float(cfg_training.get("temporal_jitter_prob", 0.0))
    drop_prob = float(cfg_training.get("temporal_drop_prob", 0.0))
    duplicate_prob = float(cfg_training.get("temporal_duplicate_prob", 0.0))

    for i in range(batch_size):
        window = augmented[i]

        if jitter_prob > 0.0 and torch.rand(1).item() < jitter_prob:
            if torch.rand(1).item() < 0.5:
                window = torch.cat([window[:1], window[:-1]], dim=0)
            else:
                window = torch.cat([window[1:], window[-1:]], dim=0)

        if drop_prob > 0.0 and torch.rand(1).item() < drop_prob and num_frames > 2:
            drop_idx = torch.randint(0, num_frames, (1,)).item()
            kept = torch.cat([window[:drop_idx], window[drop_idx + 1 :]], dim=0)
            pad_idx = min(drop_idx, kept.size(0) - 1)
            pad_frame = kept[pad_idx : pad_idx + 1]
            window = torch.cat([kept[:pad_idx], pad_frame, kept[pad_idx:]], dim=0)

        if duplicate_prob > 0.0 and torch.rand(1).item() < duplicate_prob and num_frames > 2:
            dup_idx = torch.randint(0, num_frames, (1,)).item()
            duplicate = window[dup_idx : dup_idx + 1]
            insert_at = min(dup_idx + 1, num_frames)
            expanded = torch.cat([window[:insert_at], duplicate, window[insert_at:]], dim=0)
            remove_idx = min(num_frames, expanded.size(0) - 1)
            window = torch.cat([expanded[:remove_idx], expanded[remove_idx + 1 :]], dim=0)

        augmented[i] = window

    return augmented


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: list, all_labels: list, num_classes: int = 4):
    """
    Compute overall accuracy and per-class accuracy + macro-F1.
    Returns a dict suitable for logging.
    """
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / max(len(all_labels), 1)

    # Per-class TP, FP, FN
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for p, l in zip(all_preds, all_labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    f1_per_class = []
    for c in range(num_classes):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec  = tp[c] / max(tp[c] + fn[c], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        f1_per_class.append(f1)

    macro_f1 = sum(f1_per_class) / num_classes

    per_class_acc = []
    for c in range(num_classes):
        total_c = sum(1 for l in all_labels if l == c)
        acc_c   = tp[c] / max(total_c, 1)
        per_class_acc.append(acc_c)

    return {
        "accuracy":      accuracy,
        "macro_f1":      macro_f1,
        "f1_per_class":  f1_per_class,
        "acc_per_class": per_class_acc,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer=None,
    device: str = "cuda",
    is_train: bool = True,
    cfg_training: dict = None,
):
    """Run one epoch.  Returns (avg_loss, all_preds, all_labels)."""
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    cfg_training = cfg_training or {}

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if len(batch) == 3:
                features, labels, metadata = batch
            else:
                features, labels = batch
                metadata = {}

            features = features.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            if is_train:
                features = apply_temporal_augmentation(features, cfg_training)
                features = apply_same_class_mixup(features, labels, metadata, cfg_training)

            logits = model(features)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, all_preds, all_labels


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int, meta: dict):
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "meta":        meta,
        },
        path,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SignalClassifier.")
    parser.add_argument("--config", default="signal_classifier/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tc = cfg["training"]
    mc = cfg["model"]
    wc = cfg["window"]
    dc = cfg["data"]
    fc = cfg["features"]

    device = tc.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Paths ──────────────────────────────────────────────────────────────
    meta_path    = Path(fc["output_dir"]) / "metadata.json"
    ckpt_dir     = Path(tc["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        log.error(
            f"metadata.json not found at {meta_path}. "
            "Run extract_features.py first."
        )
        return

    # ── Datasets and loaders ───────────────────────────────────────────────
    log.info("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(str(meta_path), wc, dc)

    log.info(f"  Train windows: {len(train_ds)}")
    log.info(f"  Val   windows: {len(val_ds)}")
    log.info(f"  Test  windows: {len(test_ds)}")

    train_counts = train_ds.class_counts()
    log.info("  Train class distribution: " +
             ", ".join(f"{LABEL_NAMES[i]}={train_counts[i]}" for i in range(4)))

    num_workers = tc.get("num_workers", 4)
    train_sampler = build_train_sampler(train_ds, tc)
    train_loader = DataLoader(
        train_ds,
        batch_size  = tc["batch_size"],
        shuffle     = train_sampler is None,
        sampler     = train_sampler,
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tc["batch_size"] * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    dino_name = fc.get("dino_model_name", "")
    d_dino = 1024 if "large" in dino_name else 384
    model = SignalClassifier(
        T          = wc["size"],
        P          = fc["spatial_tokens"],
        d_dino     = d_dino,
        d_model    = mc["d_model"],
        d_hidden   = mc["d_hidden"],
        num_layers = mc["num_layers"],
        num_heads  = mc["num_heads"],
        dropout    = mc.get("dropout", 0.1),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # ── Loss ───────────────────────────────────────────────────────────────
    class_weights = train_ds.class_weights()
    log.info("Class weights: " +
             ", ".join(f"{LABEL_NAMES[i]}={class_weights[i]:.3f}" for i in range(4)))
    criterion = build_criterion(tc, class_weights.to(device))
    log.info(
        "Training augmentations: "
        f"balanced_sampler={tc.get('use_balanced_sampler', False)}, "
        f"mixup={tc.get('mixup_enabled', False)}, "
        f"temporal_aug={tc.get('temporal_aug_enabled', False)}"
    )

    # ── Optimiser and scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tc["learning_rate"],
        weight_decay = tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = tc["epochs"],
        eta_min= tc["learning_rate"] * 0.01,
    )

    # ── Optional TensorBoard ────────────────────────────────────────────────
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))
        log.info("TensorBoard logging enabled.")
    except ImportError:
        pass

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc      = 0.0
    patience_counter  = 0
    patience          = tc.get("early_stopping_patience", 10)
    history           = []

    log.info("Starting training...")
    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss, train_preds, train_labels = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True, cfg_training=tc
        )
        train_metrics = compute_metrics(train_preds, train_labels)
        scheduler.step()

        # Validate
        val_loss, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer=None, device=device, is_train=False, cfg_training=tc
        )
        val_metrics = compute_metrics(val_preds, val_labels)

        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:3d}/{tc['epochs']}  "
            f"lr={lr:.2e}  "
            f"train_loss={train_loss:.4f}  train_acc={train_metrics['accuracy']:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Per-class val stats
        for i, name in enumerate(LABEL_NAMES):
            log.info(
                f"  {name:7s}  acc={val_metrics['acc_per_class'][i]:.3f}  "
                f"f1={val_metrics['f1_per_class'][i]:.3f}"
            )

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("Acc/train",  train_metrics["accuracy"], epoch)
            writer.add_scalar("Acc/val",    val_metrics["accuracy"],   epoch)
            writer.add_scalar("F1/val_macro", val_metrics["macro_f1"], epoch)

        row = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_metrics["accuracy"],
            "val_acc":    val_metrics["accuracy"],
            "val_f1":     val_metrics["macro_f1"],
        }
        history.append(row)

        # Save every-epoch checkpoint (overwrite)
        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer, scheduler, epoch,
            {"val_acc": val_metrics["accuracy"], "val_f1": val_metrics["macro_f1"]}
        )

        # Save best checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc     = val_metrics["accuracy"]
            patience_counter = 0
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer, scheduler, epoch,
                {"val_acc": best_val_acc, "val_f1": val_metrics["macro_f1"]}
            )
            log.info(f"  ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {history_path}")

    if writer:
        writer.close()

    log.info(f"Training complete.  Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
