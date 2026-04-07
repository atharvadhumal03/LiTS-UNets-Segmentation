import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet_resnet50 import build_model


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class LiTSDataset(Dataset):
    def __init__(self, manifest_csv, augment=False):
        self.records = []
        with open(manifest_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((row["image_path"], row["mask_path"]))
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, msk_path = self.records[idx]

        image = np.load(img_path).astype(np.float32)   # (H, W)
        mask  = np.load(msk_path).astype(np.int64)     # (H, W)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = np.fliplr(image).copy()
                mask  = np.fliplr(mask).copy()
            if torch.rand(1).item() > 0.5:
                image = np.flipud(image).copy()
                mask  = np.flipud(mask).copy()

        # Repeat single channel 3x and apply ImageNet normalization
        image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        mask = torch.from_numpy(mask)  # (H, W)

        return image, mask


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight, dice_weight, class_weights=None):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w)

    def soft_dice(self, logits, targets):
        probs      = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        smooth     = 1e-6
        dims       = (0, 2, 3)
        intersection     = (probs * targets_oh).sum(dim=dims)
        union            = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_class   = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, logits, targets):
        return (
            self.ce_weight   * self.ce(logits, targets)
            + self.dice_weight * self.soft_dice(logits, targets)
        )


# ── Metrics ───────────────────────────────────────────────────────────────────

CLASS_NAMES = {1: "liver", 2: "tumor"}


def compute_metrics(preds, targets, num_classes):
    metrics = {}
    smooth  = 1e-6
    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        intersection = (pred_c * tgt_c).sum()
        pred_sum     = pred_c.sum()
        tgt_sum      = tgt_c.sum()
        dice = (2.0 * intersection + smooth) / (pred_sum + tgt_sum + smooth)
        iou  = (intersection + smooth) / (pred_sum + tgt_sum - intersection + smooth)
        name = CLASS_NAMES[c]
        metrics[f"dice_{name}"] = dice.item()
        metrics[f"iou_{name}"]  = iou.item()
    return metrics


def aggregate_metrics(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0) + 1, ckpt.get("best_dice", 0.0)


# ── Epoch ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, cfg, scaler, is_train):
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    metric_list = []
    num_classes = cfg["data"]["num_classes"]
    grad_clip   = cfg["training"]["gradient_clip_max_norm"]
    use_amp     = scaler is not None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss   = criterion(logits, masks)
            else:
                logits = model(images)
                loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            metric_list.append(compute_metrics(preds, masks, num_classes))

    return total_loss / len(loader), aggregate_metrics(metric_list)


# ── Test ──────────────────────────────────────────────────────────────────────

def test(cfg, device, checkpoint_path):
    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["data"]["num_classes"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_cfg    = cfg["data"]
    test_ds     = LiTSDataset(data_cfg["test_manifest"], augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    metric_list = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images).argmax(dim=1)
            metric_list.append(compute_metrics(preds, masks, data_cfg["num_classes"]))

    results = aggregate_metrics(metric_list)

    print("\n── Test Results ─────────────────────────────────")
    print(f"  Dice  Liver : {results.get('dice_liver', 0):.4f}")
    print(f"  Dice  Tumor : {results.get('dice_tumor', 0):.4f}")
    print(f"  IoU   Liver : {results.get('iou_liver',  0):.4f}")
    print(f"  IoU   Tumor : {results.get('iou_tumor',  0):.4f}")
    print("─────────────────────────────────────────────────")

    return results


# ── Train ─────────────────────────────────────────────────────────────────────

def train(cfg, device, use_wb=False):
    if use_wb:
        from dotenv import load_dotenv
        import wandb
        load_dotenv()
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.getenv("WANDB_ENTITY"),
            config=cfg,
        )

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg  = cfg["loss"]
    sched_cfg = cfg["scheduler"]
    paths_cfg = cfg["paths"]

    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        num_classes=data_cfg["num_classes"],
    ).to(device)

    criterion = CombinedLoss(
        num_classes=data_cfg["num_classes"],
        ce_weight=loss_cfg["ce_weight"],
        dice_weight=loss_cfg["dice_weight"],
        class_weights=loss_cfg["class_weights"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    sched_type = sched_cfg["type"]
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg["T_max"], eta_min=sched_cfg["eta_min"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=sched_cfg["plateau_factor"],
            patience=sched_cfg["plateau_patience"],
            min_lr=sched_cfg["plateau_min_lr"],
        )

    use_amp = cfg["mixed_precision"]["enabled"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    pin          = data_cfg["pin_memory"] and device.type == "cuda"
    train_loader = DataLoader(LiTSDataset(data_cfg["train_manifest"], augment=True),
                              batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=data_cfg["num_workers"], pin_memory=pin)
    val_loader   = DataLoader(LiTSDataset(data_cfg["val_manifest"], augment=False),
                              batch_size=train_cfg["batch_size"], shuffle=False,
                              num_workers=data_cfg["num_workers"], pin_memory=pin)

    start_epoch      = 0
    best_dice        = 0.0
    patience_counter = 0
    patience_limit   = train_cfg["early_stopping_patience"]
    ckpt_every       = train_cfg["checkpoint_every_n_epochs"]
    ckpt_dir         = paths_cfg["checkpoint_dir"]
    best_path        = os.path.join(ckpt_dir, paths_cfg["best_model_filename"])

    resume_path = train_cfg.get("resume_checkpoint")
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_dice = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer,
                                  device, cfg, scaler, is_train=True)
        val_loss, val_m = run_epoch(model, val_loader, criterion, None,
                                    device, cfg, scaler, is_train=False)

        val_dice = np.mean([val_m.get("dice_liver", 0.0), val_m.get("dice_tumor", 0.0)])
        lr_now   = optimizer.param_groups[0]["lr"]

        if sched_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"dice_liver={val_m.get('dice_liver', 0):.4f} | "
            f"dice_tumor={val_m.get('dice_tumor', 0):.4f} | "
            f"lr={lr_now:.2e}"
        )

        if use_wb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "lr": lr_now,
                **{f"val_{k}": v for k, v in val_m.items()},
            })

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth"))

        if val_dice > best_dice:
            best_dice        = val_dice
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
            }, best_path)
            print(f"  → Best model saved (val Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")

    if os.path.exists(best_path):
        test_results = test(cfg, device, best_path)
        if use_wb:
            wandb.log({f"test_{k}": v for k, v in test_results.items()})

    if use_wb:
        wandb.finish()

    return best_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiTS Stage 2 — U-Net + ResNet50 encoder")
    parser.add_argument("--config",     default="configs/unet_resnet50.yaml")
    parser.add_argument("--mode",       choices=["train", "evaluate"], default="train")
    parser.add_argument("--wandb",      action="store_true")
    parser.add_argument("--resume",     default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    if args.mode == "train":
        if args.resume:
            cfg["training"]["resume_checkpoint"] = args.resume
        train(cfg, device, use_wb=args.wandb)

    elif args.mode == "evaluate":
        checkpoint_path = args.checkpoint or os.path.join(
            cfg["paths"]["checkpoint_dir"], cfg["paths"]["best_model_filename"]
        )
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)
        test(cfg, device, checkpoint_path)


if __name__ == "__main__":
    main()
