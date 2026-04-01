import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import UNet
from src.dataset import LiTSDataset
from src.utils import CombinedLoss, compute_metrics, aggregate_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    total_loss = 0.0
    metric_list = []
    num_classes = cfg["data"]["num_classes"]
    grad_clip = cfg["training"]["gradient_clip_max_norm"]
    use_amp = scaler is not None

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


# ── Training loop ─────────────────────────────────────────────────────────────

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
    model_cfg = cfg["model"]
    loss_cfg  = cfg["loss"]
    sched_cfg = cfg["scheduler"]
    paths_cfg = cfg["paths"]

    model = UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=data_cfg["num_classes"],
        encoder_channels=model_cfg["encoder_channels"],
        bottleneck_channels=model_cfg["bottleneck_channels"],
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

    train_ds = LiTSDataset(data_cfg["train_manifest"], augment=True)
    val_ds   = LiTSDataset(data_cfg["val_manifest"],   augment=False)

    pin = data_cfg["pin_memory"] and device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True,  num_workers=data_cfg["num_workers"],
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                              shuffle=False, num_workers=data_cfg["num_workers"],
                              pin_memory=pin)

    start_epoch     = 0
    best_dice       = 0.0
    patience_counter = 0
    patience_limit  = train_cfg["early_stopping_patience"]
    ckpt_every      = train_cfg["checkpoint_every_n_epochs"]
    ckpt_dir        = paths_cfg["checkpoint_dir"]
    best_path       = os.path.join(ckpt_dir, paths_cfg["best_model_filename"])

    resume_path = train_cfg.get("resume_checkpoint")
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_dice = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer,
                                        device, cfg, scaler, is_train=True)
        val_loss,   val_m   = run_epoch(model, val_loader,   criterion, None,
                                        device, cfg, scaler, is_train=False)

        val_dice = np.mean([val_m.get("dice_liver", 0.0), val_m.get("dice_tumor", 0.0)])

        if sched_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

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
            periodic_path = os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pth")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
            }, periodic_path)

        if val_dice > best_dice:
            best_dice = val_dice
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

    if use_wb:
        wandb.finish()

    print(f"Training complete. Best val Dice: {best_dice:.4f}")
    return best_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiTS Stage 1 — Vanilla U-Net")
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    parser.add_argument("--wandb",  action="store_true", help="Enable W&B logging")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.resume:
        cfg["training"]["resume_checkpoint"] = args.resume

    device = get_device()
    print(f"Device: {device}")

    train(cfg, device, use_wb=args.wandb)


if __name__ == "__main__":
    main()
