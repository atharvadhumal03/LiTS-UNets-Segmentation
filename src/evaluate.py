import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import UNet
from src.dataset import LiTSDataset
from src.utils import compute_metrics, aggregate_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(cfg, device, checkpoint_path):
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    model = UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=data_cfg["num_classes"],
        encoder_channels=model_cfg["encoder_channels"],
        bottleneck_channels=model_cfg["bottleneck_channels"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = LiTSDataset(data_cfg["test_manifest"], augment=False)
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
            logits = model(images)
            preds  = logits.argmax(dim=1)
            metric_list.append(compute_metrics(preds, masks, data_cfg["num_classes"]))

    results = aggregate_metrics(metric_list)

    print(f"\nEvaluation results — {checkpoint_path}")
    print(f"  Dice  Liver : {results.get('dice_liver', 0):.4f}")
    print(f"  Dice  Tumor : {results.get('dice_tumor', 0):.4f}")
    print(f"  IoU   Liver : {results.get('iou_liver',  0):.4f}")
    print(f"  IoU   Tumor : {results.get('iou_tumor',  0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--config",     default="configs/unet_baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["paths"]["checkpoint_dir"], cfg["paths"]["best_model_filename"]
    )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    evaluate(cfg, device, checkpoint_path)


if __name__ == "__main__":
    main()
