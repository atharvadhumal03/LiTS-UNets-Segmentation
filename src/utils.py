import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss ──────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight, dice_weight, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w)

    def soft_dice(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        smooth = 1e-6
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice_per_class.mean()

    def forward(self, logits, targets):
        return (
            self.ce_weight * self.ce(logits, targets)
            + self.dice_weight * self.soft_dice(logits, targets)
        )


# ── Metrics ───────────────────────────────────────────────────────────────────

CLASS_NAMES = {1: "liver", 2: "tumor"}


def compute_metrics(preds, targets, num_classes):
    """
    Compute per-class Dice and IoU, excluding background (class 0).

    Args:
        preds:   (N, H, W) int tensor — argmax predictions
        targets: (N, H, W) int tensor — ground truth labels
    Returns:
        dict with keys dice_liver, iou_liver, dice_tumor, iou_tumor
    """
    metrics = {}
    smooth = 1e-6

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
