import argparse
import json
import os

import nibabel as nib
import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dice_score(pred, gt, label, eps=1e-6):
    p = (pred == label).astype(np.float32)
    g = (gt == label).astype(np.float32)
    intersection = (p * g).sum()
    return (2.0 * intersection + eps) / (p.sum() + g.sum() + eps)


def iou_score(pred, gt, label, eps=1e-6):
    p = (pred == label).astype(np.float32)
    g = (gt == label).astype(np.float32)
    intersection = (p * g).sum()
    union = (p + g - p * g).sum()
    return (intersection + eps) / (union + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/nnunet.yaml")
    parser.add_argument("--predictions", required=True, help="Directory with nnU-Net prediction .nii.gz files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir     = cfg["nnunet"]["raw_dir"]
    dataset     = cfg["nnunet"]["dataset_name"]
    dataset_dir = os.path.join(raw_dir, dataset)
    labels_dir  = os.path.join(dataset_dir, "labelsTr")

    # Load test scan mapping
    mapping_path = os.path.join(dataset_dir, "test_scan_mapping.json")
    with open(mapping_path) as f:
        mapping = json.load(f)  # {case_id: scan_idx}

    dice_liver_list, dice_tumor_list = [], []
    iou_liver_list,  iou_tumor_list  = [], []

    for case_id in sorted(mapping.keys()):
        pred_path = os.path.join(args.predictions, f"{case_id}.nii.gz")
        # Ground truth is in labelsTr only for train cases; for test we need original seg
        scan_idx  = mapping[case_id]
        gt_path   = os.path.join(
            cfg["data"]["raw_dir"], f"segmentation-{scan_idx}.nii"
        )

        if not os.path.exists(pred_path):
            print(f"  [WARNING] Prediction not found: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"  [WARNING] GT not found: {gt_path}")
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.int16)
        gt   = nib.load(gt_path).get_fdata().astype(np.int16)

        dice_liver_list.append(dice_score(pred, gt, label=1))
        dice_tumor_list.append(dice_score(pred, gt, label=2))
        iou_liver_list.append(iou_score(pred, gt, label=1))
        iou_tumor_list.append(iou_score(pred, gt, label=2))

    print("\n── nnU-Net 3D Full-Res Test Results ─────────────────")
    print(f"  Dice  Liver : {np.mean(dice_liver_list):.4f}")
    print(f"  Dice  Tumor : {np.mean(dice_tumor_list):.4f}")
    print(f"  IoU   Liver : {np.mean(iou_liver_list):.4f}")
    print(f"  IoU   Tumor : {np.mean(iou_tumor_list):.4f}")
    print(f"  (n={len(dice_liver_list)} test cases)")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
