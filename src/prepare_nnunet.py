import argparse
import gzip
import json
import os
import shutil

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def discover_scans(raw_dir):
    indices = []
    for fname in os.listdir(raw_dir):
        if fname.startswith("volume-") and fname.endswith(".nii"):
            idx = int(fname.replace("volume-", "").replace(".nii", ""))
            if os.path.exists(os.path.join(raw_dir, f"segmentation-{idx}.nii")):
                indices.append(idx)
    return sorted(indices)


def scan_split(scan_ids, val_frac, test_frac, seed):
    remaining, test_ids = train_test_split(scan_ids, test_size=test_frac, random_state=seed)
    val_frac_adj = val_frac / (1.0 - test_frac)
    train_ids, val_ids = train_test_split(remaining, test_size=val_frac_adj, random_state=seed)
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def compress_nii(src_path, dst_path):
    """Compress a .nii file to .nii.gz."""
    with open(src_path, "rb") as f_in:
        with gzip.open(dst_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main():
    parser = argparse.ArgumentParser(description="Convert LiTS data to nnU-Net v2 format")
    parser.add_argument("--config", default="configs/nnunet.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    raw_dir = cfg["data"]["raw_dir"]
    out_dir = cfg["nnunet"]["raw_dir"]       # e.g. /scratch/.../nnunet/raw
    dataset = cfg["nnunet"]["dataset_name"]  # e.g. Dataset001_LiTS
    seed    = cfg["preprocessing"]["random_seed"]
    val_frac  = cfg["preprocessing"]["val_split"]
    test_frac = cfg["preprocessing"]["test_split"]

    dataset_dir  = os.path.join(out_dir, dataset)
    images_tr    = os.path.join(dataset_dir, "imagesTr")
    labels_tr    = os.path.join(dataset_dir, "labelsTr")
    images_ts    = os.path.join(dataset_dir, "imagesTs")

    for d in [images_tr, labels_tr, images_ts]:
        os.makedirs(d, exist_ok=True)

    scan_ids = discover_scans(raw_dir)
    if not scan_ids:
        raise FileNotFoundError(f"No volume-*.nii files found in {raw_dir}")
    print(f"Found {len(scan_ids)} scans")

    train_ids, val_ids, test_ids = scan_split(scan_ids, val_frac, test_frac, seed)
    # nnU-Net uses its own cross-validation — combine train+val into imagesTr
    trainval_ids = sorted(train_ids + val_ids)
    print(f"imagesTr: {len(trainval_ids)} scans | imagesTs: {len(test_ids)} scans")

    # Convert training+val scans
    training_cases = []
    for case_idx, scan_idx in enumerate(trainval_ids):
        case_id = f"LiTS_{case_idx:05d}"
        vol_src = os.path.join(raw_dir, f"volume-{scan_idx}.nii")
        seg_src = os.path.join(raw_dir, f"segmentation-{scan_idx}.nii")

        img_dst = os.path.join(images_tr, f"{case_id}_0000.nii.gz")
        lbl_dst = os.path.join(labels_tr, f"{case_id}.nii.gz")

        compress_nii(vol_src, img_dst)
        compress_nii(seg_src, lbl_dst)
        training_cases.append(case_id)
        print(f"  [train] scan {scan_idx:03d} → {case_id}")

    # Convert test scans
    test_cases = []
    for case_idx, scan_idx in enumerate(test_ids):
        case_id = f"LiTS_test_{case_idx:05d}"
        vol_src = os.path.join(raw_dir, f"volume-{scan_idx}.nii")
        img_dst = os.path.join(images_ts, f"{case_id}_0000.nii.gz")
        compress_nii(vol_src, img_dst)
        test_cases.append(case_id)
        print(f"  [test]  scan {scan_idx:03d} → {case_id}")

    # Save test scan index mapping for evaluation
    mapping = {f"LiTS_test_{i:05d}": idx for i, idx in enumerate(test_ids)}
    with open(os.path.join(dataset_dir, "test_scan_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "liver": 1,
            "tumor": 2,
        },
        "numTraining": len(trainval_ids),
        "file_ending": ".nii.gz",
        "name": dataset,
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset written to {dataset_dir}")
    print(f"  imagesTr:  {len(trainval_ids)} cases")
    print(f"  imagesTs:  {len(test_ids)} cases")
    print("Ready to run: nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")


if __name__ == "__main__":
    main()
