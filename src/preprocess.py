import argparse
import csv
import os

import nibabel as nib
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


def window_and_normalize(slice_2d, hu_min, hu_max):
    clipped = np.clip(slice_2d, hu_min, hu_max).astype(np.float32)
    mean, std = clipped.mean(), clipped.std()
    if std < 1e-6:
        return np.zeros_like(clipped)
    return (clipped - mean) / std


def extract_slices(scan_idx, raw_dir, out_dir, hu_min, hu_max, smoke_limit=None):
    try:
        vol = nib.load(os.path.join(raw_dir, f"volume-{scan_idx}.nii")).get_fdata(dtype=np.float32)
        seg = nib.load(os.path.join(raw_dir, f"segmentation-{scan_idx}.nii")).get_fdata().astype(np.int16)
    except OSError as e:
        print(f"  [WARNING] Skipping scan {scan_idx}: {e}")
        return []

    records = []
    extracted = 0

    for z in range(vol.shape[2]):
        if smoke_limit is not None and extracted >= smoke_limit:
            break
        if seg[:, :, z].max() == 0:
            continue

        img = window_and_normalize(vol[:, :, z], hu_min, hu_max)

        img_fname = f"slice_{scan_idx:03d}_{z:04d}.npy"
        msk_fname = f"mask_{scan_idx:03d}_{z:04d}.npy"

        np.save(os.path.join(out_dir, img_fname), img)
        np.save(os.path.join(out_dir, msk_fname), seg[:, :, z])

        records.append((os.path.join(out_dir, img_fname), os.path.join(out_dir, msk_fname)))
        extracted += 1

    return records


def write_manifest(records, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path"])
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="LiTS preprocessing: NIfTI -> 2D slices")
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    parser.add_argument("--smoke", action="store_true", help="Extract a small subset for smoke testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    pre_cfg = cfg["preprocessing"]
    smoke_cfg = cfg.get("smoke_test", {})

    raw_dir = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    hu_min = pre_cfg["hu_min"]
    hu_max = pre_cfg["hu_max"]
    seed = pre_cfg["random_seed"]
    val_frac = pre_cfg["val_split"]
    test_frac = pre_cfg["test_split"]

    smoke_total = smoke_cfg.get("num_slices", 10) if args.smoke else None

    # Smoke mode writes to a separate directory to avoid overwriting real data
    if args.smoke:
        processed_dir = os.path.join(processed_dir, "smoke")
        manifest_map_override = {
            "train": os.path.join(processed_dir, "train.csv"),
            "val":   os.path.join(processed_dir, "val.csv"),
            "test":  os.path.join(processed_dir, "test.csv"),
        }
    else:
        manifest_map_override = None

    scan_ids = discover_scans(raw_dir)
    if not scan_ids:
        raise FileNotFoundError(f"No volume-*.nii files found in {raw_dir}")
    print(f"Found {len(scan_ids)} scans")

    train_ids, val_ids, test_ids = scan_split(scan_ids, val_frac, test_frac, seed)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test scans")

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    manifest_map = manifest_map_override or {
        "train": data_cfg["train_manifest"],
        "val": data_cfg["val_manifest"],
        "test": data_cfg["test_manifest"],
    }

    smoke_budget = {}
    if smoke_total is not None:
        smoke_budget["train"] = max(1, int(smoke_total * (1 - val_frac - test_frac)))
        smoke_budget["val"] = max(1, int(smoke_total * val_frac))
        smoke_budget["test"] = max(1, smoke_total - smoke_budget["train"] - smoke_budget["val"])

    for split, ids in split_map.items():
        out_dir = os.path.join(processed_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        all_records = []
        remaining = smoke_budget.get(split) if smoke_total is not None else None

        for scan_idx in ids:
            if remaining is not None and remaining <= 0:
                break
            records = extract_slices(scan_idx, raw_dir, out_dir, hu_min, hu_max, smoke_limit=remaining)
            all_records.extend(records)
            if remaining is not None:
                remaining -= len(records)
            print(f"  [{split}] scan {scan_idx:03d}: {len(records)} slices")

        write_manifest(all_records, manifest_map[split])
        print(f"  [{split}] {len(all_records)} total slices -> {manifest_map[split]}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
