import csv

import numpy as np
import torch
from torch.utils.data import Dataset


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
        mask = np.load(msk_path).astype(np.int64)       # (H, W), values 0/1/2

        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if torch.rand(1).item() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = torch.from_numpy(image).unsqueeze(0)    # (1, H, W)
        mask = torch.from_numpy(mask)                    # (H, W)

        return image, mask
