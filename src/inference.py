import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import UNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer(cfg, device, checkpoint_path, input_path, output_path):
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    pre_cfg   = cfg["preprocessing"]

    model = UNet(
        in_channels=model_cfg["in_channels"],
        num_classes=data_cfg["num_classes"],
        encoder_channels=model_cfg["encoder_channels"],
        bottleneck_channels=model_cfg["bottleneck_channels"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hu_min = pre_cfg["hu_min"]
    hu_max = pre_cfg["hu_max"]

    vol_nii = nib.load(input_path)
    volume  = vol_nii.get_fdata(dtype=np.float32)   # H x W x D
    H, W, D = volume.shape
    pred_vol = np.zeros((H, W, D), dtype=np.int16)

    with torch.no_grad():
        for z in range(D):
            sl = volume[:, :, z]
            clipped = np.clip(sl, hu_min, hu_max).astype(np.float32)
            mean, std = clipped.mean(), clipped.std()
            if std < 1e-6:
                continue
            norm = (clipped - mean) / std
            tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(tensor)
            pred_vol[:, :, z] = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int16)

    out_nii = nib.Nifti1Image(pred_vol, vol_nii.affine, vol_nii.header)
    nib.save(out_nii, output_path)
    print(f"Saved prediction to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a NIfTI volume")
    parser.add_argument("--config",     default="configs/unet_baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input",      required=True, help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("--output",     default=None,  help="Output mask path (default: input_pred.nii.gz)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    checkpoint_path = args.checkpoint or os.path.join(
        cfg["paths"]["checkpoint_dir"], cfg["paths"]["best_model_filename"]
    )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.input.replace(".nii.gz", "_pred.nii.gz").replace(".nii", "_pred.nii.gz")

    device = get_device()
    print(f"Device: {device}")

    infer(cfg, device, checkpoint_path, args.input, output_path)


if __name__ == "__main__":
    main()
