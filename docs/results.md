# Experiment Results

## Stage 1: Vanilla U-Net

| Run | Epochs | LR | Batch | Val Dice Liver | Val Dice Tumor | Test Dice Liver | Test Dice Tumor | Test IoU Liver | Test IoU Tumor | Notes |
|-----|--------|----|-------|---------------|---------------|----------------|----------------|---------------|---------------|-------|
| 1 | 27 (early stop) | 1e-4 | 8 | 0.8983 | 0.7020 | 0.8651 | 0.6898 | 0.8047 | 0.6363 | Cosine LR, CE+Dice loss, A100, SLURM job 5684951 |

---

## Stage 2: U-Net + ResNet50

| Run | Epochs | LR | Batch | Val Dice Liver | Val Dice Tumor | Test Dice Liver | Test Dice Tumor | Test IoU Liver | Test IoU Tumor | Notes |
|-----|--------|----|-------|---------------|---------------|----------------|----------------|---------------|---------------|-------|
| 1 | 31 (early stop) | 1e-4 | 8 | 0.9325 | 0.7222 | 0.8990 | 0.6886 | 0.8477 | 0.6357 | Cosine LR, CE+Dice loss, V100, SLURM job 5830409 |

---

## Stage 3: nnU-Net

_Pending_

---

## Summary Comparison

| Stage | Architecture | Test Dice Liver | Test Dice Tumor | Test IoU Liver | Test IoU Tumor |
|-------|-------------|----------------|----------------|---------------|---------------|
| 1 | Vanilla U-Net | 0.8651 | 0.6898 | 0.8047 | 0.6363 |
| 2 | U-Net + ResNet50 | 0.8990 | 0.6886 | 0.8477 | 0.6357 |
| 3 | nnU-Net | — | — | — | — |
