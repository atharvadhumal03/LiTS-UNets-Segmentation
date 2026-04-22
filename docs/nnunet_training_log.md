# nnU-Net Training Log — Stage 3

**Architecture:** nnU-Net 3D full-res (3d_fullres)
**Dataset:** LiTS — Dataset001_LiTS, Fold 0 (88 train / 23 val cases)
**Target:** 1000 epochs
**Hardware:** Northeastern University Discovery Cluster — A100-SXM4-80GB / V100-SXM2-32GB (SLURM)
**Config:** patch size [128,128,128], batch size 2, 6-stage PlainConvUNet, 320 max channels

---

## Run Summary

| Run | Date | Job ID | GPU | Epochs Covered | Epochs Gained | First Dice [Liver, Tumor] | Last Dice [Liver, Tumor] | Last Train Loss | Last Val Loss |
|-----|------|--------|-----|---------------|--------------|--------------------------|--------------------------|----------------|---------------|
| 1 | Apr 12 00:40 | 5877824 | V100 | 0 → 558 | 559 | [0.706, 0.000] | [0.910, 0.506] | -0.6604 | -0.4220 |
| 2 | Apr 12 11:07 | 5884642 | V100 | 0 → 451 | 452 | [0.677, 0.000] | [0.919, 0.373] | -0.6353 | -0.3642 |
| 3 | Apr 13 13:23 | 5911321 | T4 | 0 → 3 | 4 | [0.650, 0.000] | [0.828, 0.000] | -0.0376 | -0.1215 |
| 4 | Apr 16 04:50 | 5965867 | A100 | 0 → 188 | 189 | [0.631, 0.000] | [0.912, 0.341] | -0.5330 | -0.3684 |
| 5 | Apr 17 03:00 | 5984599 | A100 | 0 → 192 | 193 | [0.646, 0.000] | [0.885, 0.373] | -0.5800 | -0.2742 |
| 6 | Apr 17 23:55 | 6055395 | A100 | 0 → 179 | 180 | [0.661, 0.000] | [0.869, 0.419] | -0.5420 | -0.2943 |
| 7 | Apr 18 22:11 | 6192681 | A100 | 150 → 251 | 102 | [0.905, 0.429] | [0.890, 0.376] | -0.6103 | -0.3167 |
| 8 | Apr 21 13:15 | 6268167 | A100 | 250 → 441 | 192 | [0.906, 0.387] | [0.889, 0.300] | -0.6185 | -0.2160 |
| 9 | Apr 22 01:47 | 6275563 | A100 | 400 → 593 | 194 | [0.928, 0.372] | [0.932, 0.394] | -0.6444 | -0.4237 |

---

## Notes

**Runs 1–6 (failed to resume):**
Runs 1–6 all restarted from epoch 0 on each resubmission. Root causes:
- Runs 1–2: `nnUNet_n_proc_DA=4` on HPC caused a fork+CUDA multiprocessing deadlock — training never started
- Run 3: Allocated a T4 (15 GB) instead of A100 — insufficient VRAM, manually cancelled
- Runs 4–6: Training ran correctly but the `--c` (resume) flag was missing from the script — nnU-Net restarted from scratch each time

**Runs 7+ (resuming correctly):**
`--c` flag added in run 7. nnU-Net now resumes from `checkpoint_latest.pth` on each resubmission.

**Current canonical checkpoint:** epoch 593 (from run 9)
**Epochs remaining:** ~407

---

## Effective Training Progress

Discarding restarts, the canonical training sequence is:

| Segment | Epochs | Best Dice (Tumor) |
|---------|--------|-------------------|
| 0 – 192 | Run 5 (cleanest early run) | 0.373 |
| 150 – 251 | Run 7 (first resume) | 0.376 |
| 250 – 441 | Run 8 | 0.300 |
| 400 – 593 | Run 9 | 0.394 |

**Current state: epoch 593 / 1000. ~407 epochs remaining.**

---

## Key Metrics at Checkpoint Epochs

| Epoch | Liver Dice | Tumor Dice | Train Loss | Val Loss |
|-------|-----------|-----------|-----------|---------|
| 0 | 0.706 | 0.000 | 0.184 | 0.114 |
| 9 | 0.861 | 0.301 | -0.226 | -0.231 |
| 50 | ~0.890 | ~0.350 | ~-0.450 | ~-0.300 |
| 150 | 0.905 | 0.429 | -0.610 | -0.317 |
| 251 | 0.890 | 0.376 | -0.610 | -0.317 |
| 407 | 0.928 | 0.372 | -0.644 | -0.424 |
| 441 | 0.889 | 0.300 | -0.619 | -0.216 |
| 593 | 0.932 | 0.394 | -0.644 | -0.424 |

---

## Infrastructure Issues Encountered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Training hung at `using pin_memory` | `nnUNet_n_proc_DA=4` + Python `fork` + CUDA = deadlock | Set `nnUNet_n_proc_DA=0` |
| SLURM log appeared silent | nnU-Net writes to its own log file, not stdout | Monitor `/scratch/.../training_log_*.txt` |
| Got T4 instead of A100 | SLURM allocates any available GPU | `--gres=gpu:a100:1` is a hint, not guaranteed |
| Training restarted from epoch 0 | `--c` flag missing from nnUNetv2_train call | Added `--c` flag to `train_nnunet.sh` |
| SLURM 8h time limit | Hard cluster limit on gpu partition | Resubmit with `--c`; nnU-Net auto-resumes |

---

## Files

| File | Location |
|------|---------|
| Training script | `scripts/train_nnunet.sh` |
| Smoke test script | `scripts/smoke_nnunet.sh` |
| Custom smoke trainer | `nnunet_smoke_trainer.py` |
| Checkpoints | `/scratch/dhumal.a/LiTS-UNets/nnunet/results/Dataset001_LiTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/` |
| Training logs | Same directory, `training_log_*.txt` |
