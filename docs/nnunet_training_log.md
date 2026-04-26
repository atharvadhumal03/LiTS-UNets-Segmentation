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
| 1 | Apr 12 | 5877824 | V100 | 0 → 558 | 559 | [0.706, 0.000] | [0.910, 0.506] | -0.6604 | -0.4220 |
| 2 | Apr 12 | 5884642 | V100 | 0 → 451 | 452 | [0.677, 0.000] | [0.919, 0.373] | -0.6353 | -0.3642 |
| 3 | Apr 13 | 5911321 | T4 | 0 → 3 | 4 | [0.650, 0.000] | [0.828, 0.000] | -0.0376 | -0.1215 |
| 4 | Apr 16 | 5965867 | A100 | 0 → 188 | 189 | [0.631, 0.000] | [0.912, 0.341] | -0.5330 | -0.3684 |
| 5 | Apr 17 | 5984599 | A100 | 0 → 192 | 193 | [0.646, 0.000] | [0.885, 0.373] | -0.5800 | -0.2742 |
| 6 | Apr 17 | 6055395 | A100 | 0 → 179 | 180 | [0.661, 0.000] | [0.869, 0.419] | -0.5420 | -0.2943 |
| 7 | Apr 18 | 6192681 | A100 | 150 → 251 | 102 | [0.905, 0.429] | [0.890, 0.376] | -0.6103 | -0.3167 |
| 8 | Apr 21 | 6268167 | A100 | 250 → 441 | 192 | [0.906, 0.387] | [0.889, 0.300] | -0.6185 | -0.2160 |
| 9 | Apr 22 | 6275563 | A100 | 400 → 593 | 194 | [0.928, 0.372] | [0.932, 0.394] | -0.6444 | -0.4237 |
| 10 | Apr 24 | 6331827 | A100 | 700 → 885 | 186 | [0.903, 0.476] | [0.927, 0.493] | -0.703 | -0.377 |
| 11 | Apr 25 | 6334713 | A100 | 880 → 999 | 120 | [0.938, 0.315] | [0.950, 0.534] | -0.691 | -0.536 |

---

## Notes

**Runs 1–6 (failed to resume):**
Runs 1–6 all restarted from epoch 0 on each resubmission. Root causes:
- Runs 1–2: `nnUNet_n_proc_DA=4` on HPC caused a fork+CUDA multiprocessing deadlock — training never started
- Run 3: Allocated a T4 (15 GB) instead of A100 — insufficient VRAM, manually cancelled
- Runs 4–6: Training ran correctly but the `--c` (resume) flag was missing from the script — nnU-Net restarted from scratch each time

**Runs 7+ (resuming correctly):**
`--c` flag added in run 7. nnU-Net now resumes from `checkpoint_latest.pth` on each resubmission.

**Run 10 notes:**
- Resumed from epoch 700 (last `save_every=10` checkpoint); epochs 701–741 from run 9 were lost because `checkpoint_latest.pth` saves every 10 epochs
- Previous job (6310036) was cancelled — allocated node d1029 which hung silently for 3h without starting
- Fix: added `--exclude=d1029` to `train_nnunet.sh`

**Current canonical checkpoint:** epoch 999 (run 11, complete)
**Epochs remaining:** 0 — training complete

---

## Effective Training Progress

Discarding restarts, the canonical training sequence is:

| Segment | Epochs | Best Dice (Tumor) |
|---------|--------|-------------------|
| 0 – 192 | Run 5 (cleanest early run) | 0.373 |
| 150 – 251 | Run 7 (first resume) | 0.376 |
| 250 – 441 | Run 8 | 0.300 |
| 400 – 593 | Run 9 | 0.394 |
| 700 – 885 | Run 10 | 0.493 |
| 880 – 999 | Run 11 | 0.534 |

**Training complete: epoch 999 / 1000. Final Tumor Dice: 0.534, Liver Dice: 0.950.**
**Test set results (20 cases): Dice Liver 0.9481 / Tumor 0.6777 — IoU Liver 0.9034 / Tumor 0.5644.**

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
| 700 | 0.903 | 0.476 | -0.644 | -0.414 |
| 782 | 0.934 | 0.237 | -0.676 | -0.259 |
| 783 | 0.919 | 0.431 | -0.680 | -0.372 |
| 885 | 0.927 | 0.493 | -0.703 | -0.377 |
| 999 | 0.950 | 0.534 | -0.691 | -0.536 |

---

## Infrastructure Issues Encountered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Training hung at `using pin_memory` | `nnUNet_n_proc_DA=4` + Python `fork` + CUDA = deadlock | Set `nnUNet_n_proc_DA=0` |
| SLURM log appeared silent | nnU-Net writes to its own log file, not stdout | Monitor `/scratch/.../training_log_*.txt` |
| Got T4 instead of A100 | SLURM allocates any available GPU | `--gres=gpu:a100:1` is a hint, not guaranteed |
| Training restarted from epoch 0 | `--c` flag missing from nnUNetv2_train call | Added `--c` flag to `train_nnunet.sh` |
| SLURM 8h time limit | Hard cluster limit on gpu partition | Resubmit with `--c`; nnU-Net auto-resumes |
| Node d1029 hung silently for 3h | Bad compute node — no errors, no training | Added `--exclude=d1029` to `train_nnunet.sh` |
| Lost epochs 701–741 on resume | `save_every=10` means checkpoint_latest is always a multiple of 10 | Acceptable tradeoff; max loss per timeout is 9 epochs |

---

## Files

| File | Location |
|------|---------|
| Training script | `scripts/train_nnunet.sh` |
| Smoke test script | `scripts/smoke_nnunet.sh` |
| Custom smoke trainer | `models/nnunet_smoke_trainer.py` |
| Checkpoints | `/scratch/dhumal.a/LiTS-UNets/nnunet/results/Dataset001_LiTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/` |
| Training logs | Same directory, `training_log_*.txt` |
