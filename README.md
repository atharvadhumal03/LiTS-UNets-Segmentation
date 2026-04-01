# LiTS-UNet-Segmentation

Progressive comparison of segmentation architectures on the LiTS (Liver Tumor Segmentation Challenge, MICCAI 2017) dataset.

## Stages

| Stage | Architecture | Status |
|-------|-------------|--------|
| 1 | Vanilla U-Net (baseline) | In Progress |
| 2 | U-Net + ResNet50 encoder | Pending |
| 3 | nnU-Net | Pending |

## Setup

### Environment

```bash
conda env create -f environment.yaml
conda activate lits-seg
```

On HPC, use `scripts/set.sh` instead.

### Data

Place LiTS NIfTI files under `data/raw/`. Expected layout:

```
data/raw/
  volume-0.nii.gz
  segmentation-0.nii.gz
  volume-1.nii.gz
  segmentation-1.nii.gz
  ...
```

### W&B

Copy `.env.example` to `.env` on HPC and fill in your credentials. Never commit `.env`.

## Execution

```bash
# 1. One-time HPC environment setup
bash scripts/set.sh

# 2. Preprocess NIfTI volumes into 2D slices
sbatch scripts/preprocess.sh

# 3. Smoke test — must pass before full training
bash scripts/smoke_test.sh

# 4a. Train with W&B logging
sbatch scripts/train_with_wb.sh

# 4b. Train without W&B (fallback)
sbatch scripts/train_without_wb.sh

# 5. Evaluate best checkpoint
sbatch scripts/evaluate.sh

# 6. Inference on new scans
sbatch scripts/inference.sh <input.nii.gz>
```

## Project Structure

```
configs/        Hyperparameter configs (one per stage)
docs/           Architecture notes, execution guide, data pipeline docs
src/            Python modules (dataset, preprocessing, training, evaluation)
models/         Model architecture definitions
scripts/        SLURM job scripts
data/           Raw NIfTI and processed 2D slices (local only)
checkpoints/    Saved model weights (local only)
logs/           SLURM job logs (local only)
notebooks/      Exploratory analysis
```

## Results

See [docs/results.md](docs/results.md).
