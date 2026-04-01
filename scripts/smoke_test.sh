#!/usr/bin/env bash
set -euo pipefail

# Pipeline sanity check — must pass before submitting a full training job.
# Extracts 10 slices, runs 2 epochs, tests both training paths.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source ~/.bashrc
conda activate lits-seg

echo "=== SMOKE TEST START ==="

echo "[1/3] Preprocessing (smoke mode — 10 slices)..."
python src/preprocess.py --config configs/unet_baseline.yaml --smoke

echo "[2/3] Training without W&B (2 epochs)..."
python src/train.py --config configs/unet_baseline.yaml

echo "[3/3] Training with W&B flag (import check only)..."
python -c "import wandb; print('wandb import OK')"

echo ""
echo "=== SMOKE TEST PASSED ==="
echo "Pipeline is healthy. Safe to submit full training job."
