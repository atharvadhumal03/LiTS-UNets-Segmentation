#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Smoke Test
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:15:00
#SBATCH --job-name=lits_smoke
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/smoke_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/smoke_%j.err

echo "========================================"
echo "LiTS Smoke Test"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3/2024.06

# Python from conda env
PYTHON=~/.conda/envs/lits-seg/bin/python

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd /home/dhumal.a/LiTS-UNets

echo "=== SMOKE TEST START ==="
echo "========================================"

echo "[1/3] Preprocessing (smoke mode — 10 slices)..."
$PYTHON src/preprocess.py --config configs/smoke_test.yaml --smoke

echo "[2/3] Training with W&B (2 epochs)..."
$PYTHON src/train.py --config configs/smoke_test.yaml --wandb

echo "[3/3] W&B import check..."
$PYTHON -c "import wandb; print('wandb import OK')"

echo ""
echo "========================================"
echo "=== SMOKE TEST PASSED ==="
echo "Pipeline is healthy. Safe to submit full training job."
echo "========================================"
echo "Smoke test finished at $(date)"
echo "========================================"
