#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Stage 2 Smoke Test
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:15:00
#SBATCH --job-name=lits_s2_smoke
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/smoke_s2_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/smoke_s2_%j.err

echo "========================================"
echo "LiTS Stage 2 Smoke Test - U-Net + ResNet50"
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

echo "=== STAGE 2 SMOKE TEST START ==="
echo "========================================"

echo "[1/2] Preprocessing (smoke mode — 10 slices)..."
$PYTHON src/preprocess.py --config configs/smoke_test_stage2.yaml --smoke

echo "[2/2] Training with W&B (2 epochs)..."
$PYTHON src/train_stage2.py --config configs/smoke_test_stage2.yaml --wandb

echo ""
echo "========================================"
echo "=== STAGE 2 SMOKE TEST PASSED ==="
echo "Pipeline is healthy. Safe to submit full Stage 2 training."
echo "========================================"
echo "Smoke test finished at $(date)"
echo "========================================"
