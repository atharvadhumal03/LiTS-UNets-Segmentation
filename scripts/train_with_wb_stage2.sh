#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Stage 2 Training (with W&B)
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=lits_s2_train_wb
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/train_s2_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/train_s2_%j.err

echo "========================================"
echo "LiTS Stage 2 Training - U-Net + ResNet50 (with W&B)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load cuda/12.1.1
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

mkdir -p logs/outputs logs/errors /scratch/dhumal.a/LiTS-UNets/checkpoints/unet_resnet50

CONFIG="${1:-configs/unet_resnet50.yaml}"
RESUME="${2:-}"

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Create it from .env.example before running with W&B."
    exit 1
fi

echo "Config: ${CONFIG}"
echo "========================================"
echo "Starting training..."

if [ -n "${RESUME}" ]; then
    echo "Resuming from: ${RESUME}"
    $PYTHON src/train_stage2.py --config "${CONFIG}" --wandb --resume "${RESUME}"
else
    $PYTHON src/train_stage2.py --config "${CONFIG}" --wandb
fi

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"
