#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Training (no W&B)
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=lits_train
#SBATCH --output=logs/outputs/train_%j.out
#SBATCH --error=logs/errors/train_%j.err

echo "========================================"
echo "LiTS Training - Vanilla U-Net (no W&B)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3/2024.06

# Activate environment
PYTHON=~/.conda/envs/lits-seg/bin/python

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

mkdir -p logs/outputs logs/errors checkpoints

CONFIG="${1:-configs/unet_baseline.yaml}"
RESUME="${2:-}"

echo "Config: ${CONFIG}"
echo "========================================"

echo "Starting training..."

if [ -n "${RESUME}" ]; then
    echo "Resuming from: ${RESUME}"
    $PYTHON src/train.py --config "${CONFIG}" --resume "${RESUME}"
else
    $PYTHON src/train.py --config "${CONFIG}"
fi

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"
