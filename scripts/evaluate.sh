#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Evaluation
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=lits_eval
#SBATCH --output=logs/outputs/train_%j.out
#SBATCH --error=logs/errors/train_%j.err

echo "========================================"
echo "LiTS Evaluation"
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

CONFIG="${1:-configs/unet_baseline.yaml}"
CHECKPOINT="${2:-checkpoints/unet_baseline/best.pth}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "========================================"

echo "Starting evaluation..."

$PYTHON src/train.py --config "${CONFIG}" --mode evaluate --checkpoint "${CHECKPOINT}"

echo ""
echo "========================================"
echo "Evaluation finished at $(date)"
echo "========================================"
