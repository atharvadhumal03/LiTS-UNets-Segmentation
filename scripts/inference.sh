#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Inference
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=lits_infer
#SBATCH --output=logs/outputs/train_%j.out
#SBATCH --error=logs/errors/train_%j.err

echo "========================================"
echo "LiTS Inference"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3

# Activate environment
source activate lits-seg

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

INPUT_PATH="${1:?Usage: sbatch inference.sh <input.nii> [output.nii.gz] [checkpoint.pth] [config.yaml]}"
OUTPUT_PATH="${2:-${INPUT_PATH%.nii}_pred.nii.gz}"
CHECKPOINT="${3:-checkpoints/unet_baseline/best.pth}"
CONFIG="${4:-configs/unet_baseline.yaml}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Input:      ${INPUT_PATH}"
echo "Output:     ${OUTPUT_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config:     ${CONFIG}"
echo "========================================"

echo "Starting inference..."

python src/inference.py \
    --config     "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --input      "${INPUT_PATH}" \
    --output     "${OUTPUT_PATH}"

echo ""
echo "========================================"
echo "Inference finished at $(date)"
echo "========================================"
