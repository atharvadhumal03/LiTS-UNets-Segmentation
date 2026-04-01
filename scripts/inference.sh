#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/outputs/train_%j.out
#SBATCH --error=logs/errors/train_%j.err

set -euo pipefail

source ~/.bashrc
conda activate lits-seg

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT_PATH="${1:?Usage: inference.sh <input.nii> [output.nii.gz] [checkpoint.pth] [config.yaml]}"
OUTPUT_PATH="${2:-${INPUT_PATH%.nii}_pred.nii.gz}"
CHECKPOINT="${3:-checkpoints/unet_baseline/best.pth}"
CONFIG="${4:-configs/unet_baseline.yaml}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

echo "Running inference..."
echo "  Input:      ${INPUT_PATH}"
echo "  Output:     ${OUTPUT_PATH}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Config:     ${CONFIG}"

python src/inference.py \
    --config     "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --input      "${INPUT_PATH}" \
    --output     "${OUTPUT_PATH}"
