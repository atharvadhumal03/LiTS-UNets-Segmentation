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

CONFIG="${1:-configs/unet_baseline.yaml}"
CHECKPOINT="${2:-checkpoints/unet_baseline/best.pth}"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

echo "Running evaluation..."
echo "  Config:     ${CONFIG}"
echo "  Checkpoint: ${CHECKPOINT}"

python src/evaluate.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
