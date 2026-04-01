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
RESUME="${2:-}"

mkdir -p logs/outputs logs/errors checkpoints

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Create it from .env.example before running with W&B." >&2
    exit 1
fi

echo "Starting training (with W&B)..."
echo "  Config: ${CONFIG}"

if [ -n "${RESUME}" ]; then
    echo "  Resuming from: ${RESUME}"
    python src/train.py --config "${CONFIG}" --wandb --resume "${RESUME}"
else
    python src/train.py --config "${CONFIG}" --wandb
fi
