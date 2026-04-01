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

mkdir -p logs/outputs logs/errors data/processed

echo "Starting preprocessing..."
python src/preprocess.py --config configs/unet_baseline.yaml
echo "Preprocessing complete."
