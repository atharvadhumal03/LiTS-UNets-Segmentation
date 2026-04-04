#!/bin/bash
#================================================================
# SLURM Job Script: LiTS Preprocessing
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=lits_preprocess
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/preprocess_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/preprocess_%j.err

echo "========================================"
echo "LiTS Preprocessing"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3/2024.06

# Navigate to project directory
cd /home/dhumal.a/LiTS-UNets

mkdir -p logs/outputs logs/errors data/processed

echo "Starting preprocessing..."
echo "========================================"

~/.conda/envs/lits-seg/bin/python src/preprocess.py --config configs/unet_baseline.yaml

echo ""
echo "========================================"
echo "Preprocessing finished at $(date)"
echo "========================================"
