#!/bin/bash
#================================================================
# SLURM Job Script: LiTS nnU-Net Data Preparation
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=lits_nnunet_prep
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/nnunet_prep_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/nnunet_prep_%j.err

echo "========================================"
echo "LiTS nnU-Net Data Preparation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load anaconda3/2024.06

PYTHON=~/.conda/envs/lits-seg/bin/python

cd /home/dhumal.a/LiTS-UNets

# Set nnU-Net environment variables
export nnUNet_raw=/scratch/dhumal.a/LiTS-UNets/nnunet/raw
export nnUNet_preprocessed=/scratch/dhumal.a/LiTS-UNets/nnunet/preprocessed
export nnUNet_results=/scratch/dhumal.a/LiTS-UNets/nnunet/results

mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

echo "[1/2] Converting data to nnU-Net format..."
$PYTHON src/prepare_nnunet.py --config configs/nnunet.yaml

echo "[2/2] Running nnU-Net fingerprinting and preprocessing..."
~/.conda/envs/lits-seg/bin/nnUNetv2_plan_and_preprocess \
    -d 1 \
    --verify_dataset_integrity \
    -c 3d_fullres

echo ""
echo "========================================"
echo "Preparation finished at $(date)"
echo "========================================"
