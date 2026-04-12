#!/bin/bash
#================================================================
# SLURM Job Script: LiTS nnU-Net 3D Full-Res Training
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --job-name=lits_nnunet_train
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/nnunet_train_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/nnunet_train_%j.err

echo "========================================"
echo "LiTS nnU-Net 3D Full-Res Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load cuda/12.1.1
module load anaconda3/2024.06

echo ""
echo "GPU Info:"
nvidia-smi
echo ""

cd /home/dhumal.a/LiTS-UNets

# Set nnU-Net environment variables
export nnUNet_raw=/scratch/dhumal.a/LiTS-UNets/nnunet/raw
export nnUNet_preprocessed=/scratch/dhumal.a/LiTS-UNets/nnunet/preprocessed
export nnUNet_results=/scratch/dhumal.a/LiTS-UNets/nnunet/results

echo "Starting nnU-Net 3D full-res training (fold 0)..."
echo "========================================"

# Disable torch.compile — takes hours to compile on first run
export nnUNet_compile=0

~/.conda/envs/lits-seg/bin/nnUNetv2_train 1 3d_fullres 0 \
    --npz \
    -device cuda

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"
