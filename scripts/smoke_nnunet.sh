#!/bin/bash
#================================================================
# SLURM Job Script: LiTS nnU-Net Smoke Test (3 epochs)
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --job-name=lits_nnunet_smoke
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/nnunet_smoke_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/nnunet_smoke_%j.err

echo "========================================"
echo "LiTS nnU-Net Smoke Test (3 epochs)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load cuda/12.1.1
module load anaconda3/2024.06

echo "GPU Info:"
nvidia-smi
echo ""

cd /home/dhumal.a/LiTS-UNets

export nnUNet_raw=/scratch/dhumal.a/LiTS-UNets/nnunet/raw
export nnUNet_preprocessed=/scratch/dhumal.a/LiTS-UNets/nnunet/preprocessed
export nnUNet_results=/scratch/dhumal.a/LiTS-UNets/nnunet/results

export nnUNet_compile=0
export nnUNet_n_proc_DA=0

echo "Starting smoke test (3 epochs)..."
echo "========================================"

~/.conda/envs/lits-seg/bin/nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetSmokeTrainer \
    -device cuda

echo ""
echo "========================================"
echo "Smoke test done at $(date)"
echo ""
echo "Training log:"
cat /scratch/dhumal.a/LiTS-UNets/nnunet/results/Dataset001_LiTS/nnUNetSmokeTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt 2>/dev/null || echo "No training log found."
echo "========================================"
