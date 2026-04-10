#!/bin/bash
#================================================================
# SLURM Job Script: LiTS nnU-Net Prediction + Evaluation
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=lits_nnunet_pred
#SBATCH --output=/home/dhumal.a/LiTS-UNets/logs/outputs/nnunet_pred_%j.out
#SBATCH --error=/home/dhumal.a/LiTS-UNets/logs/errors/nnunet_pred_%j.err

echo "========================================"
echo "LiTS nnU-Net Prediction + Evaluation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load cuda/12.1.1
module load anaconda3/2024.06

PYTHON=~/.conda/envs/lits-seg/bin/python

echo ""
echo "GPU Info:"
nvidia-smi
echo ""

cd /home/dhumal.a/LiTS-UNets

export nnUNet_raw=/scratch/dhumal.a/LiTS-UNets/nnunet/raw
export nnUNet_preprocessed=/scratch/dhumal.a/LiTS-UNets/nnunet/preprocessed
export nnUNet_results=/scratch/dhumal.a/LiTS-UNets/nnunet/results

INPUT_DIR=$nnUNet_raw/Dataset001_LiTS/imagesTs
OUTPUT_DIR=/scratch/dhumal.a/LiTS-UNets/nnunet/predictions

mkdir -p $OUTPUT_DIR

echo "[1/2] Running predictions on test set..."
~/.conda/envs/lits-seg/bin/nnUNetv2_predict \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -d 1 \
    -c 3d_fullres \
    -f 0 \
    --save_probabilities

echo "[2/2] Computing Dice and IoU on test set..."
$PYTHON src/evaluate_nnunet.py --config configs/nnunet.yaml --predictions $OUTPUT_DIR

echo ""
echo "========================================"
echo "Prediction and evaluation finished at $(date)"
echo "========================================"
