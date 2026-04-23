#!/bin/bash
#================================================================
# SLURM Job Script: LiTS nnU-Net Resume Smoke Test
# Northeastern University - Explorer Cluster
# Tests that --c correctly resumes from checkpoint_latest.pth
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
echo "LiTS nnU-Net Resume Smoke Test"
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

SMOKE_DIR=/scratch/dhumal.a/LiTS-UNets/nnunet/results/Dataset001_LiTS/nnUNetSmokeTrainer__nnUNetPlans__3d_fullres/fold_0

# --- Phase 1: fresh start, run 2 epochs ---
echo "========================================"
echo "PHASE 1: Fresh start (2 epochs)"
echo "========================================"

# Wipe any previous smoke test results so we start clean
rm -rf "$SMOKE_DIR"

~/.conda/envs/lits-seg/bin/nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetSmokeTrainer \
    -device cuda

echo ""
echo "Phase 1 complete. Checkpoint written:"
ls -lh "$SMOKE_DIR"/checkpoint_*.pth 2>/dev/null || echo "ERROR: No checkpoint found after phase 1"

PHASE1_LOG=$(ls -t "$SMOKE_DIR"/training_log_*.txt 2>/dev/null | head -1)
if [ -n "$PHASE1_LOG" ]; then
    echo ""
    echo "Phase 1 training log (last 10 lines):"
    tail -10 "$PHASE1_LOG"
fi

# --- Phase 2: resume with --c, should start from epoch 2 not 0 ---
echo ""
echo "========================================"
echo "PHASE 2: Resume with --c (should continue from epoch 2)"
echo "========================================"

~/.conda/envs/lits-seg/bin/nnUNetv2_train 1 3d_fullres 0 \
    -tr nnUNetSmokeTrainer \
    --c \
    -device cuda

echo ""
echo "Phase 2 complete."

PHASE2_LOG=$(ls -t "$SMOKE_DIR"/training_log_*.txt 2>/dev/null | head -1)
if [ -n "$PHASE2_LOG" ]; then
    echo ""
    echo "Full training log (should show epochs 0-1, then resume at 2):"
    cat "$PHASE2_LOG"
fi

echo ""
echo "========================================"
echo "RESUME CHECK:"
echo "If resume worked, log should show epoch_start != 0 in phase 2."
echo "If it restarted from 0, resume is broken."
echo "========================================"
echo "Smoke test done at $(date)"
echo "========================================"
