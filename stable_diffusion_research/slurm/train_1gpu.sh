#!/bin/bash
#SBATCH --job-name=sd-train-1gpu
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/train_1gpu_%j.out
#SBATCH --error=slurm/logs/train_1gpu_%j.err

# ============================================================================
# 1-GPU Training Script for Stable Diffusion (Testing)
# ============================================================================

set -e

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Show GPU info
nvidia-smi
echo ""

# Correct conda activation
module load Anaconda3/2022.05
source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate py312

# Change to project directory
cd /home/doshlom4/work/final_project/stable_diffusion_research

mkdir -p slurm/logs

CONFIG=${1:-configs/base.yaml}
shift || true

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://127.0.0.1:5000"}

# Single GPU - no accelerate launch needed
python scripts/train.py \
    --config "$CONFIG" \
    "$@"

echo "=============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=============================================="
