#!/bin/bash
#SBATCH --job-name=sd-train-2gpu
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train_2gpu_%j.out
#SBATCH --error=slurm/logs/train_2gpu_%j.err

# ============================================================================
# 2-GPU Training Script for Stable Diffusion (Optimized)
# ============================================================================
#
# This script is optimized for 2x 32GB GPUs with batch_size=64
# (32 samples per GPU after splitting)
#
# Usage:
#   sbatch slurm/train_2gpu.sh configs/base.yaml
#
# To resume from checkpoint:
#   sbatch slurm/train_2gpu.sh configs/base.yaml --resume latest
#
# ============================================================================

set -e

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: $1"
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

CONFIG=${1:-configs/base.yaml}  # Default to base config
shift || true

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=2
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Log environment details
echo "Environment details:" > slurm/logs/env_details_$SLURM_JOB_ID.log
conda info >> slurm/logs/env_details_$SLURM_JOB_ID.log
conda list >> slurm/logs/env_details_$SLURM_JOB_ID.log

# Log GPU details
nvidia-smi --query-gpu=name,memory.total --format=csv >> slurm/logs/env_details_$SLURM_JOB_ID.log

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://127.0.0.1:5000"}

accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision=bf16 \
    scripts/train.py \
    --config "$CONFIG" \
    "$@"

echo "=============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=============================================="
