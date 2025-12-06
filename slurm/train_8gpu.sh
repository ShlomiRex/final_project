#!/bin/bash
#SBATCH --job-name=train_latent_gpt
#SBATCH --output=slurm/logs/train_%j.out
#SBATCH --error=slurm/logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=500G

# ==============================================================================
# Multi-GPU Training Script for LatentGPT (8×A100)
# ==============================================================================
#
# This script trains the autoregressive latent transformer using 8 A100 GPUs.
# Uses HuggingFace Accelerate for distributed training.
#
# Usage:
#   sbatch slurm/train_8gpu.sh [config_file]
#
# Example:
#   sbatch slurm/train_8gpu.sh configs/transformer_500m.yaml
#
# ==============================================================================

set -e

# Configuration
CONFIG_FILE="${1:-configs/base.yaml}"

# Create logs directory
mkdir -p slurm/logs

echo "=============================================="
echo "LatentGPT Training Job"
echo "=============================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Config: $CONFIG_FILE"
echo "=============================================="

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate latent-gpt 2>/dev/null || {
    echo "Conda environment 'latent-gpt' not found, using system Python"
    export PATH="${HOME}/.local/bin:$PATH"
}

# Set MLflow tracking URI (update with actual server node)
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"

# GPU diagnostics
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Navigate to project directory
cd "${HOME}/work/final_project"

# Launch training with Accelerate
echo "Launching training with 8 GPUs..."
echo ""

accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --mixed_precision=bf16 \
    scripts/train_latent_gpt.py \
    --config "$CONFIG_FILE"

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
