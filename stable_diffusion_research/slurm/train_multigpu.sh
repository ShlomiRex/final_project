#!/bin/bash
#SBATCH --job-name=sd-train
#SBATCH --partition=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train_%j.out
#SBATCH --error=slurm/logs/train_%j.err

# ============================================================================
# Multi-GPU Training Script for Stable Diffusion
# ============================================================================
# 
# Usage:
#   sbatch slurm/train_multigpu.sh configs/base.yaml
#   sbatch slurm/train_multigpu.sh configs/training/train_512.yaml
#
# To resume from checkpoint:
#   sbatch slurm/train_multigpu.sh configs/base.yaml --resume latest
#
# To use different number of GPUs, edit --gres=gpu:N above (2, 4, or 8)
# ============================================================================

set -e

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config: $1"
echo "=============================================="

# Load environment
source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate

# Change to project directory
cd /home/doshlom4/work/final_project/stable_diffusion_research

# Create logs directory
mkdir -p slurm/logs

# Configuration file (passed as argument)
CONFIG=${1:-configs/base.yaml}
shift || true  # Remove first argument, rest are passed to script

# Set up distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Get number of GPUs from Slurm allocation
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "Number of GPUs allocated: $NUM_GPUS"

# Print GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set MLflow tracking URI if server is running
# Update this to your MLflow server node
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://127.0.0.1:5000"}
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"

# Launch training with Accelerate using config file
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=$NUM_GPUS \
    scripts/train.py \
    --config "$CONFIG" \
    "$@"

echo "=============================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=============================================="
