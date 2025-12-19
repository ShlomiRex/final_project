#!/bin/bash
#SBATCH --job-name=sd-eval
#SBATCH --partition=gpu6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err

# ============================================================================
# Evaluation Script for Stable Diffusion
# ============================================================================
#
# Usage:
#   sbatch slurm/evaluate.sh outputs/checkpoints/checkpoint_100000.pt
#
# With custom config:
#   sbatch slurm/evaluate.sh outputs/checkpoints/checkpoint_100000.pt configs/base.yaml
#
# ============================================================================

set -e

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Load environment
source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate

# Change to project directory
cd /home/doshlom4/work/final_project/stable_diffusion_research

mkdir -p slurm/logs

CHECKPOINT=${1:-"outputs/checkpoints/latest.pt"}
CONFIG=${2:-"configs/base.yaml"}

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo ""

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run evaluation
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir "evaluation_results/eval_$(date +%Y%m%d_%H%M%S)" \
    --num_samples 1000 \
    --batch_size 8 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --use_ema

echo "=============================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo "=============================================="
