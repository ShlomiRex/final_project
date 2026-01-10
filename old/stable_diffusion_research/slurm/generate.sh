#!/bin/bash
#SBATCH --job-name=sd-generate
#SBATCH --partition=gpu6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/generate_%j.out
#SBATCH --error=slurm/logs/generate_%j.err

# ============================================================================
# Image Generation Script
# ============================================================================
#
# Usage:
#   # Generate from prompts file
#   sbatch slurm/generate.sh outputs/checkpoints/checkpoint_100000.pt prompts.txt
#
#   # Generate with custom settings
#   sbatch slurm/generate.sh outputs/checkpoints/checkpoint_100000.pt prompts.txt 50 7.5
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

cd /home/doshlom4/work/final_project/stable_diffusion_research

mkdir -p slurm/logs

CHECKPOINT=${1:-"outputs/checkpoints/latest.pt"}
PROMPTS_FILE=${2:-"prompts.txt"}
NUM_STEPS=${3:-50}
GUIDANCE=${4:-7.5}

echo "Checkpoint: $CHECKPOINT"
echo "Prompts: $PROMPTS_FILE"
echo "Steps: $NUM_STEPS"
echo "Guidance: $GUIDANCE"
echo ""

OUTPUT_DIR="generated_images/gen_$(date +%Y%m%d_%H%M%S)"

python scripts/generate.py \
    --checkpoint "$CHECKPOINT" \
    --config configs/base.yaml \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_images_per_prompt 4 \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE" \
    --use_ema

echo "=============================================="
echo "Generation complete!"
echo "Output: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=============================================="
