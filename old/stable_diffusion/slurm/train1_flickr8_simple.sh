#!/bin/bash
#SBATCH --job-name=train1_flickr8_2gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/train1_flickr8_2gpu_%j.out
#SBATCH --error=slurm/logs/train1_flickr8_2gpu_%j.err

# ============================================================================
# Multi-GPU Training Script - SIMPLIFIED VERSION
# ============================================================================

echo "=========================================="
echo "🚀 Starting Multi-GPU Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate environment
source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Check GPUs
echo "=========================================="
echo "🎮 GPU Status"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# Navigate to project
cd /home/doshlom4/work/final_project/notebooks

# Convert notebook (skip if already exists)
echo "=========================================="
echo "📝 Converting Notebook"
echo "=========================================="
if [ ! -f "train1_flickr8_script.py" ]; then
    jupyter nbconvert --to script train1_flickr8.ipynb --output train1_flickr8_script
    echo "✅ Created train1_flickr8_script.py"
else
    echo "ℹ️  Script already exists, skipping conversion"
fi
echo ""

# Launch training
echo "=========================================="
echo "🚀 Launching Multi-GPU Training with Accelerate"
echo "=========================================="
echo "Command: accelerate launch --multi_gpu --num_processes=2 train1_flickr8_script.py"
echo "=========================================="
echo ""

accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    train1_flickr8_script.py 2>&1 | tee -a slurm/logs/train1_flickr8_training_${SLURM_JOB_ID}.log

echo ""
echo "=========================================="
echo "✅ Training Completed!"
echo "=========================================="
echo "End time: $(date)"
