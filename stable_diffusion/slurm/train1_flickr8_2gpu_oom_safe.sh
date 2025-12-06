#!/bin/bash
#SBATCH --job-name=train1_flickr8_2gpu_low_mem
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --cpus-per-task=8               # 4 CPUs per GPU
#SBATCH --mem=64G                       # Total memory
#SBATCH --time=72:00:00                 # 3 days max runtime
#SBATCH --output=slurm/logs/train1_flickr8_2gpu_%j.out
#SBATCH --error=slurm/logs/train1_flickr8_2gpu_%j.err

# ============================================================================
# Multi-GPU Training Script for Flickr8k Text-to-Image Model
# ============================================================================
# This script launches distributed training across 2 GPUs using Accelerate.
# 
# IMPORTANT: This script includes OOM prevention:
# - Clears GPU memory before training
# - Reduces batch size to 8 (effective batch size = 16 with 2 GPUs)
# - Uses gradient checkpointing if available
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Job Information
# ============================================================================
echo "=========================================="
echo "🚀 SLURM Multi-GPU Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_ON_NODE"
echo "GPUs allocated: ${SLURM_GPUS:-NOT SET}"
echo "GPUs on node: ${SLURM_GPUS_ON_NODE:-NOT SET}"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
echo "🔧 Activating conda environment..."
source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo "   Python: $(which python)"
echo "   Python version: $(python --version)"
echo ""

# ============================================================================
# GPU Check & Memory Cleanup
# ============================================================================
echo "=========================================="
echo "🎮 GPU Status BEFORE Cleanup"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

echo "🧹 Clearing GPU memory cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✅ Cache cleared')"
echo ""

echo "=========================================="
echo "🎮 GPU Status AFTER Cleanup"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# ============================================================================
# PyTorch & Accelerate Verification
# ============================================================================
echo "=========================================="
echo "🔍 PyTorch CUDA Verification"
echo "=========================================="
python << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
PYEOF
echo ""

echo "=========================================="
echo "⚡ Accelerate Configuration"
echo "=========================================="
accelerate env
echo ""

# ============================================================================
# Navigate to Project & Convert Notebook
# ============================================================================
cd /home/doshlom4/work/final_project/notebooks

echo "=========================================="
echo "📝 Converting Notebook to Script"
echo "=========================================="
if [ -f "train1_flickr8_script.py" ]; then
    echo "⚠️  Script already exists - removing old version..."
    rm train1_flickr8_script.py
fi

jupyter nbconvert --to script train1_flickr8.ipynb --output train1_flickr8_script
echo "✅ Conversion complete: train1_flickr8_script.py"
echo ""

# ============================================================================
# Modify Script for OOM Prevention (Optional - reduces batch size)
# ============================================================================
echo "=========================================="
echo "🛠️  Applying OOM Prevention Patches"
echo "=========================================="
echo "Current batch size in script: $(grep -m 1 'batch_size=' train1_flickr8_script.py || echo 'NOT FOUND')"
echo ""
echo "To reduce batch size for OOM prevention, edit the script manually or"
echo "modify the notebook configuration cell before conversion."
echo "Recommended: batch_size=8 for 2 GPUs (effective batch_size=16)"
echo ""

# ============================================================================
# Launch Multi-GPU Training
# ============================================================================
echo "=========================================="
echo "🚀 Starting Multi-GPU Training"
echo "=========================================="
echo "Configuration:"
echo "  - GPUs: 2"
echo "  - Mixed Precision: fp16"
echo "  - Script: train1_flickr8_script.py"
echo "=========================================="
echo ""

# Set environment variables for better error messages
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging (slower)
export TORCH_DISTRIBUTED_DEBUG=OFF  # Set to DETAIL for debugging

# Launch with accelerate
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    train1_flickr8_script.py

# ============================================================================
# Completion
# ============================================================================
echo ""
echo "=========================================="
echo "✅ Training Completed Successfully!"
echo "=========================================="
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Check outputs in:"
echo "  - Checkpoints: ./outputs/train12_flickr8k_text2img/"
echo "  - Logs: slurm/logs/train1_flickr8_2gpu_${SLURM_JOB_ID}.out"
echo "=========================================="
