#!/bin/bash
#SBATCH --job-name=train1_flickr8_multi_gpu
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --cpus-per-task=8               # 4 CPUs per GPU
#SBATCH --mem=64G                       # 32GB per GPU
#SBATCH --time=72:00:00                 # 3 days max runtime
#SBATCH --output=slurm/logs/train1_flickr8_multi_gpu_%j.out
#SBATCH --error=slurm/logs/train1_flickr8_multi_gpu_%j.err

# Handle node selection from command-line argument
# Usage: sbatch slurm/train1_flickr8_multi_gpu.sh [node_name]
# Example: sbatch slurm/train1_flickr8_multi_gpu.sh gpu8
if [ -n "$1" ]; then
    export SBATCH_NODELIST="$1"
    echo "Node constraint set to: $1"
    # Re-submit with nodelist constraint
    sbatch --nodelist="$1" "$0"
    exit 0
fi

# Print job information
echo "=========================================="
echo "SLURM Job Information"
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

# Activate conda environment
echo "Activating conda environment..."
source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025

# Verify environment
echo "=========================================="
echo "Environment Verification"
echo "=========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check GPU availability
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""

# Check PyTorch CUDA
echo "=========================================="
echo "PyTorch CUDA Check"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Check Accelerate installation
echo "=========================================="
echo "Accelerate Configuration"
echo "=========================================="
accelerate env
echo ""

# Navigate to project directory
cd /home/doshlom4/work/final_project/notebooks

# Convert notebook to script (ALWAYS reconvert to get latest changes)
echo "=========================================="
echo "Converting notebook to script..."
echo "=========================================="
# Remove old script to force reconversion
rm -f train1_flickr8_script.py
jupyter nbconvert --to script train1_flickr8.ipynb --output train1_flickr8_script
echo "✅ Conversion complete"
echo ""

# Clean up problematic execution cells from converted script
echo "=========================================="
echo "Cleaning up execution cells..."
echo "=========================================="
echo "✅ No cleanup needed - all execution cells are commented in notebook"
echo ""

# Run training with accelerate launch
echo "=========================================="
echo "Starting Multi-GPU Training"
echo "=========================================="
echo "Command: accelerate launch --multi_gpu --num_processes=2 train1_flickr8_script.py"
echo "=========================================="
echo ""

# Launch training with proper error handling
set -e  # Exit on error
accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=fp16 \
    train1_flickr8_script.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
