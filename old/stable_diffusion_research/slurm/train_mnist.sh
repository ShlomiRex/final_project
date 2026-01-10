#!/bin/bash
#SBATCH --job-name=mnist_diffusion
#SBATCH --output=slurm/logs/mnist_diffusion_%j.out
#SBATCH --error=slurm/logs/mnist_diffusion_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=16G
# NOTE: Add a constraint on submission if needed, e.g.:
# sbatch --constraint=gpu32g slurm/train_mnist.sh
# (Removed here to avoid invalid feature errors on clusters without this feature)

# ============================================================================
# MNIST Diffusion Model Training on HPC
# ============================================================================
# This script trains the MNIST text-conditioned diffusion model on 2 GPUs
#
# Usage:
#   sbatch slurm/train_mnist.sh
#   sbatch slurm/train_mnist.sh -N 2 -G 4  # Override nodes/GPUs
#
# To monitor:
#   squeue -u $USER
#   tail -f slurm/logs/mnist_diffusion_<job_id>.out

set -e

echo "=== MNIST Diffusion Model Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Create logs directory
mkdir -p slurm/logs

# Activate conda environment (HPC shared Anaconda location)
if [ -f "/prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh" ]; then
    source "/prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
else
    echo "[ERROR] Could not find conda.sh at /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh" >&2
    echo "Please load the Anaconda module or update this path." >&2
    exit 1
fi

conda activate shlomid_conda_12_11_2025

# Check environment
echo "Python: $(which python)"
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPUs Available: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Navigate to project directory
cd /home/doshlom4/work/final_project/stable_diffusion_research

# Run training with Accelerate on 2 GPUs
echo "Starting training on $SLURM_GPUS_PER_NODE GPUs..."
echo ""

accelerate launch \
    --num_processes=$SLURM_GPUS_PER_NODE \
    --multi_gpu \
    --mixed_precision=bf16 \
    scripts/train_mnist.py \
    --num_epochs=10 \
    --batch_size=512 \
    --learning_rate=1e-3 \
    --num_inference_steps=50 \
    --guidance_scale=8.0 \
    --output_dir=outputs/mnist_hpc \
    --num_samples_per_epoch=2

echo ""
echo "=== Training Complete ==="
echo "Checkpoints: outputs/mnist_hpc/checkpoints/"
echo "Samples: outputs/mnist_hpc/samples/"
