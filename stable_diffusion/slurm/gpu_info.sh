#!/bin/bash
#SBATCH --job-name=gpu_info
#SBATCH --output=slurm/logs/gpu_info_%j.out
#SBATCH --error=slurm/logs/gpu_info_%j.err
#SBATCH --gres=gpu:2           # Request 1 GPU
#SBATCH --time=00:05:00        # Short job, 5 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# Load CUDA module if needed (only if required by your system)
# module load cuda/11.4

echo "Running nvidia-smi to check GPU information..."
nvidia-smi

echo -e "\nExtracted Info:"
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader