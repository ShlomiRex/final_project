#!/bin/bash
#SBATCH --job-name=train7
#SBATCH --gres=gpu:1        
#SBATCH --cpus-per-task=4   
#SBATCH --mem=32G           
#SBATCH --time=48:00:00     
#SBATCH --output=slurm/logs/train7_%j.out 
#SBATCH --error=slurm/logs/train7_%j.err  

conda init bash
source ~/.bashrc
conda activate mycondaenv


# Verify the correct environment is active
echo "Using Python from:"
which python
python -c "import sys; print(sys.executable)"

# Check PyTorch CUDA compatibility
echo "Checking PyTorch CUDA compatibility:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}')"

pip list | grep torch

# Run your training script
# python /home/doshlom4/work/final_project/notebooks/complete_new_model/diffusers/train7.py