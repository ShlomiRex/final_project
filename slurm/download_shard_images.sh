#!/bin/bash
#SBATCH --job-name=download_shard_images
#SBATCH --output=slurm/logs/download_shard_images_%j.out
#SBATCH --error=slurm/logs/download_shard_images_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00

# Load modules or activate your environment if needed
# module load python/3.9
# source /home/doshlom4/work/final_project/.venv/bin/activate

source ~/.bashrc
conda activate img2dataset_venv

# Run the image download script
python laion-400m-download/download_shard_images.py

