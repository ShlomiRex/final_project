#!/bin/bash
#SBATCH --job-name=laion_parquet_sharding
#SBATCH --output=slurm/logs/laion_parquet_sharding_%A_%a.out
#SBATCH --error=slurm/logs/laion_parquet_sharding_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Activate your conda or virtual environment
source ~/.bashrc
conda activate img2dataset_venv  # or use 'source path/to/venv/bin/activate'

python laion-400m-download/parquet_sharding.py
