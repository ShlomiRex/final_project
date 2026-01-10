#!/bin/bash
#SBATCH --job-name=jupyter_lab_server
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --cpus-per-task=4   # Request CPU cores
#SBATCH --mem=32G           # Request GB of memory
#SBATCH --time=48:00:00     # Add a time limit. Jobs are killed after this.
#SBATCH --output=slurm/logs/jupyter_job_%j.out # Use %j for job ID in filename
#SBATCH --error=slurm/logs/jupyter_job_%j.err  # Use %j for job ID in filename
#SBATCH --cpus-per-task=8   # CPU cores


# Activate your virtual environment
source /home/doshlom4/torch114/bin/activate

# Diagnostic info (optional)
echo "Using Python from:"
which python
python -c "import sys; print(sys.executable)"

# Start Jupyter Lab with no authentication (since we're using SSH tunneling for security)
jupyter lab --no-browser --ip=0.0.0.0 --port=8895