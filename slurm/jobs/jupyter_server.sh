#!/bin/bash
#SBATCH --job-name=jupyter_server
#SBATCH --output=slurm/logs/jupyter_%j.out
#SBATCH --error=slurm/logs/jupyter_%j.err
#SBATCH --time=168:00:00
#SBATCH --partition=work
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# ==============================================================================
# Jupyter Server on GPU Node
# ==============================================================================
#
# This script starts a Jupyter Lab server on a GPU node with 2 GPUs
#
# Usage:
#   1. Submit this job: bash slurm/scripts/start_jupyter.sh <node_name>
#   2. Wait for the script to show the SSH tunnel command
#   3. Run the SSH tunnel command on your local machine
#   4. Open the Jupyter URL in your browser
#
# Note: Node is specified via --nodelist in the wrapper script
#
# ==============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Print server information
echo "=============================================="
echo "Jupyter Lab Server Starting"
echo "=============================================="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 2"
echo "Memory: 16GB"
echo "=============================================="
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate

# Check Python and Jupyter installation
echo "Python version: $(python --version 2>&1)"
echo ""

if ! command -v jupyter &> /dev/null; then
    echo "ERROR: Jupyter not found!"
    echo "Please install Jupyter: pip install jupyter jupyterlab"
    exit 1
fi

# Get hostname and port
HOSTNAME=$(hostname)
PORT=8888

# Find an available port
while nc -z localhost $PORT 2>/dev/null; do
    PORT=$((PORT + 1))
done

echo "Jupyter Lab will start on:"
echo "  Host: $HOSTNAME"
echo "  Port: $PORT"
echo ""

# Generate Jupyter config if it doesn't exist
JUPYTER_CONFIG_DIR="$HOME/.jupyter"
mkdir -p "$JUPYTER_CONFIG_DIR"

# Start Jupyter Lab
echo "Starting Jupyter Lab..."
echo ""
echo "=============================================="
echo "IMPORTANT: SSH Tunnel Command"
echo "=============================================="
echo ""
echo "Run this command on your LOCAL machine:"
echo ""
echo "  ssh -N -L localhost:$PORT:$HOSTNAME:$PORT $USER@login9"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:$PORT"
echo ""
echo "=============================================="
echo ""

# Start Jupyter with no browser and explicit IP
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=$PORT \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --notebook-dir="$HOME/work/final_project"

echo ""
echo "Jupyter Lab has stopped."
