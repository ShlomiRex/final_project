#!/bin/bash
#SBATCH --job-name=jupyter_interactive_nogpu
#SBATCH --qos=normal        # Normal priority QoS for CPU-only jobs
#SBATCH --cpus-per-task=4   # Request 4 CPU cores (reduced from 8)
#SBATCH --mem=16G           # Request 16GB of memory (reduced from 32G)
#SBATCH --time=24:00:00     # Max time: 24 hours (reduced from 48)
#SBATCH --output=slurm/logs/jupyter_interactive_%j.out
#SBATCH --error=slurm/logs/jupyter_interactive_%j.err

# ==============================================================================
# Interactive Jupyter Notebook on HPC Cluster (NO GPU - for quick access)
# ==============================================================================
# This script sets up a Jupyter Lab server on a compute node WITHOUT GPU.
# Use this when you need quick access for development/testing.
# For GPU-intensive work, use jupyter_notebook_interactive.sh instead.
# ==============================================================================

# Configuration - Edit these as needed
# JUPYTER_PORT can be passed as environment variable from sbatch --export
JUPYTER_PORT=${JUPYTER_PORT:-9998}  # Default port if not specified, will auto-increment if busy
ENVIRONMENT_TYPE="conda"  # Options: "virtualenv" or "conda"
VIRTUALENV_PATH="/home/doshlom4/work/conda/envs/torch114"  # If using virtualenv
CONDA_ENV_NAME="torch114"  # torch114 has Python and Jupyter; mycondaenv is broken

# ==============================================================================
# Environment Setup
# ==============================================================================

echo "======================================================================"
echo "Starting Jupyter Lab Interactive Session (NO GPU)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Activate the appropriate environment
if [ "$ENVIRONMENT_TYPE" = "virtualenv" ]; then
    echo "Activating virtualenv: $VIRTUALENV_PATH"
    source "$VIRTUALENV_PATH/bin/activate"
elif [ "$ENVIRONMENT_TYPE" = "conda" ]; then
    echo "Activating conda environment: $CONDA_ENV_NAME"
    # Source conda.sh to make conda command available
    source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
else
    echo "ERROR: Unknown ENVIRONMENT_TYPE: $ENVIRONMENT_TYPE"
    exit 1
fi

echo ""
echo "Environment activated successfully!"
echo "----------------------------------------------------------------------"

# Display Python information
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check if Jupyter is installed, if not install it
if ! command -v jupyter &> /dev/null; then
    echo "WARNING: Jupyter is not installed in this environment!"
    echo "Installing Jupyter Lab and Notebook..."
    echo ""
    pip install jupyterlab notebook ipykernel --quiet
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Jupyter!"
        echo "Please install it manually with: pip install jupyterlab notebook"
        exit 1
    fi
    echo "Jupyter installed successfully!"
    echo ""
fi

echo "Jupyter version: $(jupyter --version 2>&1 | head -n 1)"
echo ""

# Check and install project dependencies if needed
echo "Checking project dependencies..."
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing project dependencies from requirements or common packages..."
    pip install numpy torch torchvision matplotlib transformers diffusers accelerate datasets --quiet
    echo "Dependencies installed!"
fi
echo ""

# Check PyTorch (CPU mode)
echo "Checking PyTorch availability (CPU mode - no GPU requested):"
python -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'NOTE: GPU not requested for this job - using CPU mode')
except ImportError:
    print('PyTorch is not installed')
" 2>/dev/null || echo "PyTorch not available"

echo ""
echo "======================================================================"
echo ""

# ==============================================================================
# Find an available port
# ==============================================================================

find_available_port() {
    local port=$1
    local max_attempts=20
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ! netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo $port
            return 0
        fi
        port=$((port + 1))
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Could not find available port after $max_attempts attempts"
    exit 1
}

AVAILABLE_PORT=$(find_available_port $JUPYTER_PORT)
echo "Using port: $AVAILABLE_PORT"
echo ""

# ==============================================================================
# Display Connection Instructions
# ==============================================================================

echo "======================================================================"
echo "CONNECTION INSTRUCTIONS"
echo "======================================================================"
echo ""
echo "Your Jupyter Lab server is starting on compute node: $SLURMD_NODENAME"
echo ""
echo "To connect from your LOCAL machine, run this command:"
echo ""
echo "    ssh -N -L $AVAILABLE_PORT:$SLURMD_NODENAME:$AVAILABLE_PORT $USER@<hpc-login-node>"
echo ""
echo "Replace <hpc-login-node> with your HPC's login node address."
echo ""
echo "Then open your browser and go to:"
echo "    http://localhost:$AVAILABLE_PORT"
echo ""
echo "The Jupyter token will be displayed below once the server starts."
echo ""
echo "To stop this job when you're done:"
echo "    scancel $SLURM_JOB_ID"
echo ""
echo "======================================================================"
echo ""
echo "Starting Jupyter Lab server..."
echo ""

# ==============================================================================
# Start Jupyter Lab
# ==============================================================================

# Start Jupyter Lab with:
# - No browser (we'll connect via SSH tunnel)
# - Bind to all interfaces
# - Use the available port
# - Allow root (some HPC systems need this)
# - Set notebook directory to project root
NOTEBOOK_DIR="/home/doshlom4/work/final_project"

# Start Jupyter in the background so we can list servers
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=$AVAILABLE_PORT \
    --notebook-dir="$NOTEBOOK_DIR" \
    --ServerApp.allow_root=True \
    --ServerApp.token="" \
    --ServerApp.password="" &

# Wait for Jupyter to start
sleep 5

echo ""
echo "======================================================================"
echo "ACTIVE JUPYTER SERVERS"
echo "======================================================================"
echo ""
jupyter server list
echo ""

# Extract the actual port being used from jupyter server list
ACTUAL_PORT=$(jupyter server list 2>/dev/null | grep -oP 'http://[^:]+:\K[0-9]+' | head -n 1)
if [ -n "$ACTUAL_PORT" ] && [ "$ACTUAL_PORT" != "$AVAILABLE_PORT" ]; then
    echo "NOTE: Port changed from $AVAILABLE_PORT to $ACTUAL_PORT"
    echo "      Use port $ACTUAL_PORT in your SSH tunnel!"
    echo ""
fi

echo "======================================================================"
echo ""
echo "If you need a token/password, it will be shown above."
echo "If authentication is disabled, you can access directly without credentials."
echo ""
echo "======================================================================"
echo ""

# Bring Jupyter back to foreground
wait

echo ""
echo "======================================================================"
echo "Jupyter Lab server stopped"
echo "End time: $(date)"
echo "======================================================================"
