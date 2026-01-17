#!/bin/bash
#SBATCH --job-name=jupyter_lab
#SBATCH --qos=gpu           # High priority QoS for GPU jobs
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --cpus-per-task=4   # Request 4 CPU cores
#SBATCH --mem=32G           # Request 32GB of memory
#SBATCH --time=168:00:00    # Max time: 1 week (168 hours)
#SBATCH --output=slurm/logs/jupyter_lab_%j.out
#SBATCH --error=slurm/logs/jupyter_lab_%j.err

# ==============================================================================
# Jupyter Lab Server for HPC Cluster
# ==============================================================================
# This script sets up a Jupyter Lab server on an HPC compute node with detailed
# logging including connection information, tokens, URLs, and system status.
#
# Usage:
#   sbatch slurm/jupyter_lab.sh
#
# To connect from your local machine, check the output file:
#   cat slurm/logs/jupyter_lab_<JOB_ID>.out
#
# Or use the helper script:
#   bash slurm/connect_jupyter.sh <JOB_ID>
# ==============================================================================

# Configuration
JUPYTER_PORT=${JUPYTER_PORT:-9998}  # Default port, will auto-increment if busy
ENVIRONMENT_TYPE="conda"  # Options: "virtualenv" or "conda"
VIRTUALENV_PATH="/home/doshlom4/torch114/bin/activate"
CONDA_ENV_NAME="shlomid_conda_12_11_2025"
NOTEBOOK_DIR="/home/doshlom4/work/final_project"
HPC_LOGIN_NODE="login8.openu.ac.il"  # Update with your actual login node

# ==============================================================================
# Print Header
# ==============================================================================

echo "======================================================================"
echo "             JUPYTER LAB SERVER - STARTING                            "
echo "======================================================================"
echo ""
echo "Job Information:"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Job Name:      $SLURM_JOB_NAME"
echo "  Node:          $SLURMD_NODENAME"
echo "  Start Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Working Dir:   $(pwd)"
echo ""
echo "Resource Allocation:"
echo "  CPUs:          $SLURM_CPUS_PER_TASK"
echo "  Memory:        $SLURM_MEM_PER_NODE MB"
echo "  Time Limit:    $SLURM_JOB_TIME_LIMIT"
if [ -n "$SLURM_JOB_GPUS" ]; then
echo "  GPUs:          $SLURM_JOB_GPUS"
fi
echo ""
echo "======================================================================"
echo ""

# ==============================================================================
# Environment Setup
# ==============================================================================

echo "Setting up Python environment..."
echo "----------------------------------------------------------------------"

# Activate the appropriate environment
if [ "$ENVIRONMENT_TYPE" = "virtualenv" ]; then
    echo "Environment Type: virtualenv"
    echo "Path: $VIRTUALENV_PATH"
    source "$VIRTUALENV_PATH"
elif [ "$ENVIRONMENT_TYPE" = "conda" ]; then
    echo "Environment Type: conda"
    echo "Conda Environment: $CONDA_ENV_NAME"
    # Source conda.sh to make conda command available
    source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV_NAME"
else
    echo "ERROR: Unknown ENVIRONMENT_TYPE: $ENVIRONMENT_TYPE"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate environment!"
    exit 1
fi

echo "✓ Environment activated successfully"
echo ""
echo "Python Configuration:"
echo "  Python Path:   $(which python)"
echo "  Python Version: $(python --version 2>&1)"
echo ""

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "WARNING: Jupyter not found! Installing..."
    pip install jupyterlab notebook ipykernel --quiet
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Jupyter!"
        exit 1
    fi
    echo "✓ Jupyter installed successfully"
else
    echo "✓ Jupyter found: $(jupyter --version 2>&1 | head -n 1)"
fi
echo ""

# ==============================================================================
# GPU Setup (if applicable)
# ==============================================================================

if [ -n "$SLURM_JOB_GPUS" ] || [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "======================================================================"
    echo "GPU Configuration"
    echo "======================================================================"
    
    # Load CUDA module if available
    if command -v module &> /dev/null; then
        echo "Loading CUDA module..."
        module load CUDA/11.8.0 2>/dev/null || module load CUDA/11.7.0 2>/dev/null || module load CUDA 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ CUDA module loaded"
        fi
    fi
    
    # Display GPU allocation
    [ -n "$SLURM_JOB_GPUS" ] && echo "  SLURM GPUs:           $SLURM_JOB_GPUS"
    [ -n "$SLURM_GPUS_ON_NODE" ] && echo "  GPUs on Node:         $SLURM_GPUS_ON_NODE"
    [ -n "$CUDA_VISIBLE_DEVICES" ] && echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo ""
    
    # Check PyTorch and CUDA
    echo "PyTorch & CUDA Status:"
    python -c "
import torch
print(f'  PyTorch Version:      {torch.__version__}')
print(f'  CUDA Available:       {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version:         {torch.version.cuda}')
    print(f'  Number of GPUs:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:                {torch.cuda.get_device_name(i)}')
else:
    print('  WARNING: CUDA not available to PyTorch!')
" 2>/dev/null || echo "  ERROR: PyTorch not available"
    
    echo ""
    
    # Run nvidia-smi if available
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status (nvidia-smi):"
        echo "----------------------------------------------------------------------"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>/dev/null || nvidia-smi 2>/dev/null | head -20
        echo "----------------------------------------------------------------------"
    fi
    
    echo ""
    echo "======================================================================"
    echo ""
else
    echo "NOTE: No GPU allocated for this job (CPU-only mode)"
    echo ""
fi

# ==============================================================================
# Find Available Port
# ==============================================================================

echo "Searching for available port..."

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
    
    echo "0"
    return 1
}

AVAILABLE_PORT=$(find_available_port $JUPYTER_PORT)

if [ "$AVAILABLE_PORT" = "0" ]; then
    echo "ERROR: Could not find available port!"
    exit 1
fi

echo "✓ Using port: $AVAILABLE_PORT"
echo ""

# ==============================================================================
# Connection Instructions
# ==============================================================================

echo "======================================================================"
echo "                    CONNECTION INFORMATION                            "
echo "======================================================================"
echo ""
echo "Your Jupyter Lab server is starting on:"
echo "  Node:          $SLURMD_NODENAME"
echo "  Port:          $AVAILABLE_PORT"
echo ""
echo "----------------------------------------------------------------------"
echo "TO CONNECT FROM YOUR LOCAL MACHINE:"
echo "----------------------------------------------------------------------"
echo ""
echo "1. Run this SSH tunnel command on your LOCAL machine:"
echo ""
echo "   ssh -N -L $AVAILABLE_PORT:$SLURMD_NODENAME:$AVAILABLE_PORT $USER@$HPC_LOGIN_NODE"
echo ""
echo "2. Then open your browser and navigate to:"
echo ""
echo "   http://localhost:$AVAILABLE_PORT/lab"
echo ""
echo "   NOTE: Make sure to include '/lab' at the end!"
echo ""
echo "----------------------------------------------------------------------"
echo "QUICK COMMANDS:"
echo "----------------------------------------------------------------------"
echo ""
echo "  View connection info:  bash slurm/connect_jupyter.sh $SLURM_JOB_ID"
echo "  Stop this server:      scancel $SLURM_JOB_ID"
echo "  Check job status:      squeue -j $SLURM_JOB_ID"
echo "  View this log:         cat slurm/logs/jupyter_lab_${SLURM_JOB_ID}.out"
echo "  Follow this log:       tail -f slurm/logs/jupyter_lab_${SLURM_JOB_ID}.out"
echo ""
echo "======================================================================"
echo ""

# ==============================================================================
# Start Jupyter Lab
# ==============================================================================

echo "Starting Jupyter Lab server..."
echo ""

# Create a temporary config to ensure we get the token in output
export JUPYTER_CONFIG_DIR="/tmp/jupyter_config_${SLURM_JOB_ID}"
mkdir -p "$JUPYTER_CONFIG_DIR"

# Start Jupyter Lab in the background so we can query it
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=$AVAILABLE_PORT \
    --notebook-dir="$NOTEBOOK_DIR" \
    --ServerApp.allow_root=True \
    --ServerApp.token="" \
    --ServerApp.password="" \
    --ServerApp.open_browser=False &

JUPYTER_PID=$!

echo "✓ Jupyter Lab started (PID: $JUPYTER_PID)"
echo ""
echo "Waiting for server to initialize..."

# Wait for Jupyter to start (max 30 seconds)
for i in {1..30}; do
    sleep 1
    if jupyter server list 2>/dev/null | grep -q "$AVAILABLE_PORT"; then
        echo "✓ Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "WARNING: Server may not have started properly"
    fi
done

echo ""

# ==============================================================================
# Display Active Servers and Connection Details
# ==============================================================================

echo "======================================================================"
echo "                    ACTIVE JUPYTER SERVERS                            "
echo "======================================================================"
echo ""

# List all running servers
jupyter server list 2>&1

echo ""
echo "======================================================================"
echo ""

# Extract connection details
SERVER_INFO=$(jupyter server list 2>&1)
SERVER_URL=$(echo "$SERVER_INFO" | grep -oP 'http://[^:]+:\K[0-9]+' | head -n 1)
TOKEN=$(echo "$SERVER_INFO" | grep -oP '\?token=\K[a-z0-9]+' | head -n 1)

echo "Server Status:"
if [ -n "$SERVER_URL" ]; then
    echo "  ✓ Server is running on port: $SERVER_URL"
else
    echo "  Port: $AVAILABLE_PORT"
fi

if [ -n "$TOKEN" ]; then
    echo "  Token: $TOKEN"
    echo ""
    echo "  Full URL with token:"
    echo "    http://localhost:$AVAILABLE_PORT/lab?token=$TOKEN"
else
    echo "  Authentication: DISABLED (no token/password required)"
    echo ""
    echo "  Direct URL:"
    echo "    http://localhost:$AVAILABLE_PORT/lab"
fi

echo ""
echo "======================================================================"
echo ""
echo "Server is running! Press Ctrl+C to stop the server."
echo ""
echo "======================================================================"
echo ""

# Save connection info to a separate file for easy access
INFO_FILE="slurm/logs/jupyter_connection_${SLURM_JOB_ID}.txt"
cat > "$INFO_FILE" <<EOF
Jupyter Lab Connection Information
===================================

Job ID:       $SLURM_JOB_ID
Node:         $SLURMD_NODENAME
Port:         $AVAILABLE_PORT
Started:      $(date '+%Y-%m-%d %H:%M:%S')

SSH Tunnel Command (run on your local machine):
-------------------------------------------------
ssh -N -L $AVAILABLE_PORT:$SLURMD_NODENAME:$AVAILABLE_PORT $USER@$HPC_LOGIN_NODE

Browser URL:
-------------------------------------------------
http://localhost:$AVAILABLE_PORT/lab

Stop Command:
-------------------------------------------------
scancel $SLURM_JOB_ID
EOF

echo "Connection info saved to: $INFO_FILE"
echo ""

# Keep the script running and wait for Jupyter process
wait $JUPYTER_PID

# ==============================================================================
# Cleanup
# ==============================================================================

echo ""
echo "======================================================================"
echo "Jupyter Lab server stopped"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# Cleanup temp config
rm -rf "$JUPYTER_CONFIG_DIR" 2>/dev/null

exit 0
