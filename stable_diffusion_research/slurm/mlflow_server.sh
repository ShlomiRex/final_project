#!/bin/bash
#SBATCH --job-name=mlflow-server
#SBATCH --partition=work
#SBATCH --nodelist=gpu8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/mlflow_server_%j.out
#SBATCH --error=slurm/logs/mlflow_server_%j.err

# ============================================================================
# MLflow Tracking Server
# ============================================================================
#
# Usage:
#   sbatch slurm/mlflow_server.sh
#
# After job starts, get the node name:
#   squeue -u $USER | grep mlflow
#
# Then set environment variable in training jobs:
#   export MLFLOW_TRACKING_URI=http://<node>:5000
#
# Access from local machine via SSH tunnel:
#   ssh -N -L localhost:5000:<node>.hpc.pub.lan:5000 doshlom4@login9
#   Then open: http://localhost:5000
#
# ============================================================================

set -e

echo "=============================================="
echo "MLflow Server Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Load environment
source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate

# Change to project directory
cd /home/doshlom4/work/final_project/stable_diffusion_research

mkdir -p slurm/logs
mkdir -p mlruns

# Get hostname
HOSTNAME=$(hostname)
PORT=5000

echo ""
echo "=============================================="
echo "MLflow server running at: http://${HOSTNAME}:${PORT}"
echo ""
echo "To use in training:"
echo "  export MLFLOW_TRACKING_URI=http://${HOSTNAME}:${PORT}"
echo ""
echo "To access from local machine:"
echo "  ssh -N -L localhost:${PORT}:${HOSTNAME}.hpc.pub.lan:${PORT} doshlom4@login9"
echo "  Then open: http://localhost:${PORT}"
echo "=============================================="
echo ""

# Start MLflow server
mlflow server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlruns/artifacts
