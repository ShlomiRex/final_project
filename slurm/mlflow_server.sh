#!/bin/bash
#SBATCH --job-name=mlflow_server
#SBATCH --output=slurm/logs/mlflow_server_%j.out
#SBATCH --error=slurm/logs/mlflow_server_%j.err
#SBATCH --time=168:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ==============================================================================
# MLflow Tracking Server for Slurm
# ==============================================================================
#
# This script starts an MLflow tracking server on a compute node.
# Training jobs should set MLFLOW_TRACKING_URI to point to this server.
#
# Usage:
#   1. Submit this job: sbatch slurm/mlflow_server.sh
#   2. Get the node name: squeue -u $USER | grep mlflow
#   3. Set in training scripts: export MLFLOW_TRACKING_URI=http://<node>:5000
#
# ==============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Print server information
echo "=============================================="
echo "MLflow Server Starting"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate latent-gpt 2>/dev/null || echo "Using system Python"

# Create MLflow directories
MLFLOW_DIR="${HOME}/work/final_project/mlruns"
mkdir -p "$MLFLOW_DIR"

# Get the hostname for clients to connect
HOSTNAME=$(hostname)
PORT=5000

echo ""
echo "=============================================="
echo "MLflow Server Configuration"
echo "=============================================="
echo "Tracking URI: http://${HOSTNAME}:${PORT}"
echo "Backend store: sqlite:///mlflow.db"
echo "Artifact store: ${MLFLOW_DIR}"
echo "=============================================="
echo ""
echo "To connect from training scripts, set:"
echo "  export MLFLOW_TRACKING_URI=http://${HOSTNAME}:${PORT}"
echo ""

# Start MLflow server
cd "${HOME}/work/final_project"

mlflow server \
    --backend-store-uri "sqlite:///mlflow.db" \
    --default-artifact-root "${MLFLOW_DIR}" \
    --host 0.0.0.0 \
    --port ${PORT}
