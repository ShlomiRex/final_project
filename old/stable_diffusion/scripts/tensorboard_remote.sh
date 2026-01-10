#!/bin/bash

# Script to start TensorBoard on a remote HPC node and provide SSH tunnel command
# Usage: ./tensorboard_remote.sh <node_name> [port]
# Example: ./tensorboard_remote.sh gpu8
# Example: ./tensorboard_remote.sh gpu8 6006

# Check if node name is provided
if [ -z "$1" ]; then
    echo "Error: Node name is required"
    echo "Usage: $0 <node_name> [port]"
    echo "Example: $0 gpu8"
    echo "Example: $0 gpu8 6006"
    exit 1
fi

NODE_NAME="$1"
PORT="${2:-6006}"  # Default port 6006 if not specified

echo "=========================================="
echo "Starting TensorBoard on ${NODE_NAME}"
echo "=========================================="
echo "Node: ${NODE_NAME}.hpc.pub.lan"
echo "Port: ${PORT}"
echo "=========================================="
echo ""

# SSH tunnel command for the user to run locally
TUNNEL_CMD="ssh -J doshlom4@hpc-proxy-server -N -L ${PORT}:${NODE_NAME}.hpc.pub.lan:${PORT} doshlom4@hpc-login8"

echo "📋 To access TensorBoard, run this command on your LOCAL machine:"
echo ""
echo "    ${TUNNEL_CMD}"
echo ""
echo "Then open in your browser: http://localhost:${PORT}"
echo ""
echo "=========================================="
echo "Starting TensorBoard server..."
echo "=========================================="
echo ""

# SSH into the node and run TensorBoard
ssh ${NODE_NAME}.hpc.pub.lan << EOF
    # Activate conda environment
    source /prefix/software/Anaconda3/2022.05/etc/profile.d/conda.sh
    conda activate /home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025
    
    # Navigate to project directory
    cd /home/doshlom4/work/final_project
    
    # Start TensorBoard
    echo "Starting TensorBoard on port ${PORT}..."
    tensorboard --logdir /home/doshlom4/work/final_project/notebooks/outputs/train1_flickr8k_text2img/tensorboard --port ${PORT} --bind_all
EOF

echo ""
echo "=========================================="
echo "TensorBoard stopped"
echo "=========================================="
