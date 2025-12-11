#!/bin/bash
#
# Script to start Jupyter Lab server on a specified GPU node
#
# Usage: bash start_jupyter.sh <node_name>
# Example: bash start_jupyter.sh gpu8
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SLURM_DIR/.." && pwd)"

# Check if node name is provided
if [ -z "$1" ]; then
    echo "ERROR: Node name required!"
    echo ""
    echo "Usage: bash $0 <node_name>"
    echo ""
    echo "Available GPU nodes:"
    sinfo -N -o "%.12N %.10P %.6t %.4c %.8m %.10G" | grep -E "NODELIST|gpu"
    echo ""
    echo "Example: bash $0 gpu8"
    exit 1
fi

NODE_NAME="$1"

echo "=========================================="
echo "Starting Jupyter Lab on $NODE_NAME"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$SLURM_DIR/logs"

# Make the job script executable
chmod +x "$SLURM_DIR/jobs/jupyter_server.sh"

# Submit the job with specified node
echo "Submitting Jupyter server job to $NODE_NAME..."
JOB_ID=$(sbatch --parsable --nodelist="$NODE_NAME" "$SLURM_DIR/jobs/jupyter_server.sh")

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job!"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"
echo ""

# Wait for job to start
echo "Waiting for job to start..."
while true; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
    
    if [ -z "$STATE" ]; then
        echo "ERROR: Job not found or failed to start!"
        echo "Check logs: $SLURM_DIR/logs/jupyter_${JOB_ID}.err"
        exit 1
    elif [ "$STATE" = "RUNNING" ]; then
        NODE=$(squeue -j "$JOB_ID" -h -o "%N")
        echo "Job is running on node: $NODE"
        break
    else
        echo "Job state: $STATE (waiting...)"
        sleep 2
    fi
done

echo ""
echo "=========================================="
echo "Jupyter Lab is starting..."
echo "=========================================="
echo ""
echo "Waiting for Jupyter to initialize (10 seconds)..."
sleep 10

# Display the output file to show connection instructions
OUTPUT_FILE="$SLURM_DIR/logs/jupyter_${JOB_ID}.out"

echo ""
echo "=========================================="
echo "Connection Instructions:"
echo "=========================================="
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    # Show the SSH tunnel command from the output
    cat "$OUTPUT_FILE" | grep -A 20 "SSH Tunnel Command"
    echo ""
else
    echo "Output file not yet available."
    echo "Check later: $OUTPUT_FILE"
fi

echo ""
echo "=========================================="
echo "Monitoring Jupyter Server"
echo "=========================================="
echo ""
echo "To view full output:"
echo "  tail -f $OUTPUT_FILE"
echo ""
echo "To stop the server:"
echo "  scancel $JOB_ID"
echo ""
echo "Job will run for 1 week (168 hours)"
echo "=========================================="
