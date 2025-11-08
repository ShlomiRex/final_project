#!/bin/bash

# ==============================================================================
# Helper script to connect to a running Jupyter Lab job
# ==============================================================================
# Usage: ./connect_jupyter.sh [JOB_ID]
# If JOB_ID is not provided, will show all running Jupyter jobs
# ==============================================================================

HPC_LOGIN_NODE="${HPC_LOGIN_NODE:-<hpc-login-node>}"  # Set this to your HPC login node

if [ -z "$1" ]; then
    echo "======================================================================"
    echo "Finding Running Jupyter Jobs"
    echo "======================================================================"
    echo ""
    
    # Find all running Jupyter jobs for this user
    JOBS=$(squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R" | grep jupyter)
    
    if [ -z "$JOBS" ]; then
        echo "No running Jupyter jobs found."
        echo ""
        echo "To start a new Jupyter session:"
        echo "    sbatch slurm/jupyter_notebook_interactive.sh"
        exit 0
    fi
    
    echo "Running Jupyter jobs:"
    echo ""
    squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D %R" | head -n 1
    echo "$JOBS"
    echo ""
    echo "To connect to a job, run:"
    echo "    $0 <JOB_ID>"
    echo ""
    exit 0
fi

JOB_ID=$1
OUTPUT_FILE="slurm/logs/jupyter_interactive_${JOB_ID}.out"

# Check if job is running
JOB_STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)

if [ -z "$JOB_STATUS" ]; then
    echo "ERROR: Job $JOB_ID not found or not running."
    echo ""
    echo "Check your running jobs with:"
    echo "    squeue -u $USER"
    exit 1
fi

if [ "$JOB_STATUS" != "RUNNING" ]; then
    echo "WARNING: Job $JOB_ID is in state: $JOB_STATUS (not RUNNING)"
    echo ""
fi

# Check if output file exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: Output file not found: $OUTPUT_FILE"
    echo ""
    echo "The job might still be starting. Wait a moment and try again."
    echo "You can check the job status with:"
    echo "    squeue -j $JOB_ID"
    exit 1
fi

echo "======================================================================"
echo "Jupyter Lab Connection Information"
echo "======================================================================"
echo ""
echo "Job ID: $JOB_ID"
echo "Status: $JOB_STATUS"
echo ""

# Extract node and port from output file
NODE=$(grep "Node:" "$OUTPUT_FILE" | awk '{print $2}')
PORT=$(grep "Using port:" "$OUTPUT_FILE" | awk '{print $3}')

# Try to get the actual port from jupyter server list if available
ACTUAL_PORT=$(grep "Currently running servers:" -A 1 "$OUTPUT_FILE" 2>/dev/null | grep "http://" | grep -oP ':\K[0-9]+/' | tr -d '/' | head -n 1)
if [ -n "$ACTUAL_PORT" ]; then
    PORT=$ACTUAL_PORT
fi

if [ -z "$NODE" ] || [ -z "$PORT" ]; then
    echo "ERROR: Could not extract connection information from $OUTPUT_FILE"
    echo ""
    echo "The server might still be starting. Output file contents:"
    echo "----------------------------------------------------------------------"
    cat "$OUTPUT_FILE"
    exit 1
fi

echo "Compute Node: $NODE"
echo "Port: $PORT"
echo ""
echo "======================================================================"
echo "CONNECTION COMMAND"
echo "======================================================================"
echo ""
echo "Run this command on your LOCAL machine:"
echo ""
echo "    ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@${HPC_LOGIN_NODE}"
echo ""
echo "Then open your browser and go to:"
echo ""
echo "    http://localhost:${PORT}/lab"
echo ""
echo "Note: Make sure to include '/lab' at the end!"
echo ""
echo "======================================================================"
echo ""
echo "To stop the Jupyter server when done:"
echo "    scancel $JOB_ID"
echo ""

# Check if there's a token in the output
echo "Checking for authentication details..."
echo ""

# Look for jupyter server list output
SERVER_LIST=$(grep -A 5 "Currently running servers:" "$OUTPUT_FILE" 2>/dev/null)
if [ -n "$SERVER_LIST" ]; then
    echo "Active Jupyter Servers:"
    echo "$SERVER_LIST"
    echo ""
fi

# Look for token in various formats
TOKEN=$(grep -oP "token=\K[a-f0-9]+" "$OUTPUT_FILE" 2>/dev/null | head -n 1)
if [ -z "$TOKEN" ]; then
    TOKEN=$(grep -oP "\?token=\K[a-f0-9]+" "$OUTPUT_FILE" 2>/dev/null | head -n 1)
fi

if [ -n "$TOKEN" ]; then
    echo "Jupyter Token Found: $TOKEN"
    echo ""
    echo "Full URL with token: http://localhost:${PORT}/lab?token=${TOKEN}"
    echo ""
elif grep -q "authentication is disabled" "$OUTPUT_FILE" 2>/dev/null; then
    echo "✓ Authentication is DISABLED - no token/password required"
    echo "  Just go to: http://localhost:${PORT}/lab"
    echo ""
else
    echo "No token found - check the output file for authentication details"
    echo ""
fi

echo "======================================================================"
echo ""
echo "Recent output from Jupyter server:"
echo "----------------------------------------------------------------------"
tail -n 20 "$OUTPUT_FILE"
echo "----------------------------------------------------------------------"
echo ""
echo "To view full output: cat $OUTPUT_FILE"
echo "To follow output: tail -f $OUTPUT_FILE"
