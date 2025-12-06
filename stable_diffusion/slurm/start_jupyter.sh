#!/bin/bash

# ==============================================================================
# Jupyter Lab Launcher for HPC
# ==============================================================================
# This script runs on the HPC login node and:
# 1. Finds an available port on the compute node
# 2. Submits the Jupyter job with that port
# 3. Waits for the job to start
# 4. Prints the SSH tunnel command for your local Windows machine
#
# Usage:
#   bash slurm/start_jupyter.sh                    # Request GPU (any available)
#   bash slurm/start_jupyter.sh nogpu              # No GPU (faster, for development)
#   bash slurm/start_jupyter.sh gpu8               # Request specific GPU node
#   bash slurm/start_jupyter.sh cn43               # Request specific CPU node
# ==============================================================================

set -e

# Configuration
HPC_USER="${USER}"
PROXY_SERVER="sheshet.cslab.openu.ac.il"
LOGIN_NODE="login8.openu.ac.il"
MIN_PORT=9000
MAX_PORT=9999

# Parse arguments
USE_GPU=true
SCRIPT_NAME="jupyter_notebook_interactive.sh"
SPECIFIC_NODE=""
NODELIST_CONSTRAINT=""

# Check first argument
if [ "$1" = "nogpu" ] || [ "$1" = "no-gpu" ] || [ "$1" = "--nogpu" ]; then
    USE_GPU=false
    SCRIPT_NAME="jupyter_notebook_interactive_nogpu.sh"
    echo "======================================================================"
    echo "Starting Jupyter Lab (NO GPU - for quick access)"
    echo "======================================================================"
elif [[ "$1" =~ ^(gpu[1-8]|cn[0-9]+)$ ]]; then
    # User specified a specific node
    SPECIFIC_NODE="$1"
    NODELIST_CONSTRAINT="--nodelist=$SPECIFIC_NODE"
    
    # Check if it's a GPU node
    if [[ "$SPECIFIC_NODE" =~ ^gpu ]]; then
        USE_GPU=true
        SCRIPT_NAME="jupyter_notebook_interactive.sh"
        echo "======================================================================"
        echo "Starting Jupyter Lab on SPECIFIC GPU NODE: $SPECIFIC_NODE"
        echo "======================================================================"
    else
        USE_GPU=false
        SCRIPT_NAME="jupyter_notebook_interactive_nogpu.sh"
        echo "======================================================================"
        echo "Starting Jupyter Lab on SPECIFIC CPU NODE: $SPECIFIC_NODE"
        echo "======================================================================"
    fi
elif [ -n "$1" ]; then
    echo "ERROR: Invalid argument '$1'"
    echo ""
    echo "Usage:"
    echo "  bash slurm/start_jupyter.sh              # Any available GPU"
    echo "  bash slurm/start_jupyter.sh nogpu        # Any available CPU node"
    echo "  bash slurm/start_jupyter.sh gpu8         # Specific GPU node (gpu1-8)"
    echo "  bash slurm/start_jupyter.sh cn43         # Specific CPU node (cn01-44)"
    echo ""
    echo "Available GPU nodes: gpu1, gpu2, gpu3, gpu4, gpu6, gpu7, gpu8"
    echo "Available CPU nodes: cn01-cn44"
    echo ""
    echo "To check availability, run:"
    echo "  bash slurm/check_gpu_availability.sh"
    exit 1
else
    echo "======================================================================"
    echo "Starting Jupyter Lab (WITH GPU)"
    echo "======================================================================"
    echo ""
    echo "Note: GPU jobs may take longer to start. For quick access, use:"
    echo "  bash slurm/start_jupyter.sh nogpu"
    echo ""
    echo "To request a specific GPU node, use:"
    echo "  bash slurm/start_jupyter.sh gpu8"
fi
echo ""

# ==============================================================================
# Functions
# ==============================================================================

# Generate a random port number
generate_random_port() {
    echo $((MIN_PORT + RANDOM % (MAX_PORT - MIN_PORT + 1)))
}

# Check if a port is available (not in use by any Jupyter server)
check_port_available() {
    local port=$1
    # Check if any running jupyter job is using this port
    if squeue -u $USER -o "%.18i %.100k" | grep -q "port=$port"; then
        return 1  # Port is in use
    fi
    return 0  # Port is available
}

# Find an available port
find_available_port() {
    local port
    local max_attempts=20
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        port=$(generate_random_port)
        if check_port_available $port; then
            echo $port
            return 0
        fi
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Could not find available port after $max_attempts attempts" >&2
    exit 1
}

# ==============================================================================
# Main Script
# ==============================================================================

echo "======================================================================"
echo "Starting Jupyter Lab on HPC Cluster"
echo "======================================================================"
echo ""

# Show cluster resource availability
echo "Checking cluster resource availability..."
echo "----------------------------------------------------------------------"
sinfo -o "%20P %5a %.10l %16F %N" | head -10
echo "----------------------------------------------------------------------"
echo "Legend: NODES(A/I/O/T) = Allocated/Idle/Offline/Total"
echo ""

# Find available port
echo "Finding available port..."
JUPYTER_PORT=$(find_available_port)
echo "Selected port: $JUPYTER_PORT"
echo ""

# Submit the Jupyter job with the selected port
echo "Submitting Jupyter job..."
if [ -n "$SPECIFIC_NODE" ]; then
    echo "Requesting specific node: $SPECIFIC_NODE"
    JOB_ID=$(sbatch $NODELIST_CONSTRAINT --export=JUPYTER_PORT=$JUPYTER_PORT slurm/$SCRIPT_NAME | awk '{print $4}')
else
    JOB_ID=$(sbatch --export=JUPYTER_PORT=$JUPYTER_PORT slurm/$SCRIPT_NAME | awk '{print $4}')
fi

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to submit job"
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"
echo ""

# Get job details
echo "Job details:"
echo "----------------------------------------------------------------------"
scontrol show job $JOB_ID | grep -E "(JobState|Reason|StartTime|Partition|TRES|Priority)" | sed 's/^/  /'
echo "----------------------------------------------------------------------"
echo ""

# Wait for job to start
echo "Waiting for job to start..."
MAX_WAIT=60  # Maximum wait time in seconds
ELAPSED=0
SLEEP_INTERVAL=2
LAST_REASON=""

while [ $ELAPSED -lt $MAX_WAIT ]; do
    JOB_STATE=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null || echo "NOTFOUND")
    JOB_REASON=$(squeue -j $JOB_ID -h -o "%r" 2>/dev/null || echo "")
    
    if [ "$JOB_STATE" = "RUNNING" ]; then
        echo ""
        echo "✓ Job is now running!"
        break
    elif [ "$JOB_STATE" = "NOTFOUND" ] || [ "$JOB_STATE" = "COMPLETED" ] || [ "$JOB_STATE" = "FAILED" ]; then
        echo ""
        echo "ERROR: Job failed to start or completed immediately"
        echo "Check logs: slurm/logs/jupyter_interactive_${JOB_ID}.{out,err}"
        exit 1
    elif [ "$JOB_STATE" = "PENDING" ]; then
        # Show reason if it changed
        if [ "$JOB_REASON" != "$LAST_REASON" ] && [ -n "$JOB_REASON" ]; then
            echo ""
            case "$JOB_REASON" in
                "Resources")
                    echo "⏳ Waiting for resources to become available..."
                    ;;
                "Priority")
                    echo "⏳ Waiting in queue (higher priority jobs ahead)..."
                    ;;
                "Dependency")
                    echo "⏳ Waiting for dependent job to complete..."
                    ;;
                "QOSGrpCpuLimit"|"PartitionCpuLimit"|"AssociationCpuLimit")
                    echo "⏳ Waiting for CPUs to become available..."
                    ;;
                "QOSGrpNodeLimit"|"PartitionNodeLimit"|"AssociationNodeLimit")
                    echo "⏳ Waiting for nodes to become available..."
                    ;;
                *)
                    echo "⏳ Waiting (Reason: $JOB_REASON)..."
                    ;;
            esac
            LAST_REASON="$JOB_REASON"
        fi
    fi
    
    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    echo -n "."
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "⚠️  Job hasn't started yet after ${MAX_WAIT}s"
    echo ""
    
    # Get detailed reason
    JOB_REASON=$(squeue -j $JOB_ID -h -o "%r" 2>/dev/null || echo "Unknown")
    
    echo "Current job status:"
    echo "----------------------------------------------------------------------"
    scontrol show job $JOB_ID | grep -E "(JobState|Reason|StartTime|RunTime)" | sed 's/^/  /'
    echo "----------------------------------------------------------------------"
    echo ""
    
    # Provide helpful explanation based on reason
    case "$JOB_REASON" in
        "Resources")
            echo "💡 Job is waiting for resources (CPUs, GPUs, or memory)."
            echo "   This usually means the cluster is busy. Your job will start when resources free up."
            echo ""
            if [ "$USE_GPU" = true ]; then
                echo "   💡 TIP: For faster access, try the no-GPU version:"
                echo "      bash slurm/start_jupyter.sh nogpu"
                echo ""
            fi
            ;;
        "Priority")
            echo "💡 Job is waiting in queue. Higher priority jobs will run first."
            ;;
        "QOSGrp"*|"Partition"*|"Association"*)
            echo "💡 Job is waiting due to limits on available resources."
            ;;
    esac
    
    echo "Cluster queue status:"
    echo "----------------------------------------------------------------------"
    squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.10l %.6D %R" | head -10
    echo "----------------------------------------------------------------------"
    echo ""
    echo "To monitor: squeue -j $JOB_ID"
    echo "To cancel:  scancel $JOB_ID"
fi

echo ""
echo ""

# Get compute node information
sleep 3  # Give it a moment to write output
COMPUTE_NODE=$(squeue -j $JOB_ID -h -o "%N" 2>/dev/null)

if [ -z "$COMPUTE_NODE" ] || [ "$COMPUTE_NODE" = "" ]; then
    echo "WARNING: Job is still pending, compute node not yet assigned"
    echo "The job will start when resources become available."
    echo ""
    COMPUTE_NODE="<COMPUTE_NODE>"  # Placeholder
    NODE_PENDING=true
else
    NODE_PENDING=false
    # Add domain if not present
    if [[ ! "$COMPUTE_NODE" =~ \. ]]; then
        COMPUTE_NODE="${COMPUTE_NODE}.hpc.pub.lan"
    fi
fi

echo "======================================================================"
echo "JUPYTER LAB IS STARTING"
echo "======================================================================"
echo ""
echo "Job ID:       $JOB_ID"
echo "Compute Node: $COMPUTE_NODE"
echo "Jupyter Port: $JUPYTER_PORT"
echo ""
echo "======================================================================"
echo "STEP 1: Wait for Jupyter to fully start (~30 seconds)"
echo "======================================================================"
echo ""
echo "Monitor the startup with:"
echo "  tail -f slurm/logs/jupyter_interactive_${JOB_ID}.out"
echo ""
echo "Or check connection details with:"
echo "  bash slurm/connect_jupyter.sh $JOB_ID"
echo ""
echo "======================================================================"
echo "STEP 2: Run this command on your LOCAL MACHINE (Windows/Mac/Linux):"
echo "======================================================================"
echo ""

# Suggest a local port (same as Jupyter port for simplicity)
LOCAL_PORT=$JUPYTER_PORT

if [ "$NODE_PENDING" = true ]; then
    echo "⏳ Job is PENDING - waiting for cluster resources..."
    echo ""
    echo "The job hasn't been assigned to a compute node yet."
    echo "Wait for the job to start, then get the connection command with:"
    echo ""
    echo "  bash slurm/connect_jupyter.sh $JOB_ID"
    echo ""
    echo "Or monitor the job queue:"
    echo "  watch -n 2 'squeue -j $JOB_ID'"
    echo ""
else
    echo "ssh -J ${HPC_USER}@${PROXY_SERVER} -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:${JUPYTER_PORT} ${HPC_USER}@${LOGIN_NODE}"
    echo ""
fi
echo "======================================================================"
echo "STEP 3: Open your browser"
echo "======================================================================"
echo ""
echo "Go to: http://localhost:${LOCAL_PORT}/lab"
echo ""
echo "Note: Make sure to include '/lab' at the end of the URL!"
echo ""
echo "======================================================================"
echo "When finished, stop the job:"
echo "======================================================================"
echo ""
echo "  scancel $JOB_ID"
echo ""
echo "======================================================================"
echo ""
echo "Waiting for Jupyter server to start (checking logs in 15 seconds)..."
sleep 15

# Check if Jupyter has started by looking at the output
if [ -f "slurm/logs/jupyter_interactive_${JOB_ID}.out" ]; then
    echo ""
    echo "Latest output from Jupyter server:"
    echo "----------------------------------------------------------------------"
    tail -20 "slurm/logs/jupyter_interactive_${JOB_ID}.out"
    echo "----------------------------------------------------------------------"
    echo ""
    
    # Try to extract actual port if different
    ACTUAL_PORT=$(grep -oP "http://[^:]+:\K[0-9]+/" "slurm/logs/jupyter_interactive_${JOB_ID}.out" 2>/dev/null | tr -d '/' | head -n 1)
    if [ -n "$ACTUAL_PORT" ] && [ "$ACTUAL_PORT" != "$JUPYTER_PORT" ]; then
        echo "⚠️  NOTE: Jupyter is using port $ACTUAL_PORT (not $JUPYTER_PORT)"
        echo ""
        if [ "$NODE_PENDING" = false ]; then
            echo "Updated SSH tunnel command:"
            echo "  ssh -J ${HPC_USER}@${PROXY_SERVER} -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:${ACTUAL_PORT} ${HPC_USER}@${LOGIN_NODE}"
        fi
        echo ""
    fi
else
    echo ""
    echo "ℹ️  Log files not created yet."
    echo ""
    if [ "$NODE_PENDING" = true ]; then
        # Get estimated start time and reason
        echo "   Job Status Details:"
        echo "   ----------------------------------------------------------------------"
        scontrol show job $JOB_ID | grep -E "(JobState|Reason|StartTime|Priority)" | sed 's/^/     /'
        echo "   ----------------------------------------------------------------------"
        START_TIME=$(scontrol show job $JOB_ID | grep -oP 'StartTime=\K[^ ]+' 2>/dev/null)
        REASON=$(scontrol show job $JOB_ID | grep -oP 'Reason=\K[^ ]+' 2>/dev/null)
        
        if [ -n "$START_TIME" ]; then
            echo "   Estimated start time: $START_TIME"
        fi
        if [ -n "$REASON" ]; then
            echo "   Pending reason: $REASON"
        fi
        echo ""
        echo "   Job is waiting for available cluster resources."
        echo ""
        echo "   To check job status:"
        echo "     squeue -j $JOB_ID"
        echo ""
        echo "   To see cluster availability:"
        echo "     sinfo -o '%20P %5a %.10l %16F %N'"
        echo ""
        echo "   To cancel if needed:"
        echo "     scancel $JOB_ID"
        echo ""
        echo "   Once running, get the connection command with:"
        echo "     bash slurm/connect_jupyter.sh $JOB_ID"
    else
        echo "   The job just started. Logs will appear shortly."
        echo "   Check again with: bash slurm/connect_jupyter.sh $JOB_ID"
    fi
    echo ""
fi

echo "✅ Setup complete! Follow the steps above to connect."
echo ""
