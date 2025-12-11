#!/bin/bash
#
# Script to submit CUDA detection job and wait for results
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SLURM_DIR/.." && pwd)"

echo "=========================================="
echo "CUDA Detection on Compute Node"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$SLURM_DIR/logs"

# Make the job script executable
chmod +x "$SLURM_DIR/jobs/detect_cuda.sh"

# Submit the job
echo "Submitting CUDA detection job..."
JOB_ID=$(sbatch --parsable "$SLURM_DIR/jobs/detect_cuda.sh")

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
        # Job finished or not found
        break
    elif [ "$STATE" = "RUNNING" ]; then
        echo "Job is running on node: $(squeue -j "$JOB_ID" -h -o "%N")"
        break
    else
        echo "Job state: $STATE"
        sleep 2
    fi
done

# Wait for job to complete
echo "Waiting for job to complete..."
while true; do
    STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)
    
    if [ -z "$STATE" ]; then
        # Job finished
        echo "Job completed!"
        break
    fi
    
    sleep 2
done

echo ""
echo "=========================================="
echo "Job Output:"
echo "=========================================="
echo ""

# Display the output
OUTPUT_FILE="$SLURM_DIR/logs/detect_cuda_${JOB_ID}.out"
ERROR_FILE="$SLURM_DIR/logs/detect_cuda_${JOB_ID}.err"

if [ -f "$OUTPUT_FILE" ]; then
    cat "$OUTPUT_FILE"
else
    echo "Output file not found: $OUTPUT_FILE"
fi

echo ""

if [ -f "$ERROR_FILE" ] && [ -s "$ERROR_FILE" ]; then
    echo "=========================================="
    echo "Job Errors:"
    echo "=========================================="
    cat "$ERROR_FILE"
fi

echo ""
echo "=========================================="
echo "Log files:"
echo "  Output: $OUTPUT_FILE"
echo "  Errors: $ERROR_FILE"
echo "=========================================="
