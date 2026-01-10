#!/bin/bash
# ============================================================================
# Monitor Multi-GPU Training Job
# ============================================================================
# Usage: bash slurm/monitor_training.sh [job_id]
#
# If no job_id provided, monitors the most recent train1_flickr8 job
# ============================================================================

# Get job ID from argument or find most recent
if [ -n "$1" ]; then
    JOB_ID=$1
else
    # Find most recent train1_flickr8 job
    JOB_ID=$(squeue -u $USER -o "%.18i %.9P %.30j" | grep train1_flickr8 | head -1 | awk '{print $1}')
    
    if [ -z "$JOB_ID" ]; then
        echo "❌ No active train1_flickr8 job found"
        echo ""
        echo "Active jobs:"
        squeue -u $USER
        exit 1
    fi
fi

echo "=========================================="
echo "📊 Monitoring Training Job: $JOB_ID"
echo "=========================================="
echo ""

# Check if job is running
JOB_STATUS=$(squeue -j $JOB_ID -o "%.8T" 2>/dev/null | tail -1)

if [ -z "$JOB_STATUS" ]; then
    echo "⚠️  Job $JOB_ID not found in queue (may have completed or failed)"
    echo ""
    echo "Checking logs..."
    echo ""
fi

# Find log files
OUT_LOG=$(ls -t slurm/logs/train1_flickr8*_${JOB_ID}.out 2>/dev/null | head -1)
ERR_LOG=$(ls -t slurm/logs/train1_flickr8*_${JOB_ID}.err 2>/dev/null | head -1)
TRAIN_LOG=$(ls -t slurm/logs/train1_flickr8_training_${JOB_ID}.log 2>/dev/null | head -1)

echo "📁 Log Files:"
echo "   Standard output: ${OUT_LOG:-NOT FOUND}"
echo "   Standard error:  ${ERR_LOG:-NOT FOUND}"
echo "   Training log:    ${TRAIN_LOG:-NOT FOUND}"
echo ""

# Show job info
echo "=========================================="
echo "🔍 Job Information"
echo "=========================================="
squeue -j $JOB_ID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Job not in queue"
echo ""

# Show recent output
if [ -f "$OUT_LOG" ]; then
    echo "=========================================="
    echo "📝 Recent Output (last 30 lines)"
    echo "=========================================="
    tail -30 "$OUT_LOG"
    echo ""
fi

# Show errors if any
if [ -f "$ERR_LOG" ] && [ -s "$ERR_LOG" ]; then
    echo "=========================================="
    echo "⚠️  Errors Detected"
    echo "=========================================="
    tail -20 "$ERR_LOG"
    echo ""
fi

# Show training progress if available
if [ -f "$TRAIN_LOG" ]; then
    echo "=========================================="
    echo "🎯 Training Progress (last 20 lines)"
    echo "=========================================="
    tail -20 "$TRAIN_LOG"
    echo ""
    
    # Try to extract epoch/loss info
    echo "📊 Latest Metrics:"
    grep -E "Epoch|Loss|Step" "$TRAIN_LOG" | tail -5 || echo "No metrics found yet"
    echo ""
fi

# Show GPU usage on compute node (if job is running)
if [ "$JOB_STATUS" = "RUNNING" ]; then
    NODE=$(squeue -j $JOB_ID -o "%N" | tail -1)
    echo "=========================================="
    echo "🎮 GPU Usage on $NODE"
    echo "=========================================="
    ssh $NODE "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv" 2>/dev/null || echo "Could not connect to compute node"
    echo ""
fi

echo "=========================================="
echo "💡 Monitoring Commands"
echo "=========================================="
echo "Watch live output:     tail -f $OUT_LOG"
echo "Watch training log:    tail -f $TRAIN_LOG"
echo "Watch GPU usage:       ssh $NODE 'watch nvidia-smi'"
echo "Cancel job:            scancel $JOB_ID"
echo ""
