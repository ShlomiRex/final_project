#!/bin/bash

# ==============================================================================
# GPU Availability Checker
# ==============================================================================
# This script checks all GPU nodes and shows which ones have available GPUs
# ==============================================================================

echo "======================================================================"
echo "GPU Node Availability Report"
echo "======================================================================"
echo ""
echo "Checking GPU nodes: gpu1, gpu2, gpu3, gpu4, gpu6, gpu7, gpu8"
echo ""

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Arrays to store results
declare -a AVAILABLE_NODES
declare -a BUSY_NODES
declare -a DOWN_NODES

echo "======================================================================"
echo "Individual Node Details"
echo "======================================================================"
echo ""

for node in gpu1 gpu2 gpu3 gpu4 gpu6 gpu7 gpu8; do
    echo "--- $node ---"
    
    # Get node information
    NODE_INFO=$(scontrol show node $node 2>/dev/null)
    
    if [ -z "$NODE_INFO" ]; then
        echo "  ERROR: Node not found"
        echo ""
        continue
    fi
    
    # Extract key information
    STATE=$(echo "$NODE_INFO" | grep -oP 'State=\K[^ ]+' | head -n 1)
    CPU_ALLOC=$(echo "$NODE_INFO" | grep -oP 'CPUAlloc=\K[0-9]+')
    CPU_TOT=$(echo "$NODE_INFO" | grep -oP 'CPUTot=\K[0-9]+')
    MEM_ALLOC=$(echo "$NODE_INFO" | grep -oP 'AllocMem=\K[0-9]+')
    REAL_MEM=$(echo "$NODE_INFO" | grep -oP 'RealMemory=\K[0-9]+')
    GRES=$(echo "$NODE_INFO" | grep -oP 'Gres=\K[^ ]+')
    GRES_USED=$(echo "$NODE_INFO" | grep -oP 'GresUsed=\K[^ ]+')
    
    # Parse GPU information
    if [[ $GRES =~ gpu:([^:]+):([0-9]+) ]]; then
        GPU_TYPE="${BASH_REMATCH[1]}"
        GPU_TOTAL="${BASH_REMATCH[2]}"
    else
        GPU_TYPE="unknown"
        GPU_TOTAL="0"
    fi
    
    # Parse used GPUs
    if [[ $GRES_USED =~ gpu:([^:]+):([0-9]+) ]]; then
        GPU_USED="${BASH_REMATCH[2]}"
    else
        GPU_USED="0"
    fi
    
    GPU_AVAILABLE=$((GPU_TOTAL - GPU_USED))
    
    # Calculate percentages
    if [ "$CPU_TOT" -gt 0 ]; then
        CPU_PERCENT=$((CPU_ALLOC * 100 / CPU_TOT))
    else
        CPU_PERCENT=0
    fi
    
    if [ "$REAL_MEM" -gt 0 ]; then
        MEM_PERCENT=$((MEM_ALLOC * 100 / REAL_MEM))
        MEM_GB=$((REAL_MEM / 1024))
    else
        MEM_PERCENT=0
        MEM_GB=0
    fi
    
    # Display information
    echo "  State:       $STATE"
    echo "  GPU Type:    $GPU_TYPE"
    echo "  GPUs:        $GPU_AVAILABLE available (${GPU_USED}/${GPU_TOTAL} in use)"
    echo "  CPUs:        ${CPU_ALLOC}/${CPU_TOT} allocated (${CPU_PERCENT}%)"
    echo "  Memory:      ${MEM_GB}GB total (${MEM_PERCENT}% allocated)"
    
    # Categorize node
    if [[ $STATE =~ DOWN ]]; then
        echo -e "  Status:      ${RED}DOWN - Not available${NC}"
        DOWN_NODES+=("$node")
    elif [ "$GPU_AVAILABLE" -gt 0 ]; then
        echo -e "  Status:      ${GREEN}✓ AVAILABLE - ${GPU_AVAILABLE} GPU(s) free${NC}"
        AVAILABLE_NODES+=("$node ($GPU_AVAILABLE x $GPU_TYPE)")
    else
        echo -e "  Status:      ${YELLOW}⏳ BUSY - All GPUs in use${NC}"
        BUSY_NODES+=("$node")
    fi
    
    echo ""
done

echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo ""

# Show available nodes
if [ ${#AVAILABLE_NODES[@]} -gt 0 ]; then
    echo -e "${GREEN}✓ Nodes with Available GPUs:${NC}"
    for node in "${AVAILABLE_NODES[@]}"; do
        echo "  - $node"
    done
    echo ""
else
    echo -e "${YELLOW}⚠️  No nodes with available GPUs right now${NC}"
    echo ""
fi

# Show busy nodes
if [ ${#BUSY_NODES[@]} -gt 0 ]; then
    echo -e "${YELLOW}⏳ Busy Nodes (all GPUs in use):${NC}"
    for node in "${BUSY_NODES[@]}"; do
        echo "  - $node"
    done
    echo ""
fi

# Show down nodes
if [ ${#DOWN_NODES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Down/Unavailable Nodes:${NC}"
    for node in "${DOWN_NODES[@]}"; do
        echo "  - $node"
    done
    echo ""
fi

echo "======================================================================"
echo "CPU-Only Nodes (No GPU Required)"
echo "======================================================================"
echo ""

# Check for idle or lightly used CPU nodes
echo "Checking for available CPU nodes..."
echo ""

IDLE_NODES=$(sinfo -N -h -o "%N %T %c %m" | grep -E "idle|mixed" | grep -v gpu | head -10)

if [ -n "$IDLE_NODES" ]; then
    echo "Available CPU nodes (idle or partially used):"
    echo "----------------------------------------------------------------------"
    printf "%-10s %-12s %-8s %-10s\n" "NODE" "STATE" "CPUs" "MEMORY(MB)"
    echo "----------------------------------------------------------------------"
    echo "$IDLE_NODES"
    echo "----------------------------------------------------------------------"
else
    echo "No idle CPU nodes available right now."
fi

echo ""
echo "======================================================================"
echo "Recommendations"
echo "======================================================================"
echo ""

if [ ${#AVAILABLE_NODES[@]} -gt 0 ]; then
    echo "✓ For GPU jobs, run:"
    echo "    bash slurm/start_jupyter.sh"
    echo ""
    echo "  Available GPU nodes: ${AVAILABLE_NODES[0]}"
    for i in "${!AVAILABLE_NODES[@]}"; do
        if [ $i -gt 0 ]; then
            echo "                       ${AVAILABLE_NODES[$i]}"
        fi
    done
else
    echo "⚠️  All GPU nodes are currently busy or down."
    echo ""
    echo "  Options:"
    echo "    1. Wait for GPU resources to free up"
    echo "    2. Use CPU-only mode (faster for development):"
    echo "         bash slurm/start_jupyter.sh nogpu"
    echo ""
    echo "  To monitor GPU availability:"
    echo "    watch -n 30 'bash slurm/check_gpu_availability.sh'"
fi

echo ""
echo "======================================================================"
echo "Additional Cluster Information"
echo "======================================================================"
echo ""

# Show overall cluster status
echo "Overall cluster status:"
echo "----------------------------------------------------------------------"
sinfo -o "%20P %5a %.10l %16F %N" | head -5
echo "----------------------------------------------------------------------"
echo ""

# Show current queue
echo "Current job queue (your jobs):"
echo "----------------------------------------------------------------------"
QUEUE=$(squeue -u $USER -o "%.18i %.9P %.20j %.8T %.10M %.10l %.6D %R" 2>/dev/null)
if [ -n "$QUEUE" ]; then
    echo "$QUEUE"
else
    echo "No jobs in queue"
fi
echo "----------------------------------------------------------------------"
echo ""
