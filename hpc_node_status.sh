#!/bin/bash
# hpc_node_status.sh
# List all nodes, their GPU, RAM, available resources, and job count on the cluster

echo "================================================================================"
echo "                        HPC CLUSTER NODE STATUS"
echo "================================================================================"
echo ""

# Arrays to store node data
declare -a cpu_nodes
declare -a gpu_nodes

# Get node list and info from sinfo
while IFS= read -r line; do
    node=$(echo "$line" | awk '{print $1}')
    state=$(echo "$line" | awk '{print $2}')
    cpus=$(echo "$line" | awk '{print $3}')
    mem=$(echo "$line" | awk '{print $4}')
    gpus=$(echo "$line" | awk '{print $5}')
    
    # Convert memory to GB
    ram_gb=$((mem / 1024))
    
    # Get number of running jobs on this node
    jobs=$(squeue -w "$node" -h 2>/dev/null | wc -l)
    
    # Parse GPU info to make it human-readable
    if [[ "$gpus" == "(null)" ]] || [[ -z "$gpus" ]]; then
        gpu_display="None"
        cpu_nodes+=("$node|$state|$cpus|$ram_gb|$gpu_display|$jobs")
    else
        # Extract GPU count and type (e.g., gpu:a100:2 -> 2x A100)
        gpu_count=$(echo "$gpus" | grep -oP 'gpu:[^:]+:\K\d+' | head -n1)
        gpu_type=$(echo "$gpus" | grep -oP 'gpu:\K[^:]+' | head -n1 | tr '[:lower:]' '[:upper:]')
        
        if [[ -n "$gpu_count" ]] && [[ -n "$gpu_type" ]]; then
            gpu_display="${gpu_count}x ${gpu_type}"
        else
            gpu_display="GPU"
        fi
        
        gpu_nodes+=("$node|$state|$cpus|$ram_gb|$gpu_display|$jobs")
    fi
done < <(sinfo -N -o '%N %T %c %m %G' | tail -n +2)

# Print CPU nodes
if [ ${#cpu_nodes[@]} -gt 0 ]; then
    echo "CPU NODES:"
    echo "--------------------------------------------------------------------------------"
    printf "%-15s %-12s %-8s %-10s %-15s %-8s\n" "Node" "State" "CPUs" "RAM (GB)" "GPUs" "Jobs"
    echo "--------------------------------------------------------------------------------"
    
    for node_data in "${cpu_nodes[@]}"; do
        IFS='|' read -r node state cpus ram gpu jobs <<< "$node_data"
        printf "%-15s %-12s %-8s %-10s %-15s %-8s\n" "$node" "$state" "$cpus" "$ram" "$gpu" "$jobs"
    done
    echo ""
fi

# Print GPU nodes with detailed information
if [ ${#gpu_nodes[@]} -gt 0 ]; then
    echo "GPU NODES (DETAILED):"
    echo "================================================================================"
    
    for node_data in "${gpu_nodes[@]}"; do
        IFS='|' read -r node state cpus ram gpu jobs <<< "$node_data"
        
        echo ""
        echo "Node: $node"
        echo "--------------------------------------------------------------------------------"
        
        # Get detailed node info from scontrol
        node_info=$(scontrol show node "$node" 2>/dev/null)
        
        # Extract allocated and total CPUs
        alloc_cpus=$(echo "$node_info" | grep -oP 'CPUAlloc=\K\d+')
        total_cpus=$(echo "$node_info" | grep -oP 'CPUTot=\K\d+')
        avail_cpus=$((total_cpus - alloc_cpus))
        
        # Extract allocated and total memory (in MB)
        alloc_mem=$(echo "$node_info" | grep -oP 'AllocMem=\K\d+')
        total_mem=$(echo "$node_info" | grep -oP 'RealMemory=\K\d+')
        avail_mem=$((total_mem - alloc_mem))
        avail_mem_gb=$((avail_mem / 1024))
        total_mem_gb=$((total_mem / 1024))
        
        # Extract GPU info
        gres=$(echo "$node_info" | grep -oP 'Gres=\K[^\s]+')
        gres_used=$(echo "$node_info" | grep -oP 'GresUsed=\K[^\s]+')
        
        # Parse GPU allocation
        if [[ -n "$gres" ]]; then
            total_gpus=$(echo "$gres" | grep -oP 'gpu:[^:]+:\K\d+' | head -n1)
            if [[ -n "$gres_used" ]]; then
                used_gpus=$(echo "$gres_used" | grep -oP 'gpu:[^(]+\(IDX:\K[^)]+' | tr ',' '\n' | wc -l)
            else
                used_gpus=0
            fi
            avail_gpus=$((total_gpus - used_gpus))
        else
            total_gpus=0
            used_gpus=0
            avail_gpus=0
        fi
        
        # Print detailed info
        printf "  State:           %-12s\n" "$state"
        printf "  Running Jobs:    %-8s\n" "$jobs"
        printf "  GPUs:            %s / %s available (%s)\n" "$avail_gpus" "$total_gpus" "$gpu"
        printf "  CPUs:            %s / %s available\n" "$avail_cpus" "$total_cpus"
        printf "  RAM:             %s GB / %s GB available\n" "$avail_mem_gb" "$total_mem_gb"

        # Show which GPUs are in use if any
        if [[ "$used_gpus" -gt 0 ]] && [[ -n "$gres_used" ]]; then
            gpu_indices=$(echo "$gres_used" | grep -oP 'gpu:[^(]+\(IDX:\K[^)]+')
            printf "  GPUs in use:     %s\n" "$gpu_indices"
        fi

        # Per-GPU VRAM usage (using nvidia-smi via srun)
        echo "  GPU VRAM usage:"
        srun -N1 -w "$node" --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=1G --exclusive bash -c '
            if command -v nvidia-smi &>/dev/null; then
                nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | \
                awk -F"," "{printf \"    GPU %s (%s): %s MiB used / %s MiB total / %s MiB free\\n\", \$1, \$2, \$3, \$4, \$5}"
            else
                echo "    nvidia-smi not available"
            fi
        '
    done
    echo ""
    echo "================================================================================"
    echo ""
fi

# Print summary
total_nodes=$((${#cpu_nodes[@]} + ${#gpu_nodes[@]}))
echo "================================================================================"
echo "SUMMARY:"
echo "  Total Nodes:     $total_nodes"
echo "  CPU Nodes:       ${#cpu_nodes[@]}"
echo "  GPU Nodes:       ${#gpu_nodes[@]}"
echo "================================================================================"
echo ""
echo "Node States:"
echo "  idle   = Node is available"
echo "  mixed  = Some jobs running, resources still available"
echo "  alloc  = Fully allocated"
echo "  down   = Node is down"
echo ""
echo "For detailed GPU info on a specific node:"
echo "  srun -w <node_name> nvidia-smi"
echo ""
