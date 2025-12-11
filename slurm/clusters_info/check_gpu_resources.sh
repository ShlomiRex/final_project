#!/bin/bash
#
# GPU Information by Partition
# Script to analyze GPU resources available in each partition
#

echo "=========================================="
echo "GPU RESOURCES BY PARTITION"
echo "=========================================="
echo ""

# Get unique partitions
partitions=$(sinfo -h -o "%P" | sort -u | tr -d '*')

for partition in $partitions; do
    echo "Partition: $partition"
    echo "--------------------"
    
    # Get nodes in this partition
    sinfo -p "$partition" -h -o "%N %G %C %m %t" | while read node gres cpus mem state; do
        # Parse GRES to extract GPU info
        gpu_info=$(echo "$gres" | grep -o 'gpu:[^:]*:[0-9]*' || echo "no_gpu")
        
        if [ "$gpu_info" != "no_gpu" ]; then
            gpu_type=$(echo "$gpu_info" | cut -d: -f2)
            gpu_count=$(echo "$gpu_info" | cut -d: -f3)
            
            # Parse CPU info (allocated/idle/other/total)
            total_cpus=$(echo "$cpus" | awk -F'/' '{print $4}')
            avail_cpus=$(echo "$cpus" | awk -F'/' '{print $2}')
            
            printf "  Node: %-20s | GPUs: %2dx %-10s | CPUs: %4d (%4d avail) | RAM: %8d MB | State: %s\n" \
                "$node" "$gpu_count" "$gpu_type" "$total_cpus" "$avail_cpus" "$mem" "$state"
        fi
    done
    
    echo ""
done

echo ""
echo "=========================================="
echo "GPU SUMMARY"
echo "=========================================="

# Count total GPUs by type
sinfo -h -o "%G %D" | grep gpu | while read gres nodes; do
    gpu_type=$(echo "$gres" | grep -o 'gpu:[^:]*:[0-9]*' | cut -d: -f2)
    gpu_per_node=$(echo "$gres" | grep -o 'gpu:[^:]*:[0-9]*' | cut -d: -f3)
    
    if [ -n "$gpu_type" ] && [ -n "$gpu_per_node" ]; then
        total_gpus=$((gpu_per_node * nodes))
        printf "%dx %s GPUs (%d GPUs per node across %d nodes)\n" "$total_gpus" "$gpu_type" "$gpu_per_node" "$nodes"
    fi
done | sort -u

echo ""
