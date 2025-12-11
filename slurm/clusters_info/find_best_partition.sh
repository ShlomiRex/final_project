#!/bin/bash
#
# Find Best Partition
# Analyzes all partitions and recommends the one with the most resources
#

echo "=========================================="
echo "FINDING BEST PARTITION FOR TRAINING"
echo "=========================================="
echo ""

# Temporary file to store partition scores
temp_file=$(mktemp)

# Get unique partitions
partitions=$(sinfo -h -o "%P" | sort -u | tr -d '*')

echo "Analyzing partitions..."
echo ""

for partition in $partitions; do
    # Get partition info
    info=$(sinfo -p "$partition" -h -o "%N %G %C %m %t %l" | head -1)
    
    if [ -z "$info" ]; then
        continue
    fi
    
    node=$(echo "$info" | awk '{print $1}')
    gres=$(echo "$info" | awk '{print $2}')
    cpus=$(echo "$info" | awk '{print $3}')
    mem=$(echo "$info" | awk '{print $4}')
    state=$(echo "$info" | awk '{print $5}')
    timelimit=$(echo "$info" | awk '{print $6}')
    
    # Extract GPU information
    gpu_count=0
    gpu_type="none"
    if [[ "$gres" == *"gpu"* ]]; then
        gpu_info=$(echo "$gres" | grep -o 'gpu:[^:]*:[0-9]*')
        if [ -n "$gpu_info" ]; then
            gpu_type=$(echo "$gpu_info" | cut -d: -f2)
            gpu_count=$(echo "$gpu_info" | cut -d: -f3)
        fi
    fi
    
    # Extract total CPUs
    total_cpus=$(echo "$cpus" | awk -F'/' '{print $4}')
    avail_cpus=$(echo "$cpus" | awk -F'/' '{print $2}')
    
    # Count number of nodes in partition
    node_count=$(sinfo -p "$partition" -h -o "%D")
    
    # Calculate score (prioritize GPUs, then RAM, then CPUs)
    # GPU weight: 10000, RAM weight: 1, CPU weight: 10
    score=$((gpu_count * node_count * 10000 + mem * node_count + total_cpus * node_count * 10))
    
    # Store results
    printf "%s|%s|%d|%s|%d|%d|%d|%s|%s|%d\n" \
        "$partition" "$gpu_type" "$gpu_count" "$node" "$total_cpus" "$avail_cpus" "$mem" "$timelimit" "$state" "$score" >> "$temp_file"
done

# Sort by score (highest first) and display
echo "Partition Ranking:"
echo "--------------------"
printf "%-15s | %-10s | %8s | %8s | %10s | %12s | %s\n" \
    "PARTITION" "GPU_TYPE" "GPUs" "CPUs" "RAM (GB)" "TIME_LIMIT" "STATE"
echo "--------------------------------------------------------------------------------"

sort -t'|' -k10 -nr "$temp_file" | while IFS='|' read partition gpu_type gpu_count node cpus avail_cpus mem timelimit state score; do
    mem_gb=$(awk "BEGIN {printf \"%.1f\", $mem/1024}")
    printf "%-15s | %-10s | %8d | %8d | %10s | %12s | %s\n" \
        "$partition" "$gpu_type" "$gpu_count" "$cpus" "$mem_gb" "$timelimit" "$state"
done

echo ""
echo "=========================================="
echo "BEST PARTITION RECOMMENDATION"
echo "=========================================="

# Get the best partition
best=$(sort -t'|' -k10 -nr "$temp_file" | head -1)

if [ -n "$best" ]; then
    IFS='|' read partition gpu_type gpu_count node cpus avail_cpus mem timelimit state score <<< "$best"
    
    mem_gb=$(awk "BEGIN {printf \"%.1f\", $mem/1024}")
    node_count=$(sinfo -p "$partition" -h -o "%D")
    total_gpus=$((gpu_count * node_count))
    total_cpus_all=$((cpus * node_count))
    total_mem_all=$((mem * node_count / 1024))
    
    echo ""
    echo "🏆 BEST PARTITION: $partition"
    echo ""
    echo "Resources per node:"
    echo "  - GPU Type:      $gpu_type"
    echo "  - GPUs:          $gpu_count"
    echo "  - CPUs:          $cpus (currently $avail_cpus available)"
    echo "  - RAM:           $mem_gb GB"
    echo "  - Time Limit:    $timelimit"
    echo "  - State:         $state"
    echo ""
    echo "Total cluster resources in this partition:"
    echo "  - Nodes:         $node_count"
    echo "  - Total GPUs:    $total_gpus"
    echo "  - Total CPUs:    $total_cpus_all"
    echo "  - Total RAM:     $total_mem_all GB"
    echo ""
    echo "To use this partition in your Slurm script:"
    echo "  #SBATCH --partition=$partition"
    
    if [ "$gpu_count" -gt 0 ]; then
        echo "  #SBATCH --gres=gpu:$gpu_type:$gpu_count"
    fi
    
    echo ""
else
    echo "No partitions found!"
fi

# Cleanup
rm -f "$temp_file"

echo ""
