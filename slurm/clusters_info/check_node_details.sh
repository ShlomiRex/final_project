#!/bin/bash
#
# Detailed Node Information
# Script to get detailed information about all nodes in the cluster
#

echo "=========================================="
echo "DETAILED NODE INFORMATION"
echo "=========================================="
echo ""

# Get all nodes with detailed information
scontrol show nodes | awk '
BEGIN {
    node_count = 0
}
/NodeName=/ {
    if (node_count > 0) {
        print "----------------------------------------"
    }
    node_count++
    
    # Extract fields
    match($0, /NodeName=([^ ]+)/, arr); node = arr[1]
    match($0, /Partitions=([^ ]+)/, arr); partition = arr[1]
    match($0, /State=([^ ]+)/, arr); state = arr[1]
    match($0, /CPUTot=([^ ]+)/, arr); cpu = arr[1]
    match($0, /RealMemory=([^ ]+)/, arr); mem = arr[1]
    match($0, /Gres=([^ ]+)/, arr); gres = arr[1]
    
    printf "Node:       %s\n", node
    printf "Partition:  %s\n", partition
    printf "State:      %s\n", state
    printf "CPUs:       %s\n", cpu
    printf "Memory:     %s MB (%.1f GB)\n", mem, mem/1024
    printf "GPUs:       %s\n", (gres != "" ? gres : "none")
}
END {
    print "----------------------------------------"
    printf "\nTotal Nodes: %d\n", node_count
}'

echo ""
