#!/bin/bash
#
# Check All Partitions
# Script to list all available Slurm partitions with detailed information
#

echo "=========================================="
echo "SLURM PARTITION INFORMATION"
echo "=========================================="
echo ""

# Get partition list with detailed format
echo "Available Partitions:"
echo "--------------------"
sinfo -o "%.12P %.5a %.10l %.6D %.6t %.8c %.8m %.10G %.20f" | head -1
echo "--------------------"
sinfo -o "%.12P %.5a %.10l %.6D %.6t %.8c %.8m %.10G %.20f" | grep -v "PARTITION"

echo ""
echo "Legend:"
echo "  P     = Partition name"
echo "  AVAIL = Availability (up/down)"
echo "  TIMELIMIT = Maximum job time"
echo "  NODES = Number of nodes"
echo "  STATE = Node state (idle/alloc/mix/drain)"
echo "  CPUS  = Number of CPUs per node"
echo "  MEMORY = Memory per node (MB)"
echo "  GRES  = Generic resources (GPUs)"
echo "  AVAIL_FEATURES = Node features"
echo ""
