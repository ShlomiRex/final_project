#!/bin/bash
#
# Master Script - Run All Cluster Analysis
# Executes all analysis scripts to get comprehensive cluster information
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "######################################################################"
echo "#                                                                    #"
echo "#          SLURM CLUSTER RESOURCE ANALYSIS                          #"
echo "#                                                                    #"
echo "######################################################################"
echo ""
echo "Running comprehensive analysis of available cluster resources..."
echo ""

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/*.sh

# Run partition check
echo ""
bash "$SCRIPT_DIR/check_all_partitions.sh"
echo ""

# Run GPU resources check
echo ""
bash "$SCRIPT_DIR/check_gpu_resources.sh"
echo ""

# Run node details (optional, can be verbose)
read -p "Show detailed node information? (y/n) [n]: " show_details
if [[ "$show_details" =~ ^[Yy]$ ]]; then
    echo ""
    bash "$SCRIPT_DIR/check_node_details.sh"
    echo ""
fi

# Find best partition
echo ""
bash "$SCRIPT_DIR/find_best_partition.sh"
echo ""

echo "######################################################################"
echo "Analysis complete!"
echo "######################################################################"
echo ""
