#!/bin/bash
#
# Quick Status Check
# Fast overview of current cluster utilization
#

echo "=========================================="
echo "QUICK CLUSTER STATUS"
echo "=========================================="
echo ""

echo "Current Queue:"
echo "--------------------"
squeue -u $USER -o "%.8i %.9P %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs in queue"

echo ""
echo "Available GPU Resources:"
echo "--------------------"
sinfo -o "%.12P %.5a %.6D %.6t %.8G" | grep -i gpu || echo "No GPU partitions found"

echo ""
echo "Overall Cluster Load:"
echo "--------------------"
sinfo -o "%.12P %.5a %.10l %.6D %.6t"

echo ""
echo "My Resource Usage:"
echo "--------------------"
squeue -u $USER --format="%.8i %.9P %.20j %.8T %.6D %.8C %.10m %.10l %.10M" 2>/dev/null || echo "No active jobs"

echo ""
