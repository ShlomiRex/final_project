# Cluster Information Scripts

This directory contains utility scripts to analyze and find the best Slurm cluster resources for training.

## Scripts

### 🚀 Quick Start
```bash
# Run comprehensive analysis and get best partition recommendation
bash run_all.sh
```

### Individual Scripts

#### `run_all.sh`
Master script that runs all analysis tools in sequence. **Start here!**

**Output:**
- Partition list with availability
- GPU resources breakdown
- Detailed node information (optional)
- Best partition recommendation with full specs

#### `find_best_partition.sh`
**Primary script** - Analyzes all partitions and recommends the one with most resources.

**Scoring criteria:**
1. GPU count (highest priority)
2. RAM per node
3. CPU count
4. Number of nodes

**Output:**
- Ranked list of partitions
- Best partition with complete specifications
- Ready-to-use `#SBATCH` directives

**Example:**
```bash
bash find_best_partition.sh
```

#### `check_all_partitions.sh`
Lists all available Slurm partitions with key statistics.

**Shows:**
- Partition name and availability
- Time limits
- Node count and states
- CPU and memory resources
- GPU resources (GRES)

#### `check_gpu_resources.sh`
Detailed analysis of GPU resources across all partitions.

**Shows:**
- GPU type and count per partition
- Node-level GPU allocation
- Total GPU inventory
- Available vs. allocated resources

#### `check_node_details.sh`
Deep dive into individual node specifications.

**Shows:**
- Per-node CPU, memory, GPU specs
- Current node states
- Partition assignments

#### `quick_status.sh`
Fast overview of current cluster utilization.

**Shows:**
- Your active jobs
- Available GPU resources
- Overall cluster load

**Example:**
```bash
bash quick_status.sh
```

## Usage Examples

### Find Best Partition for Training
```bash
cd /home/doshlom4/work/final_project/slurm/clusters_info
bash find_best_partition.sh
```

### Check GPU Availability
```bash
bash check_gpu_resources.sh
```

### Monitor Your Jobs
```bash
bash quick_status.sh
```

### Complete Analysis
```bash
bash run_all.sh
```

## Integration with Training Scripts

After finding the best partition, update your training script:

```bash
# Example output from find_best_partition.sh:
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:8
```

Copy these lines to your training script (e.g., `train_8gpu.sh`).

## Understanding the Output

### Partition States
- `idle` - Available for immediate use
- `alloc` - Fully allocated
- `mix` - Partially allocated
- `drain` - Being taken offline

### GPU Types
Common types you might see:
- `a100` - NVIDIA A100 (80GB or 40GB)
- `v100` - NVIDIA V100 (32GB or 16GB)
- `rtx8000` - NVIDIA RTX 8000
- `a6000` - NVIDIA A6000

### Resource Scoring
The `find_best_partition.sh` script uses this formula:
```
score = (GPUs × nodes × 10000) + (RAM_MB × nodes) + (CPUs × nodes × 10)
```

This heavily prioritizes GPU count, which is most important for deep learning.

## Troubleshooting

### No partitions found
```bash
# Check if Slurm is configured
sinfo --version
```

### Permission errors
```bash
# Make scripts executable
chmod +x *.sh
```

### Incorrect GPU counts
Some clusters hide GPU info in `sinfo`. Try:
```bash
scontrol show nodes | grep -i gres
```

## Notes

- Scripts use standard Slurm commands (`sinfo`, `squeue`, `scontrol`)
- All scripts are non-destructive (read-only)
- Safe to run multiple times
- Results may vary based on current cluster load

## Quick Reference Commands

```bash
# List all partitions
sinfo

# Show node details
scontrol show nodes

# Check your jobs
squeue -u $USER

# GPU availability
sinfo -o "%P %G %D %t"

# Node features
sinfo -o "%N %f %G"
```
