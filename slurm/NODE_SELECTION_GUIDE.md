# Node Selection Guide for Jupyter on HPC

## Quick Commands

### Automatic Node Selection (Recommended)
```bash
# Any available GPU (Slurm chooses best available)
bash slurm/start_jupyter.sh

# Any available CPU node (faster startup)
bash slurm/start_jupyter.sh nogpu
```

### Specific Node Selection
```bash
# Request specific GPU node
bash slurm/start_jupyter.sh gpu8    # 8x A100 GPUs
bash slurm/start_jupyter.sh gpu7    # 8x A100 GPUs + 2TB RAM (best for large models)
bash slurm/start_jupyter.sh gpu1    # 2x Titan GPUs

# Request specific CPU node
bash slurm/start_jupyter.sh cn43    # 80 CPUs
bash slurm/start_jupyter.sh cn31    # 80 CPUs
```

## GPU Node Comparison

| Node  | GPU Type | GPU Count | Total RAM | Best For |
|-------|----------|-----------|-----------|----------|
| gpu7  | A100     | 8         | 2 TB      | 🏆 Large models, multi-GPU training |
| gpu8  | A100     | 8         | 251 GB    | Medium-large models, inference |
| gpu1-4| Titan    | 2 each    | 188 GB    | Small models, development |
| gpu6  | A100     | 2         | 188 GB    | ❌ Currently DOWN |

## CPU Node Comparison

| Nodes   | CPU Count | RAM      | State | Best For |
|---------|-----------|----------|-------|----------|
| cn31-44 | 80 each   | 193 GB   | Mixed | Development, data processing |
| cn05-30 | 64 each   | 193 GB   | Mixed | Light computation |
| cn43-44 | 80 each   | 193-241 GB | Idle | 🏆 Available now |

## Usage Examples

### Example 1: Large Model Training
You need multiple A100 GPUs with lots of RAM:
```bash
bash slurm/check_gpu_availability.sh  # Check if gpu7 is available
bash slurm/start_jupyter.sh gpu7      # Request gpu7 specifically
```

### Example 2: Quick Development
You just need to write code, no GPU required:
```bash
bash slurm/start_jupyter.sh nogpu     # Fast startup on any CPU node
```

### Example 3: Small Model Testing
You need a GPU but don't want to wait for A100s:
```bash
bash slurm/start_jupyter.sh gpu1      # Titan GPUs usually available faster
```

### Example 4: Specific Node for Reproducibility
You want the same node every time:
```bash
bash slurm/start_jupyter.sh cn43      # Always use cn43
```

## Decision Tree

```
Do you need a GPU?
│
├─ YES
│  │
│  ├─ Large model (>10GB VRAM)?
│  │  └─> bash slurm/start_jupyter.sh gpu7  (8x A100, 2TB RAM)
│  │
│  ├─ Medium model?
│  │  └─> bash slurm/start_jupyter.sh gpu8  (8x A100)
│  │
│  └─ Small model or don't know?
│     └─> bash slurm/start_jupyter.sh       (any GPU)
│
└─ NO
   │
   ├─ Just coding/development?
   │  └─> bash slurm/start_jupyter.sh nogpu  (fast)
   │
   └─ Heavy CPU computation?
      └─> bash slurm/start_jupyter.sh cn43   (80 CPUs)
```

## Tips

### 1. Check Availability First
Always check what's available before requesting specific nodes:
```bash
bash slurm/check_gpu_availability.sh
```

### 2. Be Flexible
If a specific node is busy, let Slurm choose:
```bash
# Instead of: bash slurm/start_jupyter.sh gpu7
# Use:        bash slurm/start_jupyter.sh
# (Slurm will find any available GPU node)
```

### 3. Start Fast, Scale Later
For development:
```bash
bash slurm/start_jupyter.sh nogpu  # Develop code here
```

When ready to train:
```bash
scancel <job_id>                   # Stop nogpu job
bash slurm/start_jupyter.sh gpu7   # Start with big GPUs
```

### 4. Monitor Your Jobs
```bash
squeue -u $USER                    # See all your jobs
scontrol show job <job_id>         # Detailed job info
```

## Common Issues

### "Job is pending for Resources"
Your requested node is busy. Options:
1. Wait for it to become available
2. Request a different node
3. Use automatic selection (remove node specification)

```bash
# Check what's available
bash slurm/check_gpu_availability.sh

# Cancel pending job
scancel <job_id>

# Try a different node or let Slurm choose
bash slurm/start_jupyter.sh
```

### "Invalid node name"
Make sure you're using the correct format:
- GPU nodes: `gpu1`, `gpu2`, ..., `gpu8`
- CPU nodes: `cn01`, `cn02`, ..., `cn44`

### "Node is DOWN"
The node is offline for maintenance. Check availability:
```bash
bash slurm/check_gpu_availability.sh
```

## Advanced: Multi-GPU Jobs

To request multiple GPUs on a specific node, you'll need to modify the Slurm script:

```bash
# Edit jupyter_notebook_interactive.sh
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --nodelist=gpu7           # On gpu7 specifically
```

Or create a custom launcher:
```bash
sbatch --nodelist=gpu7 --gres=gpu:4 --export=JUPYTER_PORT=9999 slurm/jupyter_notebook_interactive.sh
```

## See Also

- [JUPYTER_SETUP.md](JUPYTER_SETUP.md) - Full Jupyter setup documentation
- [check_gpu_availability.sh](check_gpu_availability.sh) - Real-time cluster status
- [Slurm Documentation](https://slurm.schedmd.com/) - Official Slurm docs
