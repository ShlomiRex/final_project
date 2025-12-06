# Multi-GPU Training Scripts - Usage Guide

## Overview
This directory contains Slurm batch scripts for launching multi-GPU training on the HPC cluster.

## Available Scripts

### 1. `train1_flickr8_multi_gpu.sh` - Standard Multi-GPU Training
Basic multi-GPU training script with 2 GPUs.

**Usage:**
```bash
cd /home/doshlom4/work/final_project
sbatch slurm/train1_flickr8_multi_gpu.sh
```

**Features:**
- 2 GPUs requested
- 8 CPUs total (4 per GPU)
- 64GB RAM
- 72-hour max runtime
- Automatic notebook-to-script conversion
- Full diagnostics

---

### 2. `train1_flickr8_2gpu_oom_safe.sh` - OOM-Safe Multi-GPU Training ⭐ **RECOMMENDED**
Enhanced script with out-of-memory (OOM) prevention measures.

**Usage:**
```bash
cd /home/doshlom4/work/final_project
sbatch slurm/train1_flickr8_2gpu_oom_safe.sh
```

**Features:**
- GPU memory cleanup before training
- Detailed memory diagnostics
- Better error messages
- Same resource allocation as standard script

**When to use:** If you encounter CUDA OOM errors like:
```
RuntimeError: CUDA error: out of memory
```

---

## Understanding the OOM Error You Encountered

Your error output showed:
```
GPU 0: NVIDIA TITAN V, 12066 MiB total, 11472 MiB used, 594 MiB free
GPU 1: NVIDIA TITAN V, 12066 MiB total, 4 MiB used, 12062 MiB free
```

**Problem:** GPU 0 already had 11.4GB allocated (from Jupyter session), leaving only 594MB free!

**Solutions:**

### Option 1: Kill Jupyter Before Training (Recommended)
If you're running Jupyter on the same GPU node:
```bash
# Find your Jupyter job
squeue -u $USER

# Cancel it
scancel <job_id>

# Then submit training
sbatch slurm/train1_flickr8_2gpu_oom_safe.sh
```

### Option 2: Reduce Batch Size in Notebook
Edit `notebooks/train1_flickr8.ipynb` cell 22 (Configuration):

**Change from:**
```python
batch_size=16,  # Current setting
```

**Change to:**
```python
batch_size=8,   # Reduced for multi-GPU with limited memory
```

**Effect:** With 2 GPUs, effective batch size = 8 × 2 = 16 (still reasonable for training)

### Option 3: Request Different GPUs
Modify the Slurm script to request GPUs with more memory:
```bash
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu32g    # Request 32GB GPUs (if available)
```

Or request specific GPU types:
```bash
#SBATCH --gres=gpu:titanv:2    # Specific GPU model
#SBATCH --gres=gpu:v100:2      # V100 GPUs (32GB)
#SBATCH --gres=gpu:a100:2      # A100 GPUs (40GB/80GB)
```

Check available GPU types:
```bash
sinfo -o "%N %G %m %C"
```

---

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
```

### View Live Logs
```bash
# Standard output
tail -f slurm/logs/train1_flickr8_2gpu_<JOB_ID>.out

# Errors
tail -f slurm/logs/train1_flickr8_2gpu_<JOB_ID>.err
```

### Check GPU Usage on Compute Node
```bash
# SSH to the node (replace gpu2 with your assigned node)
ssh gpu2

# Watch GPU usage
watch -n 1 nvidia-smi
```

### TensorBoard (from login node)
```bash
# Find your TensorBoard directory
ls -lt outputs/train12_flickr8k_text2img/tensorboard/

# Start TensorBoard
tensorboard --logdir outputs/train12_flickr8k_text2img/tensorboard/ --port 6006

# Then SSH tunnel from your local machine
ssh -L 6006:localhost:6006 <username>@<hpc_login_node>
```

---

## Troubleshooting

### Script Conversion Issues
If `from __future__ import annotations` placement causes errors:
```bash
# The notebook MUST have this as the FIRST code cell (Cell 1)
# The OOM-safe script checks this automatically
```

### Accelerate Configuration Missing
```bash
accelerate config
```
Answer the prompts (choose multi-GPU, 2 processes, fp16, etc.)

### Permission Denied
```bash
chmod +x slurm/train1_flickr8_*.sh
```

### Dataset Not Found
The scripts expect dataset cache at:
```
/home/doshlom4/work/huggingface_cache/nlphuji___flickr30k/
```

If missing, the script will re-download automatically.

---

## Advanced: Manual Multi-GPU Launch (No Slurm)

If you have an **interactive session** with 2 GPUs already allocated:

```bash
# 1. Convert notebook
cd /home/doshlom4/work/final_project/notebooks
jupyter nbconvert --to script train1_flickr8.ipynb --output train1_flickr8_script

# 2. Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# 3. Launch training
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 train1_flickr8_script.py
```

**Note:** This still won't work if Jupyter is consuming GPU memory. Kill Jupyter first!

---

## Performance Expectations

| Configuration | GPUs | Batch Size (per GPU) | Effective Batch Size | Speed |
|--------------|------|----------------------|----------------------|-------|
| Notebook     | 1    | 16                   | 16                   | 1x    |
| Script (1 GPU) | 1  | 16                   | 16                   | 1x    |
| Script (2 GPUs) | 2 | 16                   | 32                   | ~1.8x |
| Script (2 GPUs, low mem) | 2 | 8        | 16                   | ~1.5x |

**Why not 2x speedup?**
- Communication overhead between GPUs (~10-20%)
- Gradient synchronization
- I/O bottlenecks (data loading)

---

## Best Practices

1. **Always kill Jupyter before batch jobs** - Prevents GPU memory conflicts
2. **Use Slurm for long training** - Immune to SSH disconnections
3. **Monitor GPU usage** - Catch OOM early
4. **Save checkpoints frequently** - Job time limits (72h max)
5. **Use TensorBoard** - Track progress remotely
6. **Test with 1 epoch first** - Verify end-to-end pipeline before long runs

---

## Quick Reference

```bash
# Submit job
sbatch slurm/train1_flickr8_2gpu_oom_safe.sh

# Check status
squeue -u $USER

# Cancel job
scancel <job_id>

# View logs
tail -f slurm/logs/train1_flickr8_2gpu_*.out

# Check GPU (on compute node)
ssh <node_name>
nvidia-smi
```

---

## Related Documentation

- **HPC Jupyter Setup:** `slurm/JUPYTER_SETUP.md`
- **GPU Node Selection:** `slurm/NODE_SELECTION_GUIDE.md`
- **Project Overview:** `README.md`
- **AI Agent Instructions:** `.github/copilot-instructions.md`

---

**Last Updated:** 2025-11-13  
**Maintained by:** HPC Training Pipeline Team
