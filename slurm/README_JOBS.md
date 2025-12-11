# Slurm Job Management

This directory contains Slurm job scripts and wrapper scripts for easy execution.

## Directory Structure

```
slurm/
├── jobs/              # Actual sbatch job scripts (run on compute nodes)
├── scripts/           # Wrapper scripts (run from login node)
├── clusters_info/     # Cluster analysis tools
└── logs/              # Job output and error logs
```

## Quick Start

### Detect CUDA Version
```bash
bash slurm/scripts/detect_cuda.sh
```

### Get Full GPU Information
```bash
bash slurm/scripts/gpu_info.sh
```

### Start MLflow Server
```bash
sbatch slurm/mlflow_server.sh
```

### Start Training (8 GPUs)
```bash
sbatch slurm/train_8gpu.sh
```

## Jobs Directory

Contains actual `sbatch` scripts that run on compute nodes:

- **`detect_cuda.sh`** - Detects CUDA toolkit and GPU driver versions
- **`gpu_info.sh`** - Comprehensive GPU and system information
- **`mlflow_server.sh`** - MLflow tracking server
- **`train_8gpu.sh`** - Multi-GPU training job

## Scripts Directory

Contains wrapper scripts that submit jobs and wait for results:

- **`detect_cuda.sh`** - Submits CUDA detection job and displays results
- **`gpu_info.sh`** - Submits GPU info job and displays results

## CUDA Detection

To determine what CUDA version to use for conda environment:

```bash
# Run detection
bash slurm/scripts/detect_cuda.sh

# Output includes:
# - GPU driver version
# - CUDA toolkit version (if available)
# - Recommended PyTorch installation commands
# - GPU hardware details
```

Based on the output, create your conda environment:

```bash
# Example for CUDA 12.1
conda create -n latent-gpt python=3.10
conda activate latent-gpt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Example for CUDA 11.8
conda create -n latent-gpt python=3.10
conda activate latent-gpt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Usage Patterns

### Submit a job and forget
```bash
sbatch slurm/jobs/detect_cuda.sh
# Check logs later in slurm/logs/
```

### Submit and wait for results
```bash
bash slurm/scripts/detect_cuda.sh
# Script waits and displays results automatically
```

### Monitor running jobs
```bash
squeue -u $USER
```

### Cancel a job
```bash
scancel <job_id>
```

### View logs
```bash
# Most recent logs
ls -lt slurm/logs/ | head

# View specific log
cat slurm/logs/detect_cuda_<job_id>.out
```

## Job Configuration

All job scripts have configurable SBATCH directives:

```bash
#SBATCH --job-name=<name>
#SBATCH --partition=<partition>
#SBATCH --time=<time>
#SBATCH --gres=gpu:<count>
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<memory>
```

Modify these based on your cluster's available resources (use `slurm/clusters_info/find_best_partition.sh`).

## Troubleshooting

### Job fails immediately
Check error log: `slurm/logs/<job_name>_<job_id>.err`

### Conda not found
Update conda paths in job scripts or use system Python

### GPU not allocated
Ensure `--gres=gpu:X` matches available resources

### Module errors
Check if your cluster uses environment modules:
```bash
module avail
module load cuda
```
