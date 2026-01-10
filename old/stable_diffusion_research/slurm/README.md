# Slurm Job Scripts

This directory contains Slurm job scripts for running Stable Diffusion training on HPC clusters.

## Available Scripts

| Script | Description | Partition | GPUs | Max Time |
|--------|-------------|-----------|------|----------|
| `train_multigpu.sh` | Full training (8×A100) | gpu8 | 8 | 72h |
| `train_2gpu.sh` | Development training | gpu6 | 2 | 48h |
| `evaluate.sh` | Model evaluation | gpu6 | 1 | 4h |
| `generate.sh` | Image generation | gpu6 | 1 | 2h |
| `mlflow_server.sh` | MLflow tracking server | gpu6 | 0 | 72h |

## Usage

### 1. Start MLflow Server (Optional but Recommended)

```bash
# Submit MLflow server job
sbatch slurm/mlflow_server.sh

# Get the allocated node
squeue -u $USER | grep mlflow

# Note the node name (e.g., gpu6)
# Set in training jobs:
export MLFLOW_TRACKING_URI=http://gpu6:5000
```

### 2. Full Training (8×A100)

```bash
# Basic training
sbatch slurm/train_multigpu.sh configs/base.yaml

# With 512 resolution config
sbatch slurm/train_multigpu.sh configs/training/train_512.yaml

# Resume from checkpoint
sbatch slurm/train_multigpu.sh configs/base.yaml --resume latest

# With config overrides
sbatch slurm/train_multigpu.sh configs/base.yaml training.learning_rate=2e-4
```

### 3. Development Training (2 GPUs)

```bash
sbatch slurm/train_2gpu.sh configs/base.yaml
```

### 4. Evaluation

```bash
# Evaluate a checkpoint
sbatch slurm/evaluate.sh outputs/checkpoints/checkpoint_100000.pt

# With custom config
sbatch slurm/evaluate.sh outputs/checkpoints/checkpoint_100000.pt configs/base.yaml
```

### 5. Image Generation

```bash
# Create a prompts file
cat > prompts.txt << EOF
A photo of a cat sitting on a windowsill
A beautiful sunset over the ocean
A mountain landscape with snow peaks
EOF

# Generate images
sbatch slurm/generate.sh outputs/checkpoints/checkpoint_100000.pt prompts.txt
```

## Accessing MLflow UI

From your local machine:

```bash
# SSH tunnel to MLflow server
# Replace <node> with the actual node (e.g., gpu6)
ssh -N -L localhost:5000:<node>.hpc.pub.lan:5000 doshlom4@login9

# Open in browser
open http://localhost:5000
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job output in real-time
tail -f slurm/logs/train_<job_id>.out

# Cancel a job
scancel <job_id>

# Get job details
scontrol show job <job_id>
```

## GPU Resources

### gpu8 Partition (Primary)
- 8× NVIDIA A100 80GB
- 512GB RAM
- Best for full training runs

### gpu6 Partition (Development)
- 2× NVIDIA A100 80GB
- 128GB RAM
- Good for testing and evaluation

## Environment

All scripts use the shared Python environment:
```
/home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/
```

Activate manually with:
```bash
source /home/doshlom4/work/pytorch-env/venv-gpu8-pytorch/bin/activate
```

## Troubleshooting

### Job Won't Start
- Check available resources: `sinfo -p gpu8`
- Check queue: `squeue -p gpu8`
- May need to wait for resources

### Out of Memory
- Reduce batch size in config
- Enable gradient checkpointing
- Use mixed precision (bf16)

### MLflow Connection Error
- Ensure MLflow server is running
- Check MLFLOW_TRACKING_URI is correct
- Verify network connectivity
