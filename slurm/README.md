# Slurm Scripts for HPC Training

This directory contains Slurm batch scripts for running experiments on the HPC cluster.

## Available Scripts

### `mlflow_server.sh` - MLflow Tracking Server

Starts an MLflow server on a CPU node for experiment tracking.

```bash
# Submit the server job
sbatch slurm/mlflow_server.sh

# Find the allocated node
squeue -u $USER | grep mlflow
# Example output: 12345  mlflow_ser   cpu2  ...

# Set tracking URI in your environment
export MLFLOW_TRACKING_URI=http://cpu2:5000
```

The server runs for up to 7 days. Access the UI via SSH tunnel:
```bash
ssh -L 5000:cpu2:5000 user@login_node
# Then open http://localhost:5000 in your browser
```

### `train_8gpu.sh` - Multi-GPU Training (8×A100)

Trains the LatentGPT model on 8 A100 GPUs.

```bash
# Default config
sbatch slurm/train_8gpu.sh

# Custom config
sbatch slurm/train_8gpu.sh configs/transformer_500m.yaml
```

## Workflow

1. **Start MLflow server**
   ```bash
   sbatch slurm/mlflow_server.sh
   ```

2. **Wait for server to start and note the node**
   ```bash
   squeue -u $USER
   ```

3. **Submit training job with tracking URI**
   ```bash
   export MLFLOW_TRACKING_URI=http://<mlflow_node>:5000
   sbatch slurm/train_8gpu.sh
   ```

4. **Monitor training**
   ```bash
   # View logs
   tail -f slurm/logs/train_<job_id>.out
   
   # Check GPU usage
   ssh <training_node> nvidia-smi
   ```

## MLflow Experiments

| Experiment | Description |
|------------|-------------|
| `vqvae-training` | Custom VQ-VAE training |
| `latent-gpt-pretrained-vqvae` | Transformer with pretrained VQ-VAE |
| `latent-gpt-custom-vqvae` | Transformer with custom VQ-VAE |

## Troubleshooting

### Job pending for too long
Check node availability:
```bash
sinfo -p gpu
```

### Out of memory
Reduce batch size in config or use gradient accumulation:
```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 2
```

### MLflow connection refused
Ensure the server is running and you're using the correct node name:
```bash
squeue -u $USER | grep mlflow
curl http://<node>:5000/health
```
