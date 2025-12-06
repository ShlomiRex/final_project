# Autoregressive Latent Transformer for Text-to-Image Generation

A GPT-style transformer that generates images by autoregressively predicting VQ-VAE latent tokens, conditioned on CLIP text embeddings.

## Overview

This project implements a text-to-image model inspired by "Generative Pretraining from Pixels" (iGPT) but operating in VQ-VAE latent space with text conditioning. The model:

1. Encodes images to discrete tokens using VQ-VAE
2. Generates tokens autoregressively with a GPT-2-style transformer
3. Conditions on text via cross-attention to CLIP embeddings
4. Supports both conditional and unconditional generation

## Architecture

```
Text Prompt → CLIP Encoder → Cross-Attention
                                    ↓
Image → VQ-VAE Encoder → Tokens → Transformer → Tokens → VQ-VAE Decoder → Image
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Dataset loaders
│   ├── training/          # Training scripts
│   └── utils/             # Utilities
├── scripts/               # Entry point scripts
├── slurm/                 # Slurm job scripts
├── notebooks/             # Jupyter notebooks
├── configs/               # YAML configurations
└── requirements.txt       # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server (on HPC)

```bash
sbatch slurm/mlflow_server.sh
```

### 3. Train Model

```bash
# Single GPU (development)
python scripts/train_latent_gpt.py --config configs/base.yaml

# Multi-GPU via Slurm
sbatch slurm/train_8gpu.sh
```

### 4. Generate Images

```bash
python scripts/inference.py --prompt "A dog playing in the park" --checkpoint path/to/model
```

## Training Phases

| Phase | Description | MLflow Experiment |
|-------|-------------|-------------------|
| 1 | Pretrained VQ-VAE + Transformer | `latent-gpt-pretrained-vqvae` |
| 2 | Custom VQ-VAE Training | `vqvae-training` |
| 3 | Full Transformer Training | `latent-gpt-custom-vqvae` |

## MLflow Tracking

Access experiments at `http://<mlflow_node>:5000` after starting the MLflow server.

## Hardware Requirements

- **Training**: 8×A100 80GB (recommended)
- **Inference**: Single GPU with 16GB+ VRAM

## License

MIT
