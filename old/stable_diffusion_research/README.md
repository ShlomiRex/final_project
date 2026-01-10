# Stable Diffusion Research

A research-focused implementation of Latent Diffusion Models for text-to-image generation, designed for multi-GPU training on HPC clusters.

## Features

- **Custom U-Net Architecture**: Modular implementation with cross-attention for text conditioning
- **Multi-Resolution Training**: Support for 256, 384, and 512 resolution images
- **Distributed Training**: HuggingFace Accelerate for multi-GPU training
- **Experiment Tracking**: MLflow integration for metrics, images, and artifacts
- **Evaluation Metrics**: FID score and CLIP score calculation
- **Checkpoint Management**: Automatic saving, resumption, and progressive sampling
- **Classifier-Free Guidance**: CFG training and inference

## Project Structure

```
stable_diffusion_research/
├── configs/                    # YAML configuration files
│   ├── base.yaml              # Base configuration
│   ├── model/                 # Model size variants
│   ├── training/              # Training configurations
│   └── dataset/               # Dataset configurations
├── src/                       # Source code
│   ├── models/                # Neural network models
│   │   ├── unet.py           # U-Net with cross-attention
│   │   ├── vae.py            # VAE wrapper
│   │   ├── text_encoder.py   # CLIP text encoder
│   │   ├── attention.py      # Attention modules
│   │   ├── resnet.py         # ResNet blocks
│   │   └── embeddings.py     # Time/label embeddings
│   ├── diffusion/            # Diffusion process
│   │   ├── noise_scheduler.py # DDPM/DDIM schedulers
│   │   ├── sampler.py        # Sampling with CFG
│   │   └── loss.py           # Diffusion loss functions
│   ├── data/                 # Data loading
│   │   ├── dataset.py        # Dataset classes
│   │   └── transforms.py     # Image transforms
│   ├── training/             # Training utilities
│   │   ├── trainer.py        # Main trainer class
│   │   ├── checkpoint.py     # Checkpoint management
│   │   ├── ema.py            # EMA model
│   │   └── lr_scheduler.py   # Learning rate schedulers
│   ├── evaluation/           # Evaluation metrics
│   │   ├── evaluator.py      # Main evaluator
│   │   ├── fid.py            # FID calculation
│   │   ├── clip_score.py     # CLIP score
│   │   └── sample_generator.py # Sample generation
│   └── utils/                # Utilities
│       ├── config.py         # Configuration loading
│       ├── logging.py        # MLflow logging
│       ├── visualization.py  # Image grids
│       └── distributed.py    # Distributed training
├── scripts/                  # Training/inference scripts
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   ├── generate.py          # Image generation
│   └── download_pretrained.py # Download models
├── slurm/                    # HPC job scripts
│   ├── train_multigpu.sh    # 8-GPU training
│   ├── train_2gpu.sh        # 2-GPU training
│   ├── evaluate.sh          # Evaluation job
│   ├── generate.sh          # Generation job
│   ├── mlflow_server.sh     # MLflow server
│   └── README.md            # Slurm usage guide
└── requirements.txt         # Dependencies
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd stable_diffusion_research

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Download Pretrained Models

```bash
python scripts/download_pretrained.py --output_dir pretrained_models
```

### 2. Start MLflow Server (Optional)

```bash
mlflow server --host 0.0.0.0 --port 5000
# Open http://localhost:5000
```

### 3. Training

```bash
# Single GPU
python scripts/train.py --config configs/base.yaml

# Multi-GPU with Accelerate
accelerate launch --num_processes=8 --multi_gpu --mixed_precision=bf16 \
    scripts/train.py --config configs/base.yaml

# With config overrides
python scripts/train.py --config configs/base.yaml \
    training.learning_rate=1e-4 \
    training.batch_size=32
```

### 4. Resume Training

```bash
python scripts/train.py --config configs/base.yaml --resume latest
```

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/checkpoint_100000.pt \
    --config configs/base.yaml \
    --output_dir evaluation_results
```

### 6. Image Generation

```bash
# Single prompt
python scripts/generate.py \
    --checkpoint outputs/checkpoints/checkpoint_100000.pt \
    --config configs/base.yaml \
    --prompt "A photo of a cat sitting on a windowsill"

# Interactive mode
python scripts/generate.py \
    --checkpoint outputs/checkpoints/checkpoint_100000.pt \
    --config configs/base.yaml \
    --interactive
```

## HPC Training (Slurm)

### Submit Training Job

```bash
# 8-GPU training
sbatch slurm/train_multigpu.sh configs/base.yaml

# 2-GPU development
sbatch slurm/train_2gpu.sh configs/base.yaml

# Resume from checkpoint
sbatch slurm/train_multigpu.sh configs/base.yaml --resume latest
```

### Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f slurm/logs/train_<job_id>.out

# Access MLflow (after starting server)
ssh -N -L localhost:5000:<node>.hpc.pub.lan:5000 doshlom4@login9
# Open http://localhost:5000
```

See [slurm/README.md](slurm/README.md) for detailed HPC usage.

## Configuration

Training is controlled by YAML configuration files that support inheritance:

```yaml
# configs/training/train_512.yaml
_base_: "../base.yaml"  # Inherit from base config

data:
  resolution: 512

training:
  batch_size: 4  # Smaller batch for higher resolution
```

### Key Configuration Options

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `block_out_channels` | U-Net channel sizes |
| `diffusion` | `num_train_timesteps` | Number of diffusion steps |
| `training` | `learning_rate` | Learning rate |
| `training` | `batch_size` | Per-GPU batch size |
| `training` | `gradient_accumulation_steps` | Effective batch multiplier |
| `training` | `mixed_precision` | "bf16", "fp16", or "no" |
| `checkpoint` | `save_every_n_steps` | Checkpoint frequency |
| `evaluation` | `sample_every_n_steps` | Sample generation frequency |

## Evaluation Metrics

### FID Score (Fréchet Inception Distance)
- Measures similarity between generated and real image distributions
- Lower is better (0 = identical distributions)

### CLIP Score
- Measures alignment between generated images and text prompts
- Higher is better (max = 100)

## Model Architecture

### U-Net
- **Input**: 4-channel latent (from VAE)
- **Conditioning**: CLIP text embeddings via cross-attention
- **Blocks**: ResNet + SpatialTransformer (self-attention + cross-attention)
- **Sizes**: Small (~100M), Medium (~400M), Large (~800M)

### VAE
- Pretrained: `stabilityai/sd-vae-ft-mse`
- Compression: 8× spatial downsampling
- Latent: 4 channels

### Text Encoder
- Pretrained: `openai/clip-vit-large-patch14`
- Output: 768-dim embeddings, 77 tokens

## Training Tips

1. **Start Small**: Use 256 resolution and small model first
2. **Monitor Loss**: Should decrease steadily, samples improve after ~10k steps
3. **Check Samples**: Visual quality is often better indicator than loss
4. **Use EMA**: Always use EMA weights for inference (better quality)
5. **CFG**: Train with 10% unconditional dropout, use guidance_scale=7.5 for inference

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{stable_diffusion_research,
  title={Stable Diffusion Research Implementation},
  year={2024},
  author={Your Name},
  url={https://github.com/your-repo}
}
```

## License

This project is for research purposes. See LICENSE for details.

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Original implementation
- [Diffusers](https://github.com/huggingface/diffusers) - Pretrained models
- [Accelerate](https://github.com/huggingface/accelerate) - Distributed training
