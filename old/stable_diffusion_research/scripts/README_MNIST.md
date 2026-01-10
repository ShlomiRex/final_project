# MNIST Text-Conditioned Diffusion Model

This directory contains a Python script for training a text-conditioned diffusion model on the MNIST dataset.

## Overview

The `train_mnist.py` script trains a simplified diffusion model with:
- **Architecture**: Custom UNet2DConditionModel with cross-attention
- **Text Conditioning**: CLIP text embeddings (frozen)
- **Dataset**: MNIST handwritten digits
- **Training**: Full pipeline with noise scheduling, loss computation, and sampling
- **Features**: Multi-GPU support via Accelerate, classifier-free guidance

## Quick Start

### Single GPU Training

```bash
# Basic training with default settings
python scripts/train_mnist.py

# With custom parameters
python scripts/train_mnist.py \
    --num_epochs 10 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --guidance_scale 10.0
```

### Multi-GPU Training

```bash
# Training on 2 GPUs
accelerate launch --num_processes=2 scripts/train_mnist.py

# With mixed precision (faster)
accelerate launch --num_processes=2 --mixed_precision=fp16 scripts/train_mnist.py
```

## Configuration

Training can be configured via command-line arguments. Key parameters:

- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 512)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--tokenizer_max_length`: Max text token length (default: 8)
- `--num_inference_steps`: Denoising steps for sampling (default: 50)
- `--guidance_scale`: Classifier-free guidance strength (default: 8.0)
- `--output_dir`: Output directory for checkpoints and samples (default: outputs/mnist)
- `--dataset_path`: Path to MNIST dataset cache (default: ./dataset_cache)
- `--seed`: Random seed for reproducibility (default: 422)

## Output

The script generates:

```
outputs/mnist/
├── checkpoints/
│   ├── checkpoint_epoch_000.pt
│   ├── checkpoint_epoch_001.pt
│   └── ...
└── samples/
    ├── samples_epoch_000.png
    ├── samples_epoch_001.png
    └── ...
```

Each checkpoint contains:
- UNet state dictionary
- Optimizer state
- Epoch number

Sample images show generated digits for various prompts and guidance scales.

## Model Architecture

### UNet Configuration for MNIST
- **Input channels**: 1 (grayscale)
- **Output channels**: 1 (grayscale prediction)
- **Model channels**: 32 (minimal for fast training)
- **Cross-attention dimension**: 512 (CLIP ViT-B/32 embedding size)
- **Down/Up blocks**: 4 levels with attention at specific resolutions

### Text Encoder
- **Pretrained Model**: OpenAI CLIP ViT-B/32
- **Embedding Dimension**: 512
- **Token Length**: 8 (sufficient for digit captions)
- **Status**: Frozen (not trained)

### Noise Scheduler
- **Type**: DDPM (Denoising Diffusion Probabilistic Models)
- **Beta schedule**: Squared cosine
- **Training timesteps**: 1000
- **Inference timesteps**: 50 (configurable)

## Training Details

### Loss Function
Mean Squared Error between predicted and actual Gaussian noise:
```
loss = MSE(noise_pred, noise)
```

### Classifier-Free Guidance
During inference, text conditioning is optionally dropped and guidance is applied:
```
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

### Captions
Text prompts are automatically generated from labels:
- "A handwritten digit 0"
- "A handwritten digit 1"
- ... (for digits 0-9)

## Dependencies

Core dependencies (all available in the environment):
- `torch` >= 2.0
- `transformers` (for CLIP)
- `diffusers` (for noise scheduler and model utilities)
- `accelerate` (for distributed training)
- `torchvision` (for MNIST dataset)
- `matplotlib` (for visualization)

## Implementation Notes

### From Notebook Conversion
This script is converted from `train3_working_with_prompts_mnist.ipynb` with:
- **Structure**: Follows `stable_diffusion_research/` conventions
- **Modularity**: Functions for dataset prep, model setup, training loop, inference
- **Acceleration**: Native Accelerate support for multi-GPU training
- **Reproducibility**: Seed management and deterministic operations
- **Scalability**: Designed to extend for larger datasets and models

### Key Differences from Notebook
1. **Command-line interface**: Parameters via CLI arguments
2. **Multi-GPU support**: Native Accelerate integration
3. **Checkpointing**: Structured checkpoint saving
4. **Progress tracking**: TQDM progress bars and logging
5. **Configuration**: YAML-based config system (extensible)

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python scripts/train_mnist.py --batch_size 256
```

### Slow Training
Use mixed precision:
```bash
accelerate launch --mixed_precision=fp16 scripts/train_mnist.py
```

### CUDA Not Available
Train on CPU (slower):
```bash
python scripts/train_mnist.py
# Device will auto-detect and default to CPU
```

## Future Extensions

Possible enhancements:
1. Add VAE encoding/decoding for full latent diffusion pipeline
2. Implement conditional sampling with variable guidance scales
3. Add FID score evaluation
4. Support for other datasets (CIFAR-10, custom)
5. Alternative text encoders (larger CLIP models)
6. Fine-tuning of text encoder
7. Beam search for prompt optimization

## References

- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- **CLIP**: [Learning Transferable Models for Unsupervised Learning](https://arxiv.org/abs/2103.14030)
- **Classifier-Free Guidance**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
