# MNIST Notebook to Script Conversion Summary

## Overview
Successfully converted the MNIST text-conditioned diffusion training notebook (`train3_working_with_prompts_mnist.ipynb`) into a production-ready Python script following the `stable_diffusion_research` project structure.

## Files Created

### 1. **scripts/train_mnist.py** (Main Training Script)
**Location**: `/home/doshlom4/work/final_project/stable_diffusion_research/scripts/train_mnist.py`

**Features**:
- ✅ Full training pipeline with noise scheduling and loss computation
- ✅ Multi-GPU support via Accelerate framework
- ✅ Classifier-free guidance for conditional image generation
- ✅ Command-line interface for easy parameter tuning
- ✅ Checkpoint saving and sample generation
- ✅ Proper error handling and logging
- ✅ Seed management for reproducibility

**Key Functions**:
- `setup_models()` - Load UNet, CLIP encoder, and scheduler
- `prepare_dataset()` - Setup MNIST DataLoader
- `train_epoch()` - Single epoch training loop
- `generate_samples()` - Inference with classifier-free guidance
- `save_samples()` - Visualize and save generated images
- `save_checkpoint()` - Persist model state

**Usage**:
```bash
# Single GPU
python scripts/train_mnist.py --num_epochs 10 --batch_size 512

# Multi-GPU with Accelerate
accelerate launch --num_processes=2 scripts/train_mnist.py
```

### 2. **configs/mnist.yaml** (Configuration File)
**Location**: `/home/doshlom4/work/final_project/stable_diffusion_research/configs/mnist.yaml`

**Contains**:
- Model architecture specifications (UNet configuration for MNIST)
- Diffusion parameters (timesteps, beta schedule, guidance scale)
- Data configuration (batch size, image resolution)
- Training hyperparameters (learning rate, epochs, logging intervals)
- Output directory settings

### 3. **slurm/train_mnist.sh** (HPC Job Submission Script)
**Location**: `/home/doshlom4/work/final_project/stable_diffusion_research/slurm/train_mnist.sh`

**Features**:
- ✅ SLURM job configuration for 2-GPU training
- ✅ Environment activation
- ✅ GPU availability checking
- ✅ Multi-GPU Accelerate launch
- ✅ Logging to timestamped output files

**Usage**:
```bash
sbatch slurm/train_mnist.sh
```

### 4. **scripts/README_MNIST.md** (Comprehensive Documentation)
**Location**: `/home/doshlom4/work/final_project/stable_diffusion_research/scripts/README_MNIST.md`

**Includes**:
- Quick start guide for single and multi-GPU training
- Detailed parameter descriptions
- Architecture and model specifications
- Training methodology and loss functions
- Troubleshooting guide
- Future extension possibilities

## Architecture Alignment with Project Structure

The converted script follows the `stable_diffusion_research` conventions:

```
stable_diffusion_research/
├── scripts/
│   ├── train_mnist.py          ✅ NEW - Main training script
│   └── README_MNIST.md         ✅ NEW - Documentation
├── configs/
│   └── mnist.yaml              ✅ NEW - MNIST-specific config
├── slurm/
│   └── train_mnist.sh          ✅ NEW - HPC submission script
└── src/                        ✅ USED - Utility imports (config, etc.)
```

## Key Improvements Over Notebook

### 1. **Modular Design**
- Notebook: Monolithic cells with scattered code
- Script: Organized functions with clear responsibilities

### 2. **Command-Line Interface**
- Notebook: Manual cell parameter editing
- Script: Flexible CLI arguments for all hyperparameters

### 3. **Multi-GPU Support**
- Notebook: Single GPU only (even if multiple allocated)
- Script: Native Accelerate support for true distributed training

### 4. **Production Readiness**
- Notebook: Development/experimental code
- Script: Error handling, logging, checkpoint management

### 5. **Reproducibility**
- Notebook: Random seeds scattered
- Script: Centralized seed management (`setup_seed()`)

### 6. **Extensibility**
- Notebook: Hard to scale or adapt
- Script: Easy to extend with new features, datasets, models

## Model Specifications

### UNet Architecture (MNIST-Optimized)
```
Input: (B, 1, 28, 28) grayscale images
↓
DownBlocks with CrossAttention
  - 32 → 64 → 64 → 32 channels
  - Cross-attention with CLIP embeddings (512-dim)
↓
UpBlocks with CrossAttention
  - Mirror of down blocks
↓
Output: (B, 1, 28, 28) noise prediction
```

### Text Conditioning
- **Encoder**: OpenAI CLIP ViT-B/32 (frozen)
- **Embedding dimension**: 512
- **Token length**: 8 tokens (sufficient for digit captions)
- **Captions**: Auto-generated from labels ("A handwritten digit 3", etc.)

### Training Configuration
- **Optimizer**: AdamW (lr=1e-3)
- **Loss**: MSE between predicted and actual noise
- **Scheduler**: DDPM with squared-cosine beta schedule
- **Guidance**: Classifier-free guidance (scale=8.0)

## Running the Script

### Local Single GPU
```bash
cd /home/doshlom4/work/final_project/stable_diffusion_research
python scripts/train_mnist.py
```

### Local Multi-GPU
```bash
accelerate launch --num_processes=2 scripts/train_mnist.py
```

### HPC Cluster (2 GPUs, 4 hours)
```bash
sbatch slurm/train_mnist.sh
```

### With Custom Parameters
```bash
python scripts/train_mnist.py \
    --num_epochs 20 \
    --batch_size 256 \
    --learning_rate 5e-4 \
    --guidance_scale 10.0 \
    --output_dir outputs/mnist_custom
```

## Output Structure

```
outputs/mnist/
├── checkpoints/
│   ├── checkpoint_epoch_000.pt    # State dict + optimizer
│   ├── checkpoint_epoch_001.pt
│   └── ...
└── samples/
    ├── samples_epoch_000.png       # Grid of generated images
    ├── samples_epoch_001.png
    └── ...
```

## Performance Characteristics

Based on notebook observations:
- **Training time**: ~2 hours for 5 epochs on 2× V100 GPUs
- **Memory usage**: ~15GB per GPU
- **Batch size**: 512 (MNIST is small)
- **Convergence**: Loss stabilizes after 2-3 epochs

## Dependencies Verified

All dependencies are available in the HPC conda environment:
- ✅ PyTorch 2.7.1+cu118
- ✅ Accelerate 1.11.0
- ✅ Diffusers (transformers pipeline)
- ✅ Transformers (CLIP models)
- ✅ Torchvision (MNIST dataset)
- ✅ Matplotlib (visualization)

## Future Enhancements

Potential improvements for production use:
1. **VAE Integration**: Add full latent diffusion pipeline
2. **Evaluation Metrics**: FID score, CLIP score computation
3. **Distributed Logging**: MLflow/Weights & Biases integration
4. **Larger Datasets**: Support for CIFAR-10, ImageNet, custom datasets
5. **Model Variants**: Support for different UNet configurations
6. **Fine-tuning**: Option to train text encoder alongside UNet
7. **Advanced Sampling**: DDIM, Euler, DPM-solver schedulers

## Testing Checklist

- [x] Script runs without errors
- [x] Imports are correct
- [x] Device detection works (CUDA/CPU)
- [x] Dataset downloads and loads
- [x] Model initialization succeeds
- [x] Training loop executes
- [x] Checkpoint saving works
- [x] Sample generation produces valid images
- [x] Accelerate integration tested
- [x] SLURM script syntax valid

## Contact & Support

For issues or questions:
1. Check `README_MNIST.md` troubleshooting section
2. Review script inline comments for logic details
3. Verify environment with `python scripts/check_environment.py`
4. Check SLURM logs: `tail -f slurm/logs/mnist_diffusion_*.err`

---

**Conversion Date**: January 6, 2026
**Original Notebook**: `notebooks-old/complete_new_model/diffusers/train3_working_with_prompts_mnist.ipynb`
**Project Structure**: `stable_diffusion_research`
