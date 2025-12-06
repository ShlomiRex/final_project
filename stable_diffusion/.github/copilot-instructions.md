# AI Coding Agent Instructions

## Project Overview
Text-to-image diffusion model training on HPC cluster with multi-GPU support using Accelerate, Diffusers, and CLIP.

## Architecture & Components

### Core Models (Frozen - Pretrained)
- **VAE**: `stabilityai/sdxl-vae` or `runwayml/stable-diffusion-v1-5` - encodes images to latents (scaling factor 0.18215)
- **Text Encoder**: `openai/clip-vit-base-patch32` - CLIP ViT-B/32 (hidden_size=512)
- **Tokenizer**: CLIP tokenizer (max_length=16-77 depending on experiment)

### Trainable Component
- **UNet2DConditionModel**: Cross-attention conditioned on CLIP text embeddings
  - Input: Latent space (4 channels for VAE, or 1 channel for raw grayscale)
  - Architecture: CrossAttnDownBlock2D → DownBlock2D → UpBlock2D → CrossAttnUpBlock2D
  - Conditioning: `cross_attention_dim=512` (matches CLIP hidden size)

### Training Pipeline
1. **Encode**: `latents = vae.encode(images).latent_dist.sample() * 0.18215`
2. **Text**: `text_embeddings = text_encoder(input_ids).last_hidden_state`
3. **Noise**: Add Gaussian noise at random timestep `t ~ Uniform(0, 1000)`
4. **Predict**: `noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings).sample`
5. **Loss**: MSE between predicted and actual noise
6. **CFG**: Drop text conditioning with probability `cfg_dropout_p` (typically 0.1)

### Sampling Pipeline (Classifier-Free Guidance)
1. Encode prompt and empty string `""` separately
2. Concatenate embeddings: `[uncond_embeddings, text_embeddings]`
3. For each timestep, predict noise for both conditions
4. Combine: `noise_pred = uncond + guidance_scale * (text - uncond)`
5. Decode latents: `image = vae.decode(latents / 0.18215).sample`

## HPC Cluster Workflow

### Environment Setup
- **Conda**: `/home/doshlom4/work/conda/envs/shlomid_conda_12_11_2025` (CUDA 11.8, Python 3.10)
- **Poetry**: Alternative dependency manager (see `pyproject.toml`)
- **Key Packages**: PyTorch 2.7.1+cu118, Accelerate 1.11.0, Diffusers, Transformers, Torchmetrics

### Slurm Job Submission
- **Scripts**: `slurm/*.sh` (e.g., `train7.sh`, `jupyter_notebook_interactive.sh`)
- **GPU Allocation**: Request specific node types (e.g., `#SBATCH --constraint=gpu32g` for 32GB GPUs)
- **Interactive Jupyter**: `sbatch slurm/jupyter_notebook_interactive.sh`, then SSH tunnel to access

### Multi-GPU Training
**CRITICAL**: Jupyter notebooks run single-process → **1 GPU only** even if 2 allocated
- **Notebook Mode**: Accelerate defaults to single GPU (see diagnostic cells in `train1_flickr8.ipynb`)
- **Script Mode (True Multi-GPU)**: 
  1. Convert notebook: `jupyter nbconvert --to script notebook.ipynb`
  2. Ensure `from __future__ import annotations` is **first line** of code
  3. Launch: `accelerate launch --num_processes=2 script.py`

### SSH Tunneling for Jupyter
See `slurm/JUPYTER_SETUP.md` and `slurm/connect_jupyter.sh` for detailed instructions.

## Development Patterns

### Notebook Structure (Standard Template)
1. **Cell 1**: `from __future__ import annotations` (for script compatibility)
2. **Diagnostic Cells**: GPU/CUDA/Accelerate checks before heavy imports
3. **Imports**: Accelerate, Diffusers, Transformers, Torchmetrics
4. **Configuration**: Hyperparameters, paths, TensorBoard run naming with timestamp
5. **Model Loading**: Pretrained VAE, CLIP, UNet initialization
6. **Dataset**: DataLoader with collate function
7. **Training Loop**: Accelerate-wrapped, epoch-level logging only
8. **Sampling**: Classifier-free guidance with configurable `guidance_scale`
9. **Evaluation**: FID and CLIP Score (ensure metrics are on correct device)

### Critical Code Patterns

#### Device-Aware Metrics
```python
# WRONG - causes CPU/CUDA mismatch
fid.update(images, real=True)

# CORRECT - move to device first
fid.update(images.to(device), real=True)
```

#### Unique TensorBoard Runs
```python
from datetime import datetime
run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"runs/{run_name}")
```

#### AcceleratorState Reset (Notebooks Only)
```python
from accelerate.state import AcceleratorState
try:
    AcceleratorState._reset_state()
except:
    pass
accelerator = Accelerator()
```

#### Checkpoint Format (Remove Batch Losses)
```python
# Save only epoch-level metrics
checkpoint = {
    'unet': unet.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
    'epoch_losses': epoch_losses  # NOT batch_losses
}
```

### Common Pitfalls
1. **from __future__ placement**: Must be **first code line** in converted scripts (before all imports)
2. **Deprecated Accelerate attributes**: Avoid `use_fp16`, `backward_passes_per_step`
3. **Metric device mismatch**: Always `.to(device)` before passing to torchmetrics
4. **TensorBoard overwrites**: Use timestamped run names, not `"."` or hardcoded strings
5. **Multi-GPU expectations**: Notebooks are single-GPU; use scripts + `accelerate launch` for true distributed training

## File Organization
- **`notebooks/`**: Experimental Jupyter notebooks (single-GPU)
- **`scripts/`**: Production Python scripts (multi-GPU capable)
- **`slurm/`**: HPC job submission and Jupyter setup scripts
- **`dataset_cache/`**: HuggingFace datasets cache (set `HF_HOME` if relocating)
- **`train7_output/`**: Model checkpoints and generated samples

## Testing & Validation
- **Diagnostics**: Run GPU/Accelerate diagnostic cells before training
- **Sampling**: Generate images every N epochs to monitor progress
- **Metrics**: Track FID (image quality) and CLIP Score (text-image alignment)
- **Checkpoints**: Save every N epochs with epoch number in filename

## References
- **README.md**: Comprehensive HPC cluster Jupyter setup guide
- **pyproject.toml**: Poetry dependency specifications
- **environment.yml**: Conda environment with CUDA 11.8 specifics
- **slurm/NODE_SELECTION_GUIDE.md**: GPU node selection strategies
