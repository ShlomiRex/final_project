#!/usr/bin/env python3
"""
MNIST Text-Conditioned Diffusion Model Training Script.

This script trains a simplified diffusion model on MNIST digits with text conditioning.
It's designed for testing and validation of the diffusion pipeline.

Usage:
    # Single GPU training
    python scripts/train_mnist.py --config configs/base.yaml
    
    # Multi-GPU training with Accelerate
    accelerate launch --num_processes=2 scripts/train_mnist.py --config configs/base.yaml
    
    # With custom parameters
    python scripts/train_mnist.py \
        --config configs/base.yaml \
        --num_epochs 10 \
        --batch_size 256 \
        --learning_rate 1e-4
"""

from __future__ import annotations

import argparse
import math
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.state import AcceleratorState

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.utils.config import load_config, parse_cli_overrides


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MNIST text-conditioned diffusion model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mnist",
        help="Output directory for checkpoints and samples",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=8,
        help="Maximum tokenizer length",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=8.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset_cache",
        help="Path to MNIST dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=422,
        help="Random seed",
    )
    parser.add_argument(
        "--num_samples_per_epoch",
        type=int,
        default=6,
        help="Number of sample images to generate per epoch",
    )
    
    return parser.parse_args()


class CustomUNet2DConditionModel(UNet2DConditionModel):
    """Custom UNet2D model for MNIST with reduced parameters."""
    
    def __init__(self, **kwargs):
        """Initialize custom UNet with MNIST-specific settings."""
        super().__init__(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 64, 32),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,  # CLIP ViT-B/32 embedding dimension
            **kwargs
        )


def setup_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_models(device: torch.device) -> Tuple[nn.Module, CLIPTextModel, CLIPTokenizer, DDPMScheduler]:
    """Load and setup models.
    
    Args:
        device: Target device (cuda/cpu)
    
    Returns:
        Tuple of (unet, text_encoder, tokenizer, scheduler)
    """
    # Initialize UNet
    unet = CustomUNet2DConditionModel().to(device)
    
    # Load pretrained CLIP models (frozen)
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Freeze text encoder
    text_encoder.requires_grad_(False)
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"Number of trainable parameters in UNet: {num_params:,}")
    
    return unet, text_encoder, tokenizer, noise_scheduler


def prepare_dataset(
    batch_size: int,
    dataset_path: str,
) -> DataLoader:
    """Prepare MNIST dataset and dataloader.
    
    Args:
        batch_size: Batch size for training
        dataset_path: Path to download/cache dataset
    
    Returns:
        DataLoader for MNIST training set
    """
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(
        root=dataset_path,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    print(f"Dataset loaded: {len(mnist_dataset)} samples")
    print(f"Batch shape: {next(iter(dataloader))[0].shape}")
    
    return dataloader


@torch.no_grad()
def generate_samples(
    unet: nn.Module,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    prompts: list[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 8.0,
    seed: int = 422,
) -> torch.Tensor:
    """Generate image samples using the trained model.
    
    Args:
        unet: Trained UNet model
        text_encoder: Text encoder (CLIP)
        tokenizer: Tokenizer for text
        noise_scheduler: Noise scheduler
        device: Device to run on
        prompts: List of text prompts
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed
    
    Returns:
        Generated images tensor (B, 1, 28, 28)
    """
    torch.manual_seed(seed)
    unet.eval()
    text_encoder.eval()
    
    # Tokenize prompts
    tokenizer_max_length = 8
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    # Unconditional embeddings for classifier-free guidance
    uncond_inputs = tokenizer(
        [""] * len(prompts),
        padding="max_length",
        max_length=tokenizer_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
    
    # Concatenate for batch processing
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Initialize latents (noise)
    num_prompts = len(prompts)
    latents = torch.randn(
        (num_prompts, 1, 28, 28),
        device=device,
        dtype=text_embeddings.dtype
    )
    
    # Setup scheduler and timesteps
    scheduler = DDPMScheduler(
        beta_schedule="squaredcos_cap_v2",
        num_train_timesteps=1000
    )
    scheduler.set_timesteps(num_inference_steps)
    
    # Denoising loop
    for t in tqdm(scheduler.timesteps, desc="Sampling", disable=True):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Split predictions for guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        # Compute previous sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # Post-process
    images = (latents / 2 + 0.5).clamp(0, 1)
    
    return images


def save_samples(
    images: torch.Tensor,
    output_dir: Path,
    epoch: int,
    prompts: list[str],
):
    """Save generated samples as a grid image.
    
    Args:
        images: Generated images tensor
        output_dir: Directory to save samples
        epoch: Epoch number
        prompts: Prompts used for generation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create grid
    fig, axes = plt.subplots(
        nrows=len(prompts),
        ncols=1,
        figsize=(3, 3 * len(prompts))
    )
    
    if len(prompts) == 1:
        axes = [axes]
    
    for idx, (ax, prompt) in enumerate(zip(axes, prompts)):
        img = images[idx].squeeze(0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        
        ax.imshow(img, cmap="gray")
        ax.set_title(prompt, fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    output_file = output_dir / f"samples_epoch_{epoch:03d}.png"
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"Saved samples to {output_file}")


def train_epoch(
    unet: nn.Module,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accelerator: Accelerator,
    epoch: int,
    tokenizer_max_length: int = 8,
) -> float:
    """Train for one epoch.
    
    Args:
        unet: UNet model to train
        text_encoder: Frozen text encoder
        tokenizer: Tokenizer
        noise_scheduler: Noise scheduler
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        accelerator: Accelerate accelerator
        epoch: Current epoch number
        tokenizer_max_length: Max token length
    
    Returns:
        Average loss for epoch
    """
    unet.train()
    text_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}",
        disable=not accelerator.is_main_process
    )
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        
        # Create text captions from labels
        captions = [f"A handwritten digit {int(label)}" for label in labels]
        
        # Encode text
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        
        # Add noise to images
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            1000,
            (images.shape[0],),
            device=device,
            dtype=torch.long
        )
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        # Forward pass: predict noise
        noise_pred = unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 25 == 0 and accelerator.is_main_process:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    unet: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
):
    """Save training checkpoint.
    
    Args:
        unet: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        output_dir: Directory to save to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "unet": unet.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup
    setup_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Accelerator (no manual state reset in scripts)
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        print(f"Output directory: {output_dir}")
    
    # Setup models
    if accelerator.is_main_process:
        print("Loading models...")
    unet, text_encoder, tokenizer, noise_scheduler = setup_models(device)
    
    # Prepare dataset
    if accelerator.is_main_process:
        print("Preparing dataset...")
    train_dataloader = prepare_dataset(args.batch_size, args.dataset_path)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    
    # Prepare with Accelerator
    unet, text_encoder, train_dataloader, optimizer = accelerator.prepare(
        unet, text_encoder, train_dataloader, optimizer
    )
    
    # Training loop
    if accelerator.is_main_process:
        print("\nStarting training...")
    
    sample_prompts = [
        "A handwritten digit 0",
        "A handwritten digit 3",
        "A handwritten digit 5",
        "A handwritten digit 7",
        "A handwritten digit 9",
        "A handwritten digit 2",
    ]
    
    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(
            unet,
            text_encoder,
            tokenizer,
            noise_scheduler,
            train_dataloader,
            optimizer,
            device,
            accelerator,
            epoch,
            args.tokenizer_max_length,
        )
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{args.num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Generate and save samples
        if accelerator.is_main_process and (epoch + 1) % max(1, args.num_epochs // args.num_samples_per_epoch) == 0:
            print("Generating samples...")
            samples = generate_samples(
                accelerator.unwrap_model(unet),
                text_encoder,
                tokenizer,
                noise_scheduler,
                device,
                sample_prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + epoch,
            )
            save_samples(samples, output_dir / "samples", epoch, sample_prompts)
        
        # Save checkpoint
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator.unwrap_model(unet),
                optimizer,
                epoch,
                output_dir / "checkpoints"
            )
    
    if accelerator.is_main_process:
        print("\nTraining complete!")
        print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
        print(f"Samples saved to: {output_dir / 'samples'}")


if __name__ == "__main__":
    main()
