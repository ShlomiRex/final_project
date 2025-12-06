#!/usr/bin/env python
"""
Main training script for LatentGPT.

Usage:
    # Single GPU
    python scripts/train_latent_gpt.py --config configs/base.yaml
    
    # Multi-GPU with Accelerate
    accelerate launch --num_processes=8 --multi_gpu --mixed_precision=bf16 \
        scripts/train_latent_gpt.py --config configs/transformer_500m.yaml
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logging import setup_mlflow, log_config, log_metrics, end_run
from src.models.latent_gpt import LatentGPT
from src.models.clip_encoder import CLIPEncoder
from src.data.flickr30k import create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LatentGPT")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def get_lr_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(current_step):
        if current_step < config.training.warmup_steps:
            return float(current_step) / float(max(1, config.training.warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - config.training.warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: LatentGPT,
    clip_encoder: CLIPEncoder,
    batch: dict,
    accelerator: Accelerator,
) -> torch.Tensor:
    """
    Single training step.
    
    For now, we use a simplified training loop that:
    1. Encodes images through a placeholder (TODO: add VQ-VAE)
    2. Encodes text through CLIP
    3. Predicts next token autoregressively
    
    Returns:
        Loss tensor
    """
    images = batch["images"]
    captions = batch["captions"]
    
    # TODO: Encode images to tokens via VQ-VAE
    # For now, create dummy tokens for structure validation
    B = images.size(0)
    seq_len = model.max_seq_len
    
    # Placeholder: random tokens (replace with VQ-VAE encoding)
    tokens = torch.randint(0, model.vocab_size, (B, seq_len), device=images.device)
    
    # Encode text
    text_embeddings = clip_encoder(captions)
    
    # Shift for autoregressive prediction
    # Input: [BOS, t1, t2, ..., t_{n-1}]
    # Target: [t1, t2, ..., t_n]
    bos_tokens = torch.full((B, 1), model.bos_token_id, device=tokens.device)
    input_tokens = torch.cat([bos_tokens, tokens[:, :-1]], dim=1)
    target_tokens = tokens
    
    # Forward pass
    logits = model(input_tokens, cross_context=text_embeddings)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),
        target_tokens.view(-1),
    )
    
    return loss


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Print info on main process
    if accelerator.is_main_process:
        print("=" * 60)
        print("LatentGPT Training")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"Device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Mixed precision: {config.training.mixed_precision}")
        print("=" * 60)
    
    # Setup MLflow (main process only)
    if accelerator.is_main_process:
        run_id = setup_mlflow(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
            run_name=config.mlflow.run_name,
            tags=config.mlflow.tags,
        )
        log_config(config)
        print(f"MLflow run ID: {run_id}")
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    if accelerator.is_main_process:
        print("\nInitializing models...")
    
    # CLIP encoder (frozen)
    clip_encoder = CLIPEncoder(
        model_name=config.text_encoder.model_name,
        max_length=config.text_encoder.max_length,
    )
    
    # LatentGPT transformer
    model = LatentGPT(
        vocab_size=config.vqvae.codebook_size,
        hidden_size=config.transformer.hidden_size,
        num_layers=config.transformer.num_layers,
        num_heads=config.transformer.num_heads,
        max_seq_len=config.transformer.max_seq_len,
        ffn_dim=config.transformer.ffn_dim,
        dropout=config.transformer.dropout,
        cross_attention_dim=config.text_encoder.hidden_size,
        use_cross_attention=True,
    )
    
    if accelerator.is_main_process:
        num_params = model.get_num_params()
        print(f"Transformer parameters: {num_params / 1e6:.1f}M")
    
    # Create dataloader
    dataloader = create_dataloader(
        split="train",
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        cfg_dropout=config.training.cfg_dropout,
        cache_dir=config.data.cache_dir,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    num_training_steps = config.training.max_steps
    scheduler = get_lr_scheduler(optimizer, config, num_training_steps)
    
    # Prepare with accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Move CLIP to device (not wrapped by accelerator since it's frozen)
    clip_encoder = clip_encoder.to(accelerator.device)
    
    # Training loop
    global_step = 0
    model.train()
    
    if accelerator.is_main_process:
        print("\nStarting training...")
        pbar = tqdm(total=config.training.max_steps, desc="Training")
    
    while global_step < config.training.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = train_step(model, clip_encoder, batch, accelerator)
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                
                # Logging
                if global_step % config.training.log_interval == 0:
                    if accelerator.is_main_process:
                        log_metrics({
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                        }, step=global_step)
                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                
                # Checkpointing
                if global_step % config.training.checkpoint_interval == 0:
                    if accelerator.is_main_process:
                        checkpoint_path = output_dir / f"checkpoint_{global_step}.pt"
                        accelerator.save_state(str(checkpoint_path))
                        print(f"\nSaved checkpoint: {checkpoint_path}")
                
                if accelerator.is_main_process:
                    pbar.update(1)
                
                if global_step >= config.training.max_steps:
                    break
    
    # Save final model
    if accelerator.is_main_process:
        pbar.close()
        final_path = output_dir / "final_model.pt"
        accelerator.save_state(str(final_path))
        print(f"\nTraining complete! Final model saved to: {final_path}")
        end_run()


if __name__ == "__main__":
    main()
