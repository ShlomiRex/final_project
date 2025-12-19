#!/usr/bin/env python3
"""
Main training script for Stable Diffusion.

Usage:
    # Single GPU
    python scripts/train.py --config configs/base.yaml
    
    # Multi-GPU with Accelerate
    accelerate launch --num_processes=8 --multi_gpu --mixed_precision=bf16 \
        scripts/train.py --config configs/base.yaml
    
    # With config overrides
    python scripts/train.py --config configs/base.yaml \
        training.learning_rate=1e-4 \
        training.batch_size=32
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import UNet2DModel  # Changed to unconditional UNet

from src.models.vae import VAEWrapper
from src.diffusion.noise_scheduler import DDPMScheduler, SchedulerConfig
from src.diffusion.loss import DiffusionLoss
from src.training.trainer import StableDiffusionTrainer
from src.training.checkpoint import CheckpointManager
from src.training.ema import EMAModel
from src.training.lr_scheduler import get_lr_scheduler
from src.data.dataset import get_dataloader
from src.evaluation.evaluator import Evaluator
from src.utils.config import load_config, parse_cli_overrides, save_config
from src.utils.logging import MLflowLogger


# Check environment before starting training
subprocess.run(["python", "scripts/check_environment.py"], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or 'latest')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    
    # Parse known args, rest are config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides
    args.overrides = parse_cli_overrides(unknown)
    
    return args


def build_model(config: dict, device: torch.device):
    """Build unconditional UNet model (matching train11 notebook)."""
    model_config = config.get("model", {}).get("unet", {})
    
    # Calculate block out channels from config
    block_out_channels = tuple(
        model_config["model_channels"] * mult 
        for mult in model_config["channel_mult"]
    )
    num_levels = len(block_out_channels)
    
    # Unconditional UNet2DModel (no cross-attention, no text conditioning)
    # Matching train11_hpc_imagenet_diffusers.ipynb exactly
    model = UNet2DModel(
        sample_size=config.get("data", {}).get("resolution", 256) // 8,  # 32 for 256x256
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["num_res_blocks"],
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        )[:num_levels],  # Trim to match number of channels
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        )[:num_levels],  # Trim to match number of channels
    )
    
    return model


def build_scheduler(config: dict):
    """Build noise scheduler from config."""
    diffusion_config = config.get("diffusion", {})
    
    scheduler_config = SchedulerConfig(
        num_train_timesteps=diffusion_config["num_train_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
        beta_schedule=diffusion_config["beta_schedule"],
        prediction_type=diffusion_config["prediction_type"],
        clip_sample=diffusion_config.get("clip_sample", True),  # clip_sample can have default
    )
    
    scheduler = DDPMScheduler(scheduler_config)
    
    return scheduler


def build_optimizer(model: torch.nn.Module, config: dict):
    """Build optimizer from config."""
    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})
    
    lr = float(optimizer_config["learning_rate"])
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))  # weight_decay can default to 0
    betas = tuple(optimizer_config.get("betas", [0.9, 0.999]))  # betas can have default
    eps = float(optimizer_config.get("eps", 1e-8))  # eps can have default
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    
    return optimizer


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, overrides=args.overrides)
    
    # Override output dir if specified
    if args.output_dir:
        config["checkpoint"]["save_dir"] = args.output_dir
    
    # Get training config
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})
    
    # Set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        mixed_precision=training_config.get("mixed_precision", "bf16"),
        log_with="mlflow" if config.get("logging", {}).get("enabled", True) else None,
    )
    
    # Set seed
    seed = training_config.get("seed", 42)
    set_seed(seed)
    
    # Print info
    if accelerator.is_main_process:
        print(f"Accelerator initialized")
        print(f"  - Num processes: {accelerator.num_processes}")
        print(f"  - Mixed precision: {accelerator.mixed_precision}")
        print(f"  - Device: {accelerator.device}")
    
    # Create output directory
    output_dir = Path(checkpoint_config.get("save_dir", "outputs"))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        save_config(config, output_dir / "config.yaml")
    
    # Build components
    device = accelerator.device
    
    # VAE (frozen)
    vae_config = config.get("model", {}).get("vae", {})
    vae = VAEWrapper(
        pretrained=vae_config.get("pretrained", "stabilityai/sd-vae-ft-mse"),
        device=device,
    )
    
    # No text encoder needed for unconditional model
    
    # U-Net (trainable) - unconditional
    model = build_model(config, device)
    
    # Enable gradient checkpointing if specified
    if training_config.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    
    # Print model info
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"U-Net model: {num_params/1e6:.2f}M params ({num_trainable/1e6:.2f}M trainable)")
    
    # Noise scheduler
    scheduler = build_scheduler(config)
    scheduler.to(device)
    
    # Loss function
    diffusion_config = config.get("diffusion", {})
    loss_fn = DiffusionLoss(
        prediction_type=diffusion_config.get("prediction_type", "epsilon"),
        snr_gamma=diffusion_config.get("snr_gamma"),
    )
    
    # Optimizer
    optimizer = build_optimizer(model, config)
    
    # LR scheduler
    lr_scheduler = get_lr_scheduler(
        config=training_config.get("lr_scheduler", {"type": "cosine"}),
        optimizer=optimizer,
        num_training_steps=training_config.get("max_train_steps", 100000),
    )
    
    # EMA
    ema_config = training_config.get("ema", {})
    ema_model = None
    if ema_config.get("enabled", True):
        ema_model = EMAModel(
            model.parameters(),
            decay=ema_config.get("decay", 0.9999),
        )
    
    # Data loader - no tokenizer needed for unconditional model
    data_config = config.get("data", {})
    # Batch size should be in data config, not training config
    if "batch_size" not in data_config:
        raise ValueError("batch_size must be specified in data config")
    train_dataloader = get_dataloader(
        config=data_config,
        tokenizer=None,  # No tokenizer for unconditional
        accelerator=accelerator,
        split="train",
    )
    
    if accelerator.is_main_process:
        print("Starting accelerator.prepare()...")
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if accelerator.is_main_process:
        print("Accelerator.prepare() complete")
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=output_dir / "checkpoints",
        save_every_n_steps=checkpoint_config.get("save_every_n_steps", 5000),
        keep_last_n=checkpoint_config.get("keep_last_n", 5),
        save_ema_separately=checkpoint_config.get("save_ema_separately", True),
        resume_from_latest=checkpoint_config.get("resume_from_latest", True),
    )
    
    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        if args.resume == "latest":
            checkpoint_path = checkpoint_manager.get_latest_checkpoint()
        else:
            checkpoint_path = args.resume
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = checkpoint_manager.load(
                checkpoint_path,
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                scheduler=lr_scheduler,
                ema_model=ema_model,
            )
            start_step = checkpoint.get("global_step", 0)
            if accelerator.is_main_process:
                print(f"Resumed from step {start_step}")
    
    # MLflow logging (main process only)
    mlflow_logger = None
    mlflow_config = config.get("logging", {}).get("mlflow", {})
    if accelerator.is_main_process and mlflow_config.get("enabled", False):
        try:
            print("Initializing MLflow logger...")
            import signal
            
            # Set a timeout for MLflow initialization
            def timeout_handler(signum, frame):
                raise TimeoutError("MLflow initialization timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                mlflow_logger = MLflowLogger(config.get("logging", {}))
                print("MLflow logger initialized")
                mlflow_logger.log_config(config)
                
                # Log model architecture info
                num_params = sum(p.numel() for p in model.parameters())
                num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                mlflow_logger.log_metric("model/total_parameters", num_params, step=0)
                mlflow_logger.log_metric("model/trainable_parameters", num_trainable, step=0)
                mlflow_logger.log_metric("model/total_parameters_millions", num_params / 1e6, step=0)
                mlflow_logger.log_metric("model/trainable_parameters_millions", num_trainable / 1e6, step=0)
                
                mlflow_logger.set_tags({
                    "resolution": str(data_config.get("resolution", 256)),
                    "model_type": "stable_diffusion_unconditional",
                    "unet_architecture": "diffusers_unet2d",
                })
                print("MLflow tags set")
            finally:
                signal.alarm(0)  # Cancel the alarm
        except (TimeoutError, Exception) as e:
            print(f"Warning: MLflow initialization failed or timed out: {e}")
            print("Continuing without MLflow logging...")
            mlflow_logger = None
    
    # Wait for main process to finish MLflow initialization
    accelerator.wait_for_everyone()
    
    # Evaluator (main process only)
    evaluator = None
    eval_config = config.get("evaluation", {})
    if accelerator.is_main_process and eval_config.get("enabled", True):
        print("Initializing evaluator...")
        # Merge required configs for evaluator
        eval_full_config = {
            **eval_config,
            "diffusion": config.get("diffusion", {}),
            "resolution": data_config.get("resolution", 256),
        }
        evaluator = Evaluator(
            config=eval_full_config,
            device=device,
        )
        print("Evaluator initialized")
    
    # Wait for main process to finish evaluator initialization
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("Creating trainer...")
    # Create trainer (unconditional - no text encoder)
    trainer = StableDiffusionTrainer(
        config=config,
        unet=model,
        vae=vae,
        noise_scheduler=scheduler,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        evaluator=evaluator,
        mlflow_logger=mlflow_logger,
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print("\nTraining interrupted. Saving checkpoint...")
            trainer._save_checkpoint(force=True)
    finally:
        if mlflow_logger:
            mlflow_logger.end_run()
    
    if accelerator.is_main_process:
        print("Training complete!")


if __name__ == "__main__":
    main()
