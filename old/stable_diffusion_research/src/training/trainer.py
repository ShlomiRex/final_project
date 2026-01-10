"""
Main Trainer Class for Stable Diffusion.

Handles:
- Training loop orchestration
- Multi-GPU coordination via Accelerate
- Checkpoint management
- Evaluation triggering
- MLflow logging
"""

import logging
import time
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from PIL import Image

from ..diffusion.noise_scheduler import DDPMScheduler
from ..diffusion.loss import DiffusionLoss
from .checkpoint import CheckpointManager
from .ema import EMAModel
from .lr_scheduler import get_lr_scheduler


class StableDiffusionTrainer:
    """
    Trainer for Stable Diffusion models.
    
    Orchestrates the full training loop including:
    - Forward pass through diffusion training objective
    - Gradient accumulation
    - EMA updates
    - Checkpointing
    - Evaluation
    - Logging
    """
    
    def __init__(
        self,
        config: dict,
        unet: nn.Module,
        vae: nn.Module,
        noise_scheduler: DDPMScheduler,
        train_dataloader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        accelerator: Accelerator,
        evaluator=None,
        mlflow_logger=None,
    ):
        """
        Args:
            config: Training configuration dict
            unet: U-Net model (trainable)
            vae: VAE model (frozen)
            noise_scheduler: Diffusion noise scheduler
            train_dataloader: Training data loader
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            accelerator: Accelerate accelerator
            evaluator: Evaluation handler (optional)
            mlflow_logger: MLflow logger (optional)
        """
        self.config = config
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.evaluator = evaluator
        self.mlflow_logger = mlflow_logger
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Get config values with defaults
        training_config = config.get("training", {})
        self.max_train_steps = training_config.get("max_train_steps", 500000)
        self.gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = training_config.get("max_grad_norm", 1.0)
        
        # Diffusion config
        diffusion_config = config.get("diffusion", {})
        self.prediction_type = diffusion_config.get("prediction_type", "epsilon")
        self.offset_noise = diffusion_config.get("offset_noise", 0.0)
        
        # No CFG for unconditional model
        
        # EMA
        ema_config = training_config.get("ema", {})
        self.use_ema = ema_config.get("enabled", True)
        if self.use_ema:
            self.ema = EMAModel(
                parameters=self.accelerator.unwrap_model(unet).parameters(),
                decay=ema_config.get("decay", 0.9999),
                update_after_step=ema_config.get("update_after_step", 0),
                update_every=ema_config.get("update_every", 1),
            )
        else:
            self.ema = None
        
        # Checkpoint manager
        checkpoint_config = config.get("checkpoint", {})
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_config.get("checkpoint_dir", "checkpoints"),
            save_every_n_steps=checkpoint_config.get("save_every_n_steps", 5000),
            keep_last_n=checkpoint_config.get("keep_last_n", 5),
            save_ema_separately=checkpoint_config.get("save_ema_separately", True),
            resume_from_latest=checkpoint_config.get("resume_from_latest", True),
        )
        
        # Evaluation config
        eval_config = config.get("evaluation", {})
        self.eval_every_n_steps = eval_config.get("eval_every_n_steps", 5000)
        self.eval_at_start = eval_config.get("eval_at_start", False)
        
        # Logging config
        logging_config = config.get("logging", {})
        self.log_every_n_steps = logging_config.get("mlflow", {}).get("log_every_n_steps", 10)  # Log to MLflow frequently
        self.console_log_every = logging_config.get("console", {}).get("log_every_n_steps", 500)  # Reduce console spam
        
        # Loss function
        self.loss_fn = DiffusionLoss(
            prediction_type=self.prediction_type,
            snr_gamma=diffusion_config.get("snr_gamma"),
        )
        
        # VAE scale factor
        vae_config = config.get("model", {}).get("vae", {})
        self.vae_scale_factor = vae_config.get("scale_factor", 0.18215)
        
        # Setup logging with timestamps
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True  # Override any existing configuration
        )
        self.logger = logging.getLogger(__name__)
        # Ensure unbuffered output
        for handler in logging.root.handlers:
            handler.flush = lambda: sys.stdout.flush() if hasattr(sys.stdout, 'flush') else None
    
    def train(self):
        """Main training loop."""
        # Resume from checkpoint if exists
        if self.checkpoint_manager.resume_from_latest and self.checkpoint_manager.exists():
            try:
                self.global_step = self.checkpoint_manager.load(
                    model=self.accelerator.unwrap_model(self.unet),
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    ema=self.ema,
                    device=self.accelerator.device,
                )
                self.accelerator.print(f"Resumed from checkpoint at step {self.global_step}")
            except (RuntimeError, KeyError) as e:
                self.accelerator.print(f"\n⚠️  WARNING: Failed to load checkpoint: {str(e)[:200]}")
                self.accelerator.print("This is expected when switching from custom UNet to Diffusers UNet.")
                self.accelerator.print("Starting training from scratch with fresh weights.\n")
                self.global_step = 0
        
        # Verify multi-GPU setup BEFORE any MLflow operations
        if self.accelerator.is_main_process:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            num_processes = self.accelerator.num_processes
            self.accelerator.print(f"\n{'='*60}")
            self.accelerator.print(f"Multi-GPU Configuration:")
            self.accelerator.print(f"  Total GPUs available: {num_gpus}")
            self.accelerator.print(f"  Accelerator processes: {num_processes}")
            self.accelerator.print(f"  Device: {self.accelerator.device}")
            
            if num_gpus > 0:
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    self.accelerator.print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
                
                # Check initial memory usage
                gpu_stats = self._get_gpu_memory_stats()
                for i in range(num_gpus):
                    allocated = gpu_stats.get(f"system/gpu_{i}/memory_allocated_gb", 0)
                    total = gpu_stats.get(f"system/gpu_{i}/memory_total_gb", 0)
                    self.accelerator.print(f"  GPU {i} initial VRAM: {allocated:.2f}GB / {total:.0f}GB")
            
            self.accelerator.print(f"{'='*60}\n")
        
        # Log dataset samples at the beginning (only once, not on resume)
        # DISABLED: Causes NCCL issues with multi-GPU when iterating dataloader
        # if self.global_step == 0 and self.accelerator.is_main_process:
        #     self._log_dataset_samples()
        
        # Wait for all processes after logging
        self.accelerator.wait_for_everyone()
        
        # Initial evaluation
        # DISABLED: Causes multi-GPU deadlock
        # if self.eval_at_start and self.accelerator.is_main_process:
        #     self._run_evaluation()
        
        # Training loop
        num_update_steps_per_epoch = len(self.train_dataloader)
        num_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)
        
        # Disable progress bar to reduce log clutter - use MLflow for tracking
        progress_bar = tqdm(
            range(self.global_step, self.max_train_steps),
            desc="Training",
            disable=True,  # Disabled - metrics tracked in MLflow
        )
        
        self.unet.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Generate samples at the start of each epoch (except epoch 0 which was already done)
            if epoch > 0 and self.accelerator.is_main_process:
                self.accelerator.print(f"\nGenerating samples at start of epoch {epoch}...")
                self._generate_checkpoint_samples()
            
            for batch in self.train_dataloader:
                step_start_time = time.time()
                
                # Training step
                loss = self._training_step(batch)
                
                step_duration = time.time() - step_start_time
                if self.global_step % 100 == 0 and self.accelerator.is_main_process:
                    self.logger.info(f"Step {self.global_step}: Total time={step_duration:.3f}s, Loss={loss:.4f}")
                
                # Logging
                if self.global_step % self.console_log_every == 0:
                    # Get GPU memory stats
                    gpu_stats = self._get_gpu_memory_stats()
                    gpu_info = ""
                    if gpu_stats:
                        num_gpus = torch.cuda.device_count()
                        gpu_util = [f"GPU{i}: {gpu_stats.get(f'system/gpu_{i}/memory_allocated_gb', 0):.1f}GB/{gpu_stats.get(f'system/gpu_{i}/memory_total_gb', 0):.0f}GB" 
                                   for i in range(num_gpus)]
                        gpu_info = " | " + " ".join(gpu_util)
                    
                    # Print to console occasionally
                    steps_per_epoch = len(self.train_dataloader)
                    step_in_epoch = self.global_step % steps_per_epoch
                    self.accelerator.print(
                        f"Epoch {epoch} Step {step_in_epoch}/{steps_per_epoch} (Global: {self.global_step}/{self.max_train_steps}) | "
                        f"Loss: {loss:.4f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e}{gpu_info}"
                    )
                
                if self.global_step % self.log_every_n_steps == 0:
                    # Log to MLflow frequently for detailed tracking
                    steps_per_epoch = len(self.train_dataloader)
                    metrics = {
                        "train/loss": loss,
                        "train/lr": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                        "train/step_in_epoch": self.global_step % steps_per_epoch,
                        "train/steps_per_epoch": steps_per_epoch,
                    }
                    # Add GPU memory stats
                    gpu_stats = self._get_gpu_memory_stats()
                    metrics.update(gpu_stats)
                    
                    self._log_metrics(metrics)
                
                # Checkpointing
                if self.checkpoint_manager.should_save(self.global_step):
                    self._save_checkpoint()
                    
                    # Generate samples after saving checkpoint (main process only)
                    if self.accelerator.is_main_process:
                        self._generate_checkpoint_samples()
                    
                    # Wait for all processes to finish
                    self.accelerator.wait_for_everyone()
                
                self.global_step += 1
                progress_bar.update(1)
                
                if self.global_step >= self.max_train_steps:
                    break
            
            if self.global_step >= self.max_train_steps:
                break
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Final sample generation (main process only)
        if self.accelerator.is_main_process:
            self._generate_checkpoint_samples()
        
        # Wait for all processes to finish
        self.accelerator.wait_for_everyone()
        
        self.accelerator.print("Training complete!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
        
        Returns:
            Loss value
        """
        with self.accelerator.accumulate(self.unet):
            # Get batch data (unconditional - images only)
            pixel_values = batch["pixel_values"]
            
            # Encode images to latents
            vae_start = time.time()
            with torch.no_grad():
                latents = self.vae.encode(pixel_values)
                if hasattr(latents, 'latent_dist'):
                    latents = latents.latent_dist.sample()
                latents = latents * self.vae_scale_factor
            vae_time = time.time() - vae_start
            
            # Sample noise
            noise_start = time.time()
            noise = torch.randn_like(latents)
            
            # Add offset noise if enabled
            if self.offset_noise > 0:
                offset = torch.randn(
                    latents.shape[0], latents.shape[1], 1, 1,
                    device=latents.device, dtype=latents.dtype
                )
                noise = noise + self.offset_noise * offset
            
            # Sample timesteps
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps,
                (batch_size,), device=latents.device, dtype=torch.long
            )
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            noise_time = time.time() - noise_start
            
            # Predict noise (unconditional - no text conditioning)
            forward_start = time.time()
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                return_dict=False,
            )
            
            if isinstance(model_pred, tuple):
                model_pred = model_pred[0]
            forward_time = time.time() - forward_start
            
            # Get target and compute loss
            loss_start = time.time()
            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
            snr = self.noise_scheduler.get_snr(timesteps).to(latents.device) if hasattr(self.noise_scheduler, 'get_snr') else None
            loss = self.loss_fn(model_pred, target, timesteps, snr)
            loss_time = time.time() - loss_start
            
            # Backward
            backward_start = time.time()
            self.accelerator.backward(loss)
            backward_time = time.time() - backward_start
            
            # Gradient clipping
            clip_start = time.time()
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.unet.parameters(), self.max_grad_norm)
            clip_time = time.time() - clip_start
            
            # Optimizer step
            optim_start = time.time()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            optim_time = time.time() - optim_start
            
            # Log detailed timing every 100 steps
            if self.global_step % 100 == 0 and self.accelerator.is_main_process:
                self.logger.info(f"  VAE encode: {vae_time:.3f}s")
                self.logger.info(f"  Noise+timesteps: {noise_time:.3f}s")
                self.logger.info(f"  UNet forward: {forward_time:.3f}s")
                self.logger.info(f"  Loss calc: {loss_time:.3f}s")
                self.logger.info(f"  Backward: {backward_time:.3f}s")
                self.logger.info(f"  Grad clip: {clip_time:.3f}s")
                self.logger.info(f"  Optimizer: {optim_time:.3f}s")
            
            # EMA update
            if self.use_ema:
                self.ema.step(self.accelerator.unwrap_model(self.unet).parameters())
        
        return loss.detach().item()
    
    def _save_checkpoint(self):
        """Save a training checkpoint."""
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            # Log GPU memory stats before generating samples
            gpu_stats = self._get_gpu_memory_stats()
            if gpu_stats:
                self.accelerator.print(f"\nGPU Memory at checkpoint {self.global_step}:")
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    util = gpu_stats.get(f"system/gpu_{i}/memory_utilization_pct", 0)
                    allocated = gpu_stats.get(f"system/gpu_{i}/memory_allocated_gb", 0)
                    total = gpu_stats.get(f"system/gpu_{i}/memory_total_gb", 0)
                    self.accelerator.print(f"  GPU {i}: {allocated:.2f}GB / {total:.0f}GB ({util:.1f}% utilized)")
                self.accelerator.print()
            
            # Generate samples before saving checkpoint
            self._generate_checkpoint_samples()
            
            self.checkpoint_manager.save(
                step=self.global_step,
                model=self.unet,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                ema=self.ema,
                config=self.config,
                extra_state={"epoch": self.epoch},
                accelerator=self.accelerator,
            )
            self.accelerator.print(f"Saved checkpoint at step {self.global_step}")
    
    def _run_evaluation(self):
        """Run evaluation and log results."""
        if self.evaluator is None:
            return
        
        self.accelerator.print(f"Running evaluation at step {self.global_step}...")
        
        # Use EMA weights for evaluation if available
        unet_for_eval = self.accelerator.unwrap_model(self.unet)
        
        if self.use_ema:
            self.ema.store(unet_for_eval.parameters())
            self.ema.copy_to(unet_for_eval.parameters())
        
        try:
            unet_for_eval.eval()
            
            # Generate samples
            samples = self.evaluator.generate_samples(
                unet=unet_for_eval,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=self.noise_scheduler,
                device=self.accelerator.device,
            )
            
            # Log samples
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_images(samples, step=self.global_step)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(
                unet=unet_for_eval,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                scheduler=self.noise_scheduler,
                device=self.accelerator.device,
            )
            
            # Log metrics
            self._log_metrics({f"eval/{k}": v for k, v in metrics.items()})
            
            self.accelerator.print(f"Evaluation metrics: {metrics}")
            
        finally:
            # Restore original weights
            if self.use_ema:
                self.ema.restore(unet_for_eval.parameters())
            
            unet_for_eval.train()
    
    def _generate_checkpoint_samples(self):
        """Generate and save sample images at checkpoint time (unconditional)."""
        from pathlib import Path
        
        self.accelerator.print(f"Generating unconditional samples for checkpoint at step {self.global_step}...")
        
        # Get config
        eval_config = self.config.get("evaluation", {})
        num_samples = min(eval_config.get("num_samples", 16), 16)  # Limit to 16 for grid
        diffusion_config = self.config.get("diffusion", {})
        num_inference_steps = diffusion_config.get("num_inference_steps", 50)
        
        # Use EMA weights if available
        unet_for_eval = self.accelerator.unwrap_model(self.unet)
        if self.use_ema:
            self.ema.store(unet_for_eval.parameters())
            self.ema.copy_to(unet_for_eval.parameters())
        
        try:
            unet_for_eval.eval()
            
            # Generate samples (unconditional)
            latent_height = self.config.get("data", {}).get("resolution", 256) // 8
            latent_shape = (num_samples, 4, latent_height, latent_height)
            
            # Use a subset of timesteps for faster inference
            timestep_indices = torch.linspace(
                self.noise_scheduler.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.long
            )
            
            with torch.no_grad():
                # Start from pure noise
                latents = torch.randn(latent_shape, device=self.accelerator.device, dtype=torch.float32)
                
                # Denoise iteratively
                for t_idx in timestep_indices:
                    t = int(t_idx.item())
                    timesteps = torch.full((num_samples,), t, device=latents.device, dtype=torch.long)
                    
                    # Predict noise
                    model_output = unet_for_eval(latents, timesteps)
                    noise_pred = model_output.sample if hasattr(model_output, 'sample') else model_output
                    
                    # Scheduler step
                    step_output = self.noise_scheduler.step(noise_pred, t, latents)
                    latents = step_output["prev_sample"] if isinstance(step_output, dict) else step_output
                
                # Decode latents to images
                latents = latents / self.vae_scale_factor
                vae_output = self.vae.decode(latents)
                images = vae_output.sample if hasattr(vae_output, 'sample') else vae_output
                
                # Convert to PIL images
                images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                pil_images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
            
            # Create simple grid
            from PIL import Image as PILImage
            cols = 4
            rows = (len(pil_images) + cols - 1) // cols
            img_w, img_h = pil_images[0].size
            grid = PILImage.new('RGB', (cols * img_w, rows * img_h))
            for idx, img in enumerate(pil_images):
                grid.paste(img, ((idx % cols) * img_w, (idx // cols) * img_h))
            
            # Save locally
            samples_dir = Path(self.checkpoint_manager.checkpoint_dir).parent / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            grid_path = samples_dir / f"samples_step_{self.global_step:08d}.png"
            grid.save(grid_path)
            self.accelerator.print(f"Saved unconditional sample grid to {grid_path}")
            
            # Log to MLflow
            if self.mlflow_logger is not None:
                # Log individual images
                image_dict = {f"sample_{i:02d}": img for i, img in enumerate(pil_images)}
                self.mlflow_logger.log_images(image_dict, step=self.global_step, prefix="checkpoint_samples")
                
                # Also log the grid
                self.mlflow_logger.log_image_grid(grid, step=self.global_step, name="checkpoint_samples_grid")
                
            self.accelerator.print(f"Generated {len(images)} samples for step {self.global_step}")
            
        except Exception as e:
            self.accelerator.print(f"Warning: Failed to generate checkpoint samples: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original weights
            if self.use_ema:
                self.ema.restore(unet_for_eval.parameters())
            unet_for_eval.train()
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to MLflow."""
        if self.mlflow_logger is not None:
            self.mlflow_logger.log_metrics(metrics, step=self.global_step)
    
    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics for all devices."""
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        num_gpus = torch.cuda.device_count()
        
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            # Prefix with 'system/' for MLflow categorization
            stats[f"system/gpu_{i}/memory_allocated_gb"] = allocated
            stats[f"system/gpu_{i}/memory_reserved_gb"] = reserved
            stats[f"system/gpu_{i}/memory_max_allocated_gb"] = max_allocated
            stats[f"system/gpu_{i}/memory_total_gb"] = total
            stats[f"system/gpu_{i}/memory_utilization_pct"] = (allocated / total) * 100 if total > 0 else 0
        
        # Overall stats
        total_allocated = sum(torch.cuda.memory_allocated(i) for i in range(num_gpus)) / 1024**3
        total_reserved = sum(torch.cuda.memory_reserved(i) for i in range(num_gpus)) / 1024**3
        stats[f"system/gpu/total_allocated_gb"] = total_allocated
        stats[f"system/gpu/total_reserved_gb"] = total_reserved
        
        return stats
    
    def _log_dataset_samples(self, num_samples: int = 16):
        """Log sample images from the dataset to MLflow (unconditional - no text)."""
        if self.mlflow_logger is None:
            return
        
        if self.accelerator.is_main_process:
            self.accelerator.print("Logging dataset samples to MLflow...")
        
        try:
            from PIL import Image
            import numpy as np
            
            samples = {}
            count = 0
            
            # CRITICAL: ALL processes must participate in dataloader iteration
            # to avoid DDP/NCCL desynchronization. Only main process logs.
            dataloader_iter = iter(self.train_dataloader)
            
            # Get samples from dataloader iterator
            while count < num_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                
                # Only main process needs to process and log
                if self.accelerator.is_main_process:
                    pixel_values = batch["pixel_values"]
                    
                    # Move to CPU and convert to images
                    pixel_values = pixel_values.cpu()
                    
                    for i in range(pixel_values.shape[0]):
                        if count >= num_samples:
                            break
                        
                        # Convert from [-1, 1] to [0, 255]
                        img_array = pixel_values[i].permute(1, 2, 0).numpy()
                        img_array = ((img_array + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                        img = Image.fromarray(img_array)
                        
                        samples[f"sample_{count:02d}"] = img
                        count += 1
                else:
                    # Other processes just count to stay in sync
                    count += batch["pixel_values"].shape[0]
            
            # Only main process logs to MLflow
            if samples and self.accelerator.is_main_process:
                self.mlflow_logger.log_images(samples, step=0, prefix="dataset_samples")
                self.accelerator.print(f"Logged {len(samples)} dataset samples to MLflow")
        
        except Exception as e:
            if self.accelerator.is_main_process:
                self.accelerator.print(f"Warning: Failed to log dataset samples: {e}")
    
    @property
    def training(self) -> bool:
        """Check if model is in training mode."""
        return self.unet.training
