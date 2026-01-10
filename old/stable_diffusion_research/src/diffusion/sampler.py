"""
Diffusion Sampler.

Handles the full sampling loop for generating images from noise.
Supports various sampling algorithms and classifier-free guidance.
"""

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from .noise_scheduler import DDIMScheduler, DDPMScheduler


class DiffusionSampler:
    """
    Sampler for diffusion models.
    
    Handles the full denoising loop from noise to clean samples.
    Supports:
    - DDPM and DDIM sampling
    - Classifier-free guidance
    - Progress tracking
    """
    
    def __init__(
        self,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        guidance_scale: float = 7.5,
    ):
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
    
    @torch.no_grad()
    def sample(
        self,
        unet: nn.Module,
        shape: tuple,
        encoder_hidden_states: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the diffusion model.
        
        Args:
            unet: U-Net model for denoising
            shape: Output shape (B, C, H, W)
            encoder_hidden_states: Text embeddings [B, L, D] or [2*B, L, D] for CFG
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance (None uses default)
            generator: Random generator for reproducibility
            latents: Optional starting latents (for img2img)
            callback: Optional callback function called every callback_steps
            callback_steps: Frequency of callback calls
            show_progress: Whether to show progress bar
        
        Returns:
            Generated latents [B, C, H, W]
        """
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        device = encoder_hidden_states.device
        batch_size = shape[0]
        
        # Check if CFG is being used (encoder_hidden_states has 2x batch size)
        do_cfg = guidance_scale > 1.0 and encoder_hidden_states.shape[0] == 2 * batch_size
        
        # Set timesteps
        if isinstance(self.scheduler, DDIMScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        else:
            # DDPM uses all timesteps
            step_ratio = self.scheduler.num_train_timesteps // num_inference_steps
            timesteps = torch.arange(
                self.scheduler.num_train_timesteps - 1, -1, -step_ratio, device=device
            )
        
        # Initialize latents
        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=encoder_hidden_states.dtype
            )
        
        # Denoising loop
        progress_bar = tqdm(
            timesteps, desc="Sampling", disable=not show_progress
        )
        
        for i, t in enumerate(progress_bar):
            # Expand latents for CFG
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            # Predict noise
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )
            
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]
            
            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            
            # Denoise step
            t_int = t.item() if isinstance(t, torch.Tensor) else t
            step_output = self.scheduler.step(
                noise_pred, t_int, latents, generator=generator
            )
            
            if isinstance(step_output, dict):
                latents = step_output["prev_sample"]
            else:
                latents = step_output
            
            # Callback
            if callback is not None and (i + 1) % callback_steps == 0:
                callback(i + 1, len(timesteps), latents)
        
        return latents
    
    @torch.no_grad()
    def sample_with_intermediate(
        self,
        unet: nn.Module,
        shape: tuple,
        encoder_hidden_states: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        save_intermediate_every: int = 10,
    ) -> tuple:
        """
        Generate samples and save intermediate results.
        
        Useful for visualizing the denoising process.
        
        Args:
            unet: U-Net model
            shape: Output shape
            encoder_hidden_states: Text embeddings
            num_inference_steps: Number of steps
            guidance_scale: CFG scale
            generator: Random generator
            save_intermediate_every: Save intermediate every N steps
        
        Returns:
            Tuple of (final_latents, list of intermediate latents)
        """
        intermediates = []
        
        def save_intermediate(step: int, total_steps: int, latents: torch.Tensor):
            if step % save_intermediate_every == 0 or step == total_steps:
                intermediates.append(latents.clone())
        
        final_latents = self.sample(
            unet=unet,
            shape=shape,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback=save_intermediate,
            callback_steps=1,
        )
        
        return final_latents, intermediates


class CFGSampler(DiffusionSampler):
    """
    Sampler with explicit classifier-free guidance support.
    
    Automatically handles the unconditional/conditional split.
    """
    
    def __init__(
        self,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
    ):
        super().__init__(scheduler, guidance_scale)
        self.guidance_rescale = guidance_rescale
    
    @torch.no_grad()
    def sample(
        self,
        unet: nn.Module,
        shape: tuple,
        cond_embeddings: torch.Tensor,
        uncond_embeddings: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples with explicit conditional and unconditional embeddings.
        
        Args:
            unet: U-Net model
            shape: Output shape (B, C, H, W)
            cond_embeddings: Conditional text embeddings [B, L, D]
            uncond_embeddings: Unconditional embeddings [B, L, D]
            num_inference_steps: Number of steps
            guidance_scale: CFG scale (None uses default)
            generator: Random generator
            latents: Optional starting latents
            show_progress: Whether to show progress bar
        
        Returns:
            Generated latents [B, C, H, W]
        """
        # Concatenate embeddings for batched inference
        encoder_hidden_states = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
        
        return super().sample(
            unet=unet,
            shape=shape,
            encoder_hidden_states=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            show_progress=show_progress,
        )
    
    def _rescale_noise_cfg(
        self,
        noise_pred: torch.Tensor,
        noise_pred_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rescale classifier-free guidance to prevent over-saturation.
        
        From "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        """
        if self.guidance_rescale <= 0:
            return noise_pred
        
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
        
        # Rescale
        factor = std_text / (std_cfg + 1e-8)
        noise_pred = noise_pred * (self.guidance_rescale * factor + (1 - self.guidance_rescale))
        
        return noise_pred
