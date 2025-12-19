"""
Noise Schedulers for Diffusion Models.

Implements:
- DDPM (Denoising Diffusion Probabilistic Models) scheduler
- DDIM (Denoising Diffusion Implicit Models) scheduler

These schedulers handle:
- Beta schedule (linear, scaled_linear, cosine)
- Adding noise to samples
- Computing alphas and related values
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import numpy as np


@dataclass
class SchedulerConfig:
    """Configuration for noise scheduler."""
    
    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"  # epsilon, v_prediction, sample
    clip_sample: bool = False
    set_alpha_to_one: bool = False
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models scheduler.
    
    Implements the forward diffusion process (adding noise)
    and provides utilities for training and sampling.
    """
    
    def __init__(
        self,
        config: Union[SchedulerConfig, dict],
    ):
        if isinstance(config, dict):
            config = SchedulerConfig(**config)
        
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.prediction_type = config.prediction_type
        
        # Compute betas
        self.betas = self._get_beta_schedule(
            config.beta_schedule,
            config.num_train_timesteps,
            config.beta_start,
            config.beta_end,
        )
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Shift alphas_cumprod for previous timestep
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])
        
        # Compute sqrt values for adding noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For v-prediction
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance for sampling
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Clamp to avoid log(0)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # SNR (Signal-to-Noise Ratio)
        self.snr = self.alphas_cumprod / (1.0 - self.alphas_cumprod)
    
    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.snr = self.snr.to(device)
        return self
    
    def _get_beta_schedule(
        self,
        schedule: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Compute beta schedule."""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        
        elif schedule == "scaled_linear":
            # Stable Diffusion uses this schedule
            return torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32
            ) ** 2
        
        elif schedule == "squaredcos_cap_v2":
            # Cosine schedule (Improved DDPM)
            return self._cosine_beta_schedule(num_timesteps)
        
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def _cosine_beta_schedule(
        self,
        num_timesteps: int,
        s: float = 0.008,
    ) -> torch.Tensor:
        """Cosine beta schedule from Improved DDPM."""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to timesteps.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_cumprod_t) * x_0, (1 - alpha_cumprod_t) * I)
        
        Args:
            original_samples: Clean samples [B, C, H, W]
            noise: Gaussian noise [B, C, H, W]
            timesteps: Timesteps [B]
        
        Returns:
            Noisy samples [B, C, H, W]
        """
        # Move schedule to correct device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        while sqrt_alpha_prod.dim() < original_samples.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # Add noise
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get velocity for v-prediction objective.
        
        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * sample
        
        Args:
            sample: Clean samples [B, C, H, W]
            noise: Noise [B, C, H, W]
            timesteps: Timesteps [B]
        
        Returns:
            Velocity [B, C, H, W]
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(sample.device)
        
        while sqrt_alpha_prod.dim() < sample.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        
        return velocity
    
    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get SNR for given timesteps."""
        return self.snr[timesteps]
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[dict, torch.Tensor]:
        """
        Single denoising step (for sampling).
        
        Args:
            model_output: Predicted noise or velocity from model
            timestep: Current timestep
            sample: Current noisy sample
            generator: Random generator for reproducibility
            return_dict: Whether to return a dict
        
        Returns:
            Previous sample (less noisy)
        """
        t = timestep
        
        # Get values for this timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Move to correct device
        device = sample.device
        alpha_prod_t = alpha_prod_t.to(device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device)
        
        # Predict x_0 based on prediction type
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - torch.sqrt(beta_prod_t).to(device) * model_output
            ) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                torch.sqrt(alpha_prod_t).to(device) * sample - 
                torch.sqrt(beta_prod_t).to(device) * model_output
            )
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip predicted x_0
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute posterior mean
        pred_prev_sample = (
            self.posterior_mean_coef1[t].to(device) * pred_original_sample +
            self.posterior_mean_coef2[t].to(device) * sample
        )
        
        # Add noise for t > 0
        if t > 0:
            noise = torch.randn(
                sample.shape, generator=generator, device=device, dtype=sample.dtype
            )
            variance = self.posterior_variance[t].to(device)
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise
        
        if not return_dict:
            return pred_prev_sample
        
        return {
            "prev_sample": pred_prev_sample,
            "pred_original_sample": pred_original_sample,
        }


class DDIMScheduler(DDPMScheduler):
    """
    Denoising Diffusion Implicit Models scheduler.
    
    Allows for faster sampling with fewer steps by using
    a deterministic sampling process.
    """
    
    def __init__(
        self,
        config: Union[SchedulerConfig, dict],
        eta: float = 0.0,  # 0 for deterministic, 1 for stochastic (DDPM)
    ):
        super().__init__(config)
        self.eta = eta
        self.timesteps = None
        self.num_inference_steps = None
    
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device] = None,
    ):
        """
        Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of denoising steps
            device: Device for timesteps tensor
        """
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            np.arange(0, num_inference_steps) * step_ratio
        ).round()[::-1].copy().astype(np.int64)
        
        self.timesteps = torch.from_numpy(timesteps)
        
        if device is not None:
            self.timesteps = self.timesteps.to(device)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[dict, torch.Tensor]:
        """
        DDIM denoising step.
        
        Args:
            model_output: Predicted noise or velocity
            timestep: Current timestep
            sample: Current noisy sample
            generator: Random generator
            return_dict: Whether to return dict
        
        Returns:
            Previous sample
        """
        # Find previous timestep
        step_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if step_idx.numel() == 0:
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        else:
            step_idx = step_idx.item()
            if step_idx + 1 < len(self.timesteps):
                prev_timestep = self.timesteps[step_idx + 1].item()
            else:
                prev_timestep = 0
        
        device = sample.device
        
        # Get alphas
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep].to(device) if prev_timestep >= 0 
            else torch.tensor(1.0, device=device)
        )
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - torch.sqrt(beta_prod_t) * model_output
            ) / torch.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                torch.sqrt(alpha_prod_t) * sample - 
                torch.sqrt(beta_prod_t) * model_output
            )
            pred_epsilon = (
                torch.sqrt(alpha_prod_t) * model_output + 
                torch.sqrt(beta_prod_t) * sample
            )
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - torch.sqrt(alpha_prod_t) * pred_original_sample
            ) / torch.sqrt(beta_prod_t)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip prediction
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = self.eta * torch.sqrt(variance)
        
        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev ** 2) * pred_epsilon
        
        # Compute previous sample
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        )
        
        # Add noise if eta > 0
        if self.eta > 0 and prev_timestep > 0:
            noise = torch.randn(sample.shape, generator=generator, device=device, dtype=sample.dtype)
            prev_sample = prev_sample + std_dev * noise
        
        if not return_dict:
            return prev_sample
        
        return {
            "prev_sample": prev_sample,
            "pred_original_sample": pred_original_sample,
        }
