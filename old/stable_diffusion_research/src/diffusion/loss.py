"""
Loss Functions for Diffusion Models.

Implements:
- Simple MSE loss (epsilon prediction)
- V-prediction loss
- Min-SNR loss weighting
- Offset noise
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Loss function for training diffusion models.
    
    Supports multiple prediction types and loss weighting strategies.
    """
    
    def __init__(
        self,
        prediction_type: str = "epsilon",
        snr_gamma: Optional[float] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            prediction_type: What the model predicts ("epsilon", "v_prediction", "sample")
            snr_gamma: Min-SNR weighting gamma (None to disable)
            reduction: Loss reduction ("mean", "sum", "none")
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        self.reduction = reduction
    
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            model_output: Model prediction [B, C, H, W]
            target: Target (noise or velocity) [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            snr: Signal-to-noise ratio for each timestep [B]
        
        Returns:
            Loss value
        """
        # Compute per-sample MSE loss
        loss = F.mse_loss(model_output, target, reduction="none")
        
        # Reduce spatial dimensions
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # [B]
        
        # Apply Min-SNR weighting
        if self.snr_gamma is not None and snr is not None:
            snr_weights = self._compute_snr_weights(snr)
            loss = loss * snr_weights
        
        # Final reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def _compute_snr_weights(self, snr: torch.Tensor) -> torch.Tensor:
        """
        Compute Min-SNR loss weights.
        
        From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        
        Args:
            snr: Signal-to-noise ratio [B]
        
        Returns:
            Loss weights [B]
        """
        # Clamp SNR to gamma
        snr_clipped = torch.clamp(snr, max=self.snr_gamma)
        
        if self.prediction_type == "epsilon":
            # Weight = min(SNR, gamma) / SNR
            weights = snr_clipped / snr
        elif self.prediction_type == "v_prediction":
            # Weight = min(SNR, gamma) / (SNR + 1)
            weights = snr_clipped / (snr + 1)
        else:
            weights = torch.ones_like(snr)
        
        return weights


class VLBLoss(nn.Module):
    """
    Variational Lower Bound loss for diffusion models.
    
    Combines reconstruction loss with KL divergence terms.
    Used for more principled training but rarely used in practice.
    """
    
    def __init__(
        self,
        scheduler,
        lambda_vlb: float = 0.001,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.lambda_vlb = lambda_vlb
    
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict:
        """
        Compute VLB loss.
        
        Args:
            model_output: Model prediction
            target: Target noise
            x_start: Clean samples
            x_t: Noisy samples
            timesteps: Timesteps
        
        Returns:
            Dictionary with loss components
        """
        # Simple loss
        simple_loss = F.mse_loss(model_output, target)
        
        # VLB loss (simplified - actual VLB is more complex)
        vlb_loss = self._compute_vlb(x_start, x_t, model_output, timesteps)
        
        # Combined loss
        total_loss = simple_loss + self.lambda_vlb * vlb_loss
        
        return {
            "loss": total_loss,
            "simple_loss": simple_loss,
            "vlb_loss": vlb_loss,
        }
    
    def _compute_vlb(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VLB term (simplified)."""
        # This is a simplified version - full VLB requires more computation
        # In practice, simple MSE loss works well
        return F.mse_loss(model_output, x_start - x_t)


def compute_loss_with_offset_noise(
    unet: nn.Module,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    noise_scheduler,
    offset_noise_strength: float = 0.1,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """
    Compute loss with offset noise for better handling of bright/dark images.
    
    From "Common Diffusion Noise Schedules and Sample Steps are Flawed"
    
    Args:
        unet: U-Net model
        latents: Clean latents
        noise: Gaussian noise
        timesteps: Diffusion timesteps
        encoder_hidden_states: Text embeddings
        noise_scheduler: Noise scheduler
        offset_noise_strength: Strength of offset noise
        prediction_type: Prediction type
    
    Returns:
        Loss value
    """
    # Add offset noise
    offset_noise = torch.randn(
        latents.shape[0], latents.shape[1], 1, 1,
        device=latents.device, dtype=latents.dtype
    )
    noise = noise + offset_noise_strength * offset_noise
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Predict
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)
    if isinstance(model_pred, tuple):
        model_pred = model_pred[0]
    
    # Get target
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    
    # Compute loss
    loss = F.mse_loss(model_pred, target)
    
    return loss
