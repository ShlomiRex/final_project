"""
VAE Wrapper for Stable Diffusion.

Provides a consistent interface for pretrained VAE models.
"""

from typing import Optional, Union

import torch
import torch.nn as nn


class VAEWrapper(nn.Module):
    """
    Wrapper for pretrained VAE models.
    
    Provides a simple interface for encoding images to latents
    and decoding latents back to images.
    
    Note: The VAE is frozen during training - only the U-Net learns.
    """
    
    def __init__(
        self,
        pretrained: str = "stabilityai/sd-vae-ft-mse",
        scale_factor: float = 0.18215,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.pretrained = pretrained
        self.scale_factor = scale_factor
        self._device = device
        
        # Load pretrained VAE
        self.vae = self._load_pretrained(pretrained)
        
        # Move to device if specified
        if device is not None:
            self.vae = self.vae.to(device)
        
        # Freeze VAE
        self.vae.requires_grad_(False)
        self.vae.eval()
    
    def _load_pretrained(self, pretrained: str):
        """Load pretrained VAE from HuggingFace."""
        from diffusers import AutoencoderKL
        
        vae = AutoencoderKL.from_pretrained(
            pretrained,
            torch_dtype=torch.float32,
        )
        
        return vae
    
    def encode(
        self,
        images: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: Image tensor [B, 3, H, W] in range [-1, 1]
            return_dict: Whether to return a dict with distribution
        
        Returns:
            Latent tensor [B, 4, H/8, W/8]
        """
        with torch.no_grad():
            posterior = self.vae.encode(images).latent_dist
            
            if return_dict:
                return posterior
            
            # Sample and scale
            latents = posterior.sample()
            latents = latents * self.scale_factor
            
            return latents
    
    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents to images.
        
        Args:
            latents: Latent tensor [B, 4, H/8, W/8]
        
        Returns:
            Image tensor [B, 3, H, W] in range [-1, 1]
        """
        with torch.no_grad():
            # Unscale
            latents = latents / self.scale_factor
            
            # Decode
            images = self.vae.decode(latents).sample
            
            return images
    
    def encode_deterministic(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images using the mean of the posterior (deterministic).
        
        Args:
            images: Image tensor [B, 3, H, W] in range [-1, 1]
        
        Returns:
            Latent tensor [B, 4, H/8, W/8]
        """
        with torch.no_grad():
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.mode()
            latents = latents * self.scale_factor
            
            return latents
    
    @property
    def latent_channels(self) -> int:
        """Number of latent channels."""
        return 4
    
    @property
    def downsample_factor(self) -> int:
        """Spatial downsampling factor."""
        return 8
    
    def get_latent_size(self, image_size: int) -> int:
        """Calculate latent size for a given image size."""
        return image_size // self.downsample_factor
    
    def to(self, device: torch.device):
        """Move VAE to device."""
        self.vae = self.vae.to(device)
        self._device = device
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for reconstruction)."""
        latents = self.encode(x)
        return self.decode(latents)


def load_vae(
    pretrained: str = "stabilityai/sd-vae-ft-mse",
    device: Optional[torch.device] = None,
) -> VAEWrapper:
    """
    Load a pretrained VAE.
    
    Args:
        pretrained: HuggingFace model ID
        device: Device to load model on
    
    Returns:
        VAEWrapper instance
    """
    vae = VAEWrapper(pretrained=pretrained, device=device)
    
    if device is not None:
        vae = vae.to(device)
    
    return vae
