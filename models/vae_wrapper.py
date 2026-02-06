"""
VAE Wrapper for CelebA-HQ Latent Diffusion

Provides a frozen pretrained VAE for encoding images to latent space
and decoding latents back to images. This enables training diffusion
models in latent space (much faster than pixel space).

Uses the Stable Diffusion VAE: stabilityai/sd-vae-ft-mse
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
import sys

# Import configuration
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import VAE_CONFIG, DATASET_CACHE_DIR


class VAEWrapper(nn.Module):
    """
    Wrapper for pretrained Stable Diffusion VAE.
    
    The VAE is frozen during training - only the U-Net learns to denoise
    in latent space. This provides a huge speedup compared to pixel-space
    diffusion.
    
    Key features:
    - 8x spatial compression (256×256 -> 32×32)
    - 4 latent channels
    - Pretrained on large-scale image datasets
    - Frozen weights (requires_grad=False)
    
    Usage:
        from models.vae_wrapper import VAEWrapper
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = VAEWrapper().to(device)
        
        # Encode images to latents
        images = torch.randn(4, 3, 256, 256).to(device)  # Range [-1, 1]
        latents = vae.encode(images)  # Shape: (4, 4, 32, 32)
        
        # Decode latents to images
        reconstructed = vae.decode(latents)  # Shape: (4, 3, 256, 256)
    """
    
    def __init__(
        self,
        pretrained_model: str = None,
        scale_factor: float = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        # Load config defaults
        self.pretrained_model = pretrained_model or VAE_CONFIG["pretrained_model"]
        self.scale_factor = scale_factor or VAE_CONFIG["scale_factor"]
        self._device = device
        
        # Load pretrained VAE from HuggingFace
        print(f"Loading VAE from: {self.pretrained_model}")
        self.vae = self._load_pretrained(self.pretrained_model)
        
        # Move to device if specified
        if device is not None:
            self.vae = self.vae.to(device)
        
        # Freeze VAE parameters
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        print(f"VAE loaded and frozen:")
        print(f"  - Latent channels: {self.latent_channels}")
        print(f"  - Downsample factor: {self.downsample_factor}x")
        print(f"  - Scale factor: {self.scale_factor}")
    
    def _load_pretrained(self, pretrained: str):
        """Load pretrained VAE from HuggingFace."""
        from diffusers import AutoencoderKL
        
        vae = AutoencoderKL.from_pretrained(
            pretrained,
            cache_dir=str(DATASET_CACHE_DIR / "huggingface"),
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
            return_dict: If True, return full posterior distribution
        
        Returns:
            Latent tensor [B, 4, H/8, W/8] (scaled by scale_factor)
        """
        with torch.no_grad():
            posterior = self.vae.encode(images).latent_dist
            
            if return_dict:
                return posterior
            
            # Sample from posterior and scale
            latents = posterior.sample()
            latents = latents * self.scale_factor
            
            return latents
    
    def encode_deterministic(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images using the mean of the posterior (deterministic).
        
        Useful for evaluation where we want consistent latents.
        
        Args:
            images: Image tensor [B, 3, H, W] in range [-1, 1]
        
        Returns:
            Latent tensor [B, 4, H/8, W/8] (scaled by scale_factor)
        """
        with torch.no_grad():
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.mode()  # Use mode instead of sampling
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
            # Unscale latents
            latents = latents / self.scale_factor
            
            # Decode
            images = self.vae.decode(latents).sample
            
            return images
    
    @property
    def latent_channels(self) -> int:
        """Number of latent channels (always 4 for SD VAE)."""
        return VAE_CONFIG["latent_channels"]
    
    @property
    def downsample_factor(self) -> int:
        """Spatial downsampling factor (always 8 for SD VAE)."""
        return VAE_CONFIG["downsample_factor"]
    
    def get_latent_size(self, image_size: int) -> int:
        """
        Calculate latent spatial size for a given image size.
        
        Args:
            image_size: Image size (H or W, assuming square)
        
        Returns:
            Latent size (H/8 or W/8)
        """
        return image_size // self.downsample_factor
    
    def to(self, device: torch.device):
        """Move VAE to device."""
        super().to(device)
        self.vae = self.vae.to(device)
        self._device = device
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode (for testing reconstruction).
        
        Args:
            x: Image tensor [B, 3, H, W]
        
        Returns:
            Reconstructed image tensor [B, 3, H, W]
        """
        latents = self.encode(x)
        return self.decode(latents)


def load_vae(
    pretrained_model: str = None,
    device: Optional[torch.device] = None,
) -> VAEWrapper:
    """
    Convenience function to load a pretrained VAE.
    
    Args:
        pretrained_model: HuggingFace model ID (uses config default if None)
        device: Device to load model to
    
    Returns:
        VAEWrapper instance
    """
    vae = VAEWrapper(pretrained_model=pretrained_model, device=device)
    
    if device is not None:
        vae = vae.to(device)
    
    return vae


def test_vae():
    """Test VAE encoding and decoding."""
    print("Testing VAE...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(device=device)
    
    # Test with random images
    batch_size = 4
    image_size = 256
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    images = torch.clamp(images, -1, 1)  # Ensure in range [-1, 1]
    
    print(f"\nInput images shape: {images.shape}")
    
    # Encode
    latents = vae.encode(images)
    print(f"Latents shape: {latents.shape}")
    
    # Decode
    reconstructed = vae.decode(latents)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Check reconstruction error
    error = (images - reconstructed).abs().mean().item()
    print(f"Mean absolute error: {error:.4f}")
    
    print("\n✓ VAE test passed!")


if __name__ == "__main__":
    test_vae()
