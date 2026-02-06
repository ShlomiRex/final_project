"""
Custom UNet Model for CelebA-HQ Latent Diffusion

This module contains the custom UNet2DConditionModel configured for CelebA-HQ
latent diffusion:
- 32×32 latent space (from 256×256 images via VAE)
- 4 latent channels (VAE encoding)
- Cross-attention for text conditioning (CLIP embeddings)
- Operates in latent space for faster training
"""

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Import configuration
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UNET_CELEBA_LDM_CONFIG


class CustomUNet2DConditionModelCelebaLDM(UNet2DConditionModel):
    """
    Custom UNet2DConditionModel optimized for CelebA-HQ latent diffusion.
    
    Key differences from pixel-space UNets:
    - Operates on 32×32 latents (not 256×256 pixels)
    - 4 input/output channels (VAE latent channels)
    - Much faster training due to smaller spatial dimensions
    
    Configuration:
    - sample_size: 32 (latent size, not image size)
    - in_channels: 4 (VAE latent channels)
    - out_channels: 4 (VAE latent channels)
    - cross_attention_dim: 512 (CLIP-ViT-B/32 embedding dimension)
    - Moderate block_out_channels between CIFAR-10 and WikiArt
    
    Usage:
        from models.custom_unet_celeba_ldm import CustomUNet2DConditionModelCelebaLDM
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = CustomUNet2DConditionModelCelebaLDM().to(device)
        
        # Training: predict noise in latent space
        noisy_latents = torch.randn(4, 4, 32, 32).to(device)
        timesteps = torch.randint(0, 1000, (4,)).to(device)
        encoder_hidden_states = torch.randn(4, 77, 512).to(device)
        
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    """
    
    def __init__(self, **kwargs):
        # Merge default config with any overrides
        config = {**UNET_CELEBA_LDM_CONFIG, **kwargs}
        super().__init__(**config)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        """
        Load model from a checkpoint file.
        
        Handles different checkpoint formats:
        - Dict with "unet_state_dict" key
        - Dict with "model_state_dict" key
        - Dict with "state_dict" key
        - Raw state dict
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model to (defaults to CUDA if available)
        
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = cls().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "unet_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["unet_state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        return model, checkpoint
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def print_parameter_count(self):
        """Print formatted parameter count."""
        num_params = self.get_num_parameters()
        num_params_m = num_params / 1_000_000
        print(f"Number of trainable parameters: {num_params:,} ({num_params_m:.1f}M)")


def load_celeba_ldm_unet_from_checkpoint(checkpoint_path: str, device: torch.device = None):
    """
    Load CelebA LDM UNet from a specific checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    model, checkpoint = CustomUNet2DConditionModelCelebaLDM.from_checkpoint(
        checkpoint_path, device
    )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded CelebA LDM UNet from epoch {checkpoint['epoch']}")
    else:
        print(f"Loaded CelebA LDM UNet from checkpoint")
    
    return model, checkpoint


def load_celeba_ldm_unet_from_latest_checkpoint(device: torch.device = None):
    """
    Convenience function to load CelebA LDM UNet from the latest checkpoint.
    
    Args:
        device: Device to load the model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    from config import get_latest_celeba_ldm_unet_checkpoint
    
    checkpoint_path = get_latest_celeba_ldm_unet_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, checkpoint = CustomUNet2DConditionModelCelebaLDM.from_checkpoint(
        str(checkpoint_path), device
    )
    
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, checkpoint


def test_unet():
    """Test UNet forward pass."""
    print("Testing CelebA LDM UNet...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = CustomUNet2DConditionModelCelebaLDM().to(device)
    
    # Print parameter count
    unet.print_parameter_count()
    
    # Test forward pass
    batch_size = 2
    latent_size = 32
    latent_channels = 4
    seq_length = 77
    hidden_dim = 512
    
    # Random inputs
    noisy_latents = torch.randn(batch_size, latent_channels, latent_size, latent_size).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    encoder_hidden_states = torch.randn(batch_size, seq_length, hidden_dim).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Noisy latents: {noisy_latents.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    print(f"  Encoder hidden states: {encoder_hidden_states.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = unet(noisy_latents, timesteps, encoder_hidden_states)
        noise_pred = output.sample
    
    print(f"\nOutput shape: {noise_pred.shape}")
    
    # Verify output shape matches input
    assert noise_pred.shape == noisy_latents.shape, "Output shape mismatch!"
    
    print("\n✓ UNet test passed!")


if __name__ == "__main__":
    test_unet()
