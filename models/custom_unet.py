"""
Custom UNet Model for MNIST Text-to-Image Generation

This module contains the custom UNet2DConditionModel configured for MNIST:
- 28x28 grayscale images
- Cross-attention for text conditioning (CLIP embeddings)
- Reduced parameters compared to standard Stable Diffusion UNet
"""

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Import configuration
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UNET_CONFIG


class CustomUNet2DConditionModel(UNet2DConditionModel):
    """
    Custom UNet2DConditionModel optimized for MNIST text-to-image generation.
    
    Configuration:
    - sample_size: 28 (MNIST image size)
    - in_channels: 1 (grayscale)
    - out_channels: 1 (grayscale)
    - cross_attention_dim: 512 (CLIP-ViT-B/32 embedding dimension)
    - Reduced block_out_channels for faster training on simple images
    
    Usage:
        from models.custom_unet import CustomUNet2DConditionModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = CustomUNet2DConditionModel().to(device)
    """
    
    def __init__(self, **kwargs):
        # Merge default config with any overrides
        config = {**UNET_CONFIG, **kwargs}
        super().__init__(**config)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        """
        Load model from a checkpoint file.
        
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
        model.load_state_dict(checkpoint["unet_state_dict"])
        
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
        print(f"Number of trainable parameters: {num_params:,}")


def load_unet_from_latest_checkpoint(device: torch.device = None):
    """
    Convenience function to load UNet from the latest checkpoint.
    
    Args:
        device: Device to load the model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    from config import get_latest_unet_checkpoint
    
    checkpoint_path = get_latest_unet_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, checkpoint = CustomUNet2DConditionModel.from_checkpoint(
        str(checkpoint_path), device
    )
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, checkpoint
