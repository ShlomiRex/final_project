"""
Custom UNet Model for WikiArt Text-to-Image Generation

This module contains the custom UNet2DConditionModel configured for WikiArt:
- 128x128 RGB images
- Cross-attention for text conditioning (CLIP embeddings)
- Larger capacity for complex artistic images and textures
"""

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# Import configuration
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UNET_WIKIART_CONFIG


class CustomUNet2DConditionModelWikiArt(UNet2DConditionModel):
    """
    Custom UNet2DConditionModel optimized for WikiArt text-to-image generation.
    
    Configuration:
    - sample_size: 128 (WikiArt image size)
    - in_channels: 3 (RGB)
    - out_channels: 3 (RGB)
    - cross_attention_dim: 512 (CLIP-ViT-B/32 embedding dimension)
    - Larger block_out_channels for complex artistic images
    
    Usage:
        from models.custom_unet_wikiart import CustomUNet2DConditionModelWikiArt
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet = CustomUNet2DConditionModelWikiArt().to(device)
    """
    
    def __init__(self, **kwargs):
        # Merge default config with any overrides
        config = {**UNET_WIKIART_CONFIG, **kwargs}
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
        
        # Handle different checkpoint formats
        if "unet_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["unet_state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the checkpoint is the state dict itself
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
        print(f"Number of trainable parameters: {num_params:,}")


def load_wikiart_unet_from_checkpoint(checkpoint_path: str, device: torch.device = None):
    """
    Load WikiArt UNet from a specific checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    model, checkpoint = CustomUNet2DConditionModelWikiArt.from_checkpoint(
        checkpoint_path, device
    )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Loaded WikiArt UNet from epoch {checkpoint['epoch']}")
    else:
        print(f"Loaded WikiArt UNet from checkpoint")
    
    return model, checkpoint


def load_wikiart_unet_from_latest_checkpoint(device: torch.device = None):
    """
    Convenience function to load WikiArt UNet from the latest checkpoint.
    
    Args:
        device: Device to load the model to
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    from config import get_latest_wikiart_unet_checkpoint
    
    checkpoint_path = get_latest_wikiart_unet_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, checkpoint = CustomUNet2DConditionModelWikiArt.from_checkpoint(
        str(checkpoint_path), device
    )
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    return model, checkpoint
