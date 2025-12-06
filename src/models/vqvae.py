"""
VQ-VAE Wrapper Module

Provides a unified interface for both pretrained VQ-GAN models and custom-trained VQ-VAE.
Supports multiple backends: HuggingFace, taming-transformers, and custom implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Literal
from abc import ABC, abstractmethod


class VQVAEBase(ABC, nn.Module):
    """Abstract base class for VQ-VAE models."""
    
    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        """
        Encode images to discrete token indices.
        
        Args:
            images: Input images [B, C, H, W] in range [-1, 1]
            
        Returns:
            Token indices [B, H', W'] where H' = H / downsample_factor
        """
        pass
    
    @abstractmethod
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Decode token indices back to images.
        
        Args:
            tokens: Token indices [B, H', W']
            
        Returns:
            Reconstructed images [B, C, H, W] in range [-1, 1]
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the codebook size."""
        pass
    
    @property
    @abstractmethod
    def downsample_factor(self) -> int:
        """Return the spatial downsampling factor."""
        pass


class VQVAEWrapper(VQVAEBase):
    """
    Unified wrapper for VQ-VAE models.
    
    Supports:
    - Pretrained VQGAN from HuggingFace (dalle-mini/vqgan_imagenet_f16_16384)
    - Pretrained VQGAN from taming-transformers
    - Custom trained VQ-VAE models
    
    Example:
        >>> vqvae = VQVAEWrapper.from_pretrained("dalle-mini/vqgan_imagenet_f16_16384")
        >>> tokens = vqvae.encode(images)  # [B, 16, 16] for 256x256 images
        >>> recon = vqvae.decode(tokens)   # [B, 3, 256, 256]
    """
    
    def __init__(
        self,
        model: nn.Module,
        codebook_size: int,
        downsample_f: int,
        source: Literal["pretrained_hf", "pretrained_taming", "custom"],
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self._vocab_size = codebook_size
        self._downsample_factor = downsample_f
        self.source = source
        self.checkpoint_path = checkpoint_path
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
    ) -> "VQVAEWrapper":
        """
        Load a pretrained VQ-GAN model.
        
        Args:
            model_name_or_path: HuggingFace model ID or path to checkpoint
            device: Device to load model on
            
        Returns:
            VQVAEWrapper instance
            
        Supported models:
            - "dalle-mini/vqgan_imagenet_f16_16384" (16384 codes, f=16)
            - "flax-community/vqgan_f16_16384" (16384 codes, f=16)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # HuggingFace VQGAN
        if "vqgan" in model_name_or_path.lower():
            from src.models._vqgan_hf import load_vqgan_from_hf
            model, codebook_size, downsample_f = load_vqgan_from_hf(model_name_or_path)
            model = model.to(device)
            return cls(
                model=model,
                codebook_size=codebook_size,
                downsample_f=downsample_f,
                source="pretrained_hf",
                checkpoint_path=model_name_or_path,
            )
        else:
            raise ValueError(f"Unknown model: {model_name_or_path}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> "VQVAEWrapper":
        """
        Load a custom-trained VQ-VAE from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration dict
            device: Device to load model on
            
        Returns:
            VQVAEWrapper instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from src.models._vqvae_custom import CustomVQVAE
        
        model = CustomVQVAE(**config)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        return cls(
            model=model,
            codebook_size=config["codebook_size"],
            downsample_f=config["downsample_factor"],
            source="custom",
            checkpoint_path=checkpoint_path,
        )
    
    def encode(self, images: torch.Tensor) -> torch.LongTensor:
        """Encode images to discrete token indices."""
        return self.model.encode(images)
    
    def decode(self, tokens: torch.LongTensor) -> torch.Tensor:
        """Decode token indices back to images."""
        return self.model.decode(tokens)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def downsample_factor(self) -> int:
        return self._downsample_factor
    
    def get_mlflow_params(self) -> dict:
        """Return parameters for MLflow logging."""
        return {
            "vqvae_source": self.source,
            "vqvae_checkpoint": self.checkpoint_path or "N/A",
            "codebook_size": self._vocab_size,
            "downsample_factor": self._downsample_factor,
        }
    
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        Full forward pass: encode and decode.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Tuple of (reconstructed images, token indices)
        """
        tokens = self.encode(images)
        recon = self.decode(tokens)
        return recon, tokens
