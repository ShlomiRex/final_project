"""
Embedding layers for U-Net.

Implements:
- Sinusoidal time embeddings
- Learned embeddings
- Position embeddings
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.
    
    Based on "Attention Is All You Need" positional encoding,
    adapted for diffusion timesteps.
    """
    
    def __init__(self, dim: int, max_positions: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Timestep tensor [B] or [B, 1]
        
        Returns:
            Embeddings [B, dim]
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        
        half_dim = self.dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')
        
        return emb


class TimestepEmbedding(nn.Module):
    """
    Full timestep embedding with MLP projection.
    
    Combines sinusoidal embeddings with learned projections.
    """
    
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        
        out_dim = out_dim or time_embed_dim
        
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim)
    
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample: Sinusoidal embeddings [B, in_channels]
        
        Returns:
            Projected embeddings [B, out_dim]
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    """
    Wrapper for timestep encoding.
    
    Combines sinusoidal positional encoding with optional learned projection.
    """
    
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Timestep tensor [B]
        
        Returns:
            Embeddings [B, num_channels]
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Handle odd dimensions
        if self.num_channels % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        
        return emb


class LabelEmbedding(nn.Module):
    """
    Embedding layer for discrete labels (e.g., class conditioning).
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.dropout_prob = dropout_prob
        
        # Null embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.randn(embed_dim))
    
    def forward(
        self,
        labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            labels: Class labels [B]
            force_drop_ids: Boolean tensor indicating which labels to drop [B]
        
        Returns:
            Embeddings [B, embed_dim]
        """
        embeddings = self.embedding(labels)
        
        # Apply dropout for classifier-free guidance during training
        if self.training and self.dropout_prob > 0:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)
        
        # Override with forced drops
        if force_drop_ids is not None:
            drop_ids = drop_ids | force_drop_ids
        
        # Replace dropped embeddings with null embedding
        embeddings = torch.where(
            drop_ids[:, None],
            self.null_embedding[None, :].expand(labels.shape[0], -1),
            embeddings,
        )
        
        return embeddings


class CombinedTimestepLabelEmbedding(nn.Module):
    """
    Combined embedding for timesteps and class labels.
    """
    
    def __init__(
        self,
        num_classes: int,
        time_embed_dim: int,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        self.time_proj = Timesteps(time_embed_dim)
        self.time_embedding = TimestepEmbedding(time_embed_dim, time_embed_dim)
        
        self.class_embedding = LabelEmbedding(
            num_classes, time_embed_dim, class_dropout_prob
        )
    
    def forward(
        self,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            timestep: Timesteps [B]
            class_labels: Class labels [B]
            force_drop_ids: Boolean tensor for CFG [B]
        
        Returns:
            Combined embeddings [B, time_embed_dim]
        """
        time_emb = self.time_proj(timestep)
        time_emb = self.time_embedding(time_emb)
        
        class_emb = self.class_embedding(class_labels, force_drop_ids)
        
        return time_emb + class_emb
