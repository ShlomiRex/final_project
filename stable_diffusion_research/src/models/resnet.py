"""
ResNet blocks for U-Net.

Implements:
- ResBlock with time embedding
- Downsampling blocks
- Upsampling blocks
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 precision for stability."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer with optional convolution."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ResBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    
    Architecture:
        x -> GroupNorm -> SiLU -> Conv -> + time_emb -> GroupNorm -> SiLU -> Dropout -> Conv -> + residual
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        
        # First conv block
        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Up/down sampling
        if up:
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # Second conv block
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            time_emb: Time embedding [B, D]
        
        Returns:
            Output tensor [B, C', H', W']
        """
        # Handle up/down sampling
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]
        
        # Second conv block
        h = self.out_layers(h)
        
        # Skip connection
        return h + self.skip_connection(x)


class ResBlockConditioned(nn.Module):
    """
    ResBlock with additional conditioning (e.g., class labels).
    Uses adaptive group normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cond = cond_dim is not None
        
        # First block
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        
        # Condition embedding (for adaptive norm)
        if self.use_cond:
            self.cond_emb_proj = nn.Linear(cond_dim, out_channels * 2)
        
        # Second block
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            time_emb: Time embedding [B, D]
            cond: Optional condition embedding [B, D']
        
        Returns:
            Output tensor [B, C', H, W]
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time conditioning (scale and shift)
        time_params = self.time_emb_proj(time_emb)
        scale, shift = time_params.chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        # Additional conditioning
        if self.use_cond and cond is not None:
            cond_params = self.cond_emb_proj(cond)
            cond_scale, cond_shift = cond_params.chunk(2, dim=-1)
            h = h * (1 + cond_scale[:, :, None, None]) + cond_shift[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)
