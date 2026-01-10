"""
Attention mechanisms for U-Net.

Implements:
- Self-Attention
- Cross-Attention (for text conditioning)
- Spatial Transformer blocks
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GEGLU(nn.Module):
    """GEGLU activation function."""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    """
    Cross-attention layer.
    
    Can function as self-attention (when context is None) or
    cross-attention (when context is provided).
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor [B, N, C]
            context: Context tensor for cross-attention [B, M, C'] or None for self-attention
            mask: Attention mask [B, N, M] (optional)
        
        Returns:
            Output tensor [B, N, C]
        """
        h = self.heads
        
        # Self-attention if no context provided
        if context is None:
            context = x
        
        # Compute Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b m (h d) -> b h m d', h=h)
        v = rearrange(v, 'b m (h d) -> b h m d', h=h)
        
        # Scaled dot-product attention
        # Use PyTorch's efficient attention when available
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=0.0 if not self.training else 0.0,
            )
        else:
            # Fallback implementation
            sim = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
            
            if mask is not None:
                sim = sim.masked_fill(~mask, float('-inf'))
            
            attn = sim.softmax(dim=-1)
            out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block with self-attention, cross-attention, and FFN.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Self-attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        # Cross-attention
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C]
            context: Context for cross-attention [B, M, C'] (e.g., text embeddings)
        
        Returns:
            Output tensor [B, N, C]
        """
        # Self-attention
        x = x + self.attn1(self.norm1(x))
        
        # Cross-attention
        x = x + self.attn2(self.norm2(x), context=context)
        
        # Feed-forward
        x = x + self.ff(self.norm3(x))
        
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for spatial feature maps.
    
    Reshapes 2D feature maps to sequence, applies transformer blocks,
    then reshapes back to 2D.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        dim_head: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        inner_dim = num_heads * dim_head
        
        # Input projection
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=inner_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                context_dim=context_dim,
            )
            for _ in range(depth)
        ])
        
        # Output projection
        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
            context: Context for cross-attention [B, M, C']
        
        Returns:
            Output feature map [B, C, H, W]
        """
        b, c, h, w = x.shape
        residual = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Reshape to sequence
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        # Reshape back to 2D
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # Project out and add residual
        x = self.proj_out(x)
        x = x + residual
        
        return x


class AttentionBlock(nn.Module):
    """
    Simple self-attention block for feature maps.
    Used when cross-attention is not needed.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_head_channels: int = -1,
    ):
        super().__init__()
        
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = channels // num_head_channels
        
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            Output feature map [B, C, H, W]
        """
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(b, c, -1)  # [B, C, H*W]
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        scale = 1.0 / math.sqrt(c // self.num_heads)
        q = q.view(b, self.num_heads, c // self.num_heads, -1)
        k = k.view(b, self.num_heads, c // self.num_heads, -1)
        v = v.view(b, self.num_heads, c // self.num_heads, -1)
        
        # Attention
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        out = out.view(b, c, -1)
        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        
        return out + residual
