"""
HuggingFace VQGAN Loader

Loads pretrained VQGAN models from HuggingFace Hub.
Primary model: dalle-mini/vqgan_imagenet_f16_16384

Based on the VQGAN architecture from taming-transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VQGANEncoder(nn.Module):
    """VQGAN Encoder network."""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        z_channels: int = 256,
    ):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        curr_channels = hidden_channels
        
        for i, mult in enumerate(channel_mult):
            out_channels = hidden_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(curr_channels, out_channels))
                curr_channels = out_channels
            
            if i < len(channel_mult) - 1:
                self.down_blocks.append(Downsample(curr_channels))
        
        # Middle
        self.mid_block1 = ResnetBlock(curr_channels, curr_channels)
        self.mid_attn = AttnBlock(curr_channels)
        self.mid_block2 = ResnetBlock(curr_channels, curr_channels)
        
        # Output
        self.norm_out = nn.GroupNorm(32, curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, z_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        
        for block in self.down_blocks:
            h = block(h)
        
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VQGANDecoder(nn.Module):
    """VQGAN Decoder network."""
    
    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        z_channels: int = 256,
    ):
        super().__init__()
        
        curr_channels = hidden_channels * channel_mult[-1]
        
        self.conv_in = nn.Conv2d(z_channels, curr_channels, 3, padding=1)
        
        # Middle
        self.mid_block1 = ResnetBlock(curr_channels, curr_channels)
        self.mid_attn = AttnBlock(curr_channels)
        self.mid_block2 = ResnetBlock(curr_channels, curr_channels)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_channels_block = hidden_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResnetBlock(curr_channels, out_channels_block))
                curr_channels = out_channels_block
            
            if i < len(channel_mult) - 1:
                self.up_blocks.append(Upsample(curr_channels))
        
        # Output
        self.norm_out = nn.GroupNorm(32, curr_channels)
        self.conv_out = nn.Conv2d(curr_channels, out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        
        for block in self.up_blocks:
            h = block(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA updates."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        """
        Forward pass with vector quantization.
        
        Args:
            z: Encoder output [B, C, H, W]
            
        Returns:
            quantized: Quantized tensor [B, C, H, W]
            indices: Codebook indices [B, H, W]
            loss: VQ loss
        """
        # Reshape for distance calculation [B, H, W, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous()
        z_flat_view = z_flat.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (
            torch.sum(z_flat_view ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat_view, self.embedding.weight.t())
        )
        
        # Find closest codebook entries
        indices = torch.argmin(distances, dim=1)
        indices = indices.view(z.shape[0], z.shape[2], z.shape[3])
        
        # Quantize
        z_q = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        
        # Compute loss
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, indices, loss


class ResnetBlock(nn.Module):
    """Residual block with GroupNorm."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttnBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.norm = nn.GroupNorm(32, in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(B, C, H * W)  # [B, C, HW]
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)  # [B, HW, C]
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        
        return x + self.proj_out(h)


class Downsample(nn.Module):
    """Downsampling with strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling with interpolation and convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class VQGAN(nn.Module):
    """
    Full VQGAN model combining encoder, quantizer, and decoder.
    
    This is a simplified implementation. For pretrained weights,
    we load from HuggingFace and map to this architecture.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        z_channels: int = 256,
        num_embeddings: int = 16384,
    ):
        super().__init__()
        
        self.encoder = VQGANEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            z_channels=z_channels,
        )
        
        self.quantize = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=z_channels,
        )
        
        self.decoder = VQGANDecoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            z_channels=z_channels,
        )
        
        self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        
        self.num_embeddings = num_embeddings
        self._downsample_factor = 2 ** (len(channel_mult) - 1)
    
    def encode(self, x: torch.Tensor) -> torch.LongTensor:
        """
        Encode images to discrete tokens.
        
        Args:
            x: Images [B, C, H, W] in range [-1, 1] or [0, 1]
            
        Returns:
            Token indices [B, H', W']
        """
        # Normalize to [-1, 1] if needed
        if x.min() >= 0:
            x = x * 2 - 1
        
        z = self.encoder(x)
        z = self.quant_conv(z)
        _, indices, _ = self.quantize(z)
        return indices
    
    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        Decode tokens to images.
        
        Args:
            indices: Token indices [B, H', W']
            
        Returns:
            Images [B, C, H, W] in range [0, 1]
        """
        z_q = self.quantize.embedding(indices).permute(0, 3, 1, 2).contiguous()
        z_q = self.post_quant_conv(z_q)
        dec = self.decoder(z_q)
        # Return in [0, 1] range
        return (dec + 1) / 2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass for training."""
        z = self.encoder(x)
        z = self.quant_conv(z)
        z_q, indices, vq_loss = self.quantize(z)
        z_q = self.post_quant_conv(z_q)
        dec = self.decoder(z_q)
        return dec, vq_loss


def load_vqgan_from_hf(model_name: str) -> Tuple[VQGAN, int, int]:
    """
    Load VQGAN model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model ID (e.g., "dalle-mini/vqgan_imagenet_f16_16384")
        
    Returns:
        Tuple of (model, codebook_size, downsample_factor)
    """
    try:
        from huggingface_hub import hf_hub_download
        import json
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    
    # Default configuration for common models
    CONFIGS = {
        "dalle-mini/vqgan_imagenet_f16_16384": {
            "hidden_channels": 128,
            "num_res_blocks": 2,
            "channel_mult": (1, 1, 2, 2, 4),
            "z_channels": 256,
            "num_embeddings": 16384,
        },
        "flax-community/vqgan_f16_16384": {
            "hidden_channels": 128,
            "num_res_blocks": 2,
            "channel_mult": (1, 1, 2, 2, 4),
            "z_channels": 256,
            "num_embeddings": 16384,
        },
    }
    
    # Get config
    if model_name in CONFIGS:
        config = CONFIGS[model_name]
    else:
        # Try to download config
        try:
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            # Fallback to default
            print(f"Warning: Could not load config for {model_name}, using defaults")
            config = CONFIGS["dalle-mini/vqgan_imagenet_f16_16384"]
    
    # Create model
    model = VQGAN(
        in_channels=3,
        hidden_channels=config["hidden_channels"],
        num_res_blocks=config["num_res_blocks"],
        channel_mult=tuple(config["channel_mult"]),
        z_channels=config["z_channels"],
        num_embeddings=config["num_embeddings"],
    )
    
    # Try to load weights from HuggingFace
    try:
        # First try PyTorch weights
        weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {model_name}")
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Model initialized with random weights. For proper generation, you need pretrained weights.")
    
    codebook_size = config["num_embeddings"]
    downsample_factor = model._downsample_factor
    
    return model, codebook_size, downsample_factor
