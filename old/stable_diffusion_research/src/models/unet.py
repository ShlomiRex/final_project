"""
U-Net 2D Conditional Model for Stable Diffusion.

A U-Net architecture with cross-attention for text conditioning,
designed for latent diffusion models.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import AttentionBlock, SpatialTransformer
from .embeddings import TimestepEmbedding, Timesteps
from .resnet import Downsample, GroupNorm32, ResBlock, Upsample


@dataclass
class UNetConfig:
    """Configuration for U-Net model."""
    
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)
    num_heads: int = 8
    num_head_channels: int = 64
    use_spatial_transformer: bool = True
    transformer_depth: int = 1
    context_dim: int = 768
    use_checkpoint: bool = True
    dropout: float = 0.0


@dataclass
class UNet2DConditionOutput:
    """Output of U-Net."""
    sample: torch.Tensor


class DownBlock(nn.Module):
    """Downsampling block with ResBlocks and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        add_downsample: bool = True,
        has_attention: bool = False,
        num_heads: int = 8,
        num_head_channels: int = 64,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.use_checkpoint = use_checkpoint
        self.has_attention = has_attention
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            
            self.resnets.append(
                ResBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
            )
            
            if has_attention:
                if use_spatial_transformer:
                    self.attentions.append(
                        SpatialTransformer(
                            in_channels=out_channels,
                            num_heads=num_heads,
                            dim_head=num_head_channels,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            dropout=dropout,
                        )
                    )
                else:
                    self.attentions.append(
                        AttentionBlock(
                            channels=out_channels,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
        
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample(out_channels, use_conv=True)
            ])
        else:
            self.downsamplers = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            hidden_states: Input tensor [B, C, H, W]
            temb: Time embedding [B, D]
            encoder_hidden_states: Text embeddings [B, L, D] for cross-attention
        
        Returns:
            Tuple of (output tensor, list of intermediate outputs for skip connections)
        """
        output_states = []
        
        for i, resnet in enumerate(self.resnets):
            if self.use_checkpoint and self.training:
                hidden_states = checkpoint(
                    resnet, hidden_states, temb, use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            
            if self.has_attention:
                attn = self.attentions[i]
                if self.use_checkpoint and self.training:
                    hidden_states = checkpoint(
                        attn, hidden_states, encoder_hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = attn(hidden_states, encoder_hidden_states)
            
            output_states.append(hidden_states)
        
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)
        
        return hidden_states, output_states


class UpBlock(nn.Module):
    """Upsampling block with ResBlocks and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        add_upsample: bool = True,
        has_attention: bool = False,
        num_heads: int = 8,
        num_head_channels: int = 64,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.use_checkpoint = use_checkpoint
        self.has_attention = has_attention
        
        for i in range(num_res_blocks):
            res_skip_channels = in_channels if i == num_res_blocks - 1 else out_channels
            resnet_in_channels = prev_out_channels if i == 0 else out_channels
            
            self.resnets.append(
                ResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
            )
            
            if has_attention:
                if use_spatial_transformer:
                    self.attentions.append(
                        SpatialTransformer(
                            in_channels=out_channels,
                            num_heads=num_heads,
                            dim_head=num_head_channels,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            dropout=dropout,
                        )
                    )
                else:
                    self.attentions.append(
                        AttentionBlock(
                            channels=out_channels,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
        
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample(out_channels, use_conv=True)
            ])
        else:
            self.upsamplers = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [B, C, H, W]
            res_hidden_states_tuple: Skip connection tensors from encoder
            temb: Time embedding [B, D]
            encoder_hidden_states: Text embeddings [B, L, D]
        
        Returns:
            Output tensor [B, C', H', W']
        """
        for i, resnet in enumerate(self.resnets):
            # Pop skip connection
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            # Concatenate skip connection
            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            
            if self.use_checkpoint and self.training:
                hidden_states = checkpoint(
                    resnet, hidden_states, temb, use_reentrant=False
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            
            if self.has_attention:
                attn = self.attentions[i]
                if self.use_checkpoint and self.training:
                    hidden_states = checkpoint(
                        attn, hidden_states, encoder_hidden_states, use_reentrant=False
                    )
                else:
                    hidden_states = attn(hidden_states, encoder_hidden_states)
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        
        return hidden_states


class MidBlock(nn.Module):
    """Middle block with ResBlocks and attention."""
    
    def __init__(
        self,
        in_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        num_heads: int = 8,
        num_head_channels: int = 64,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.resnet_1 = ResBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        
        if use_spatial_transformer:
            self.attention = SpatialTransformer(
                in_channels=in_channels,
                num_heads=num_heads,
                dim_head=num_head_channels,
                depth=transformer_depth,
                context_dim=context_dim,
                dropout=dropout,
            )
        else:
            self.attention = AttentionBlock(
                channels=in_channels,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            )
        
        self.resnet_2 = ResBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [B, C, H, W]
            temb: Time embedding [B, D]
            encoder_hidden_states: Text embeddings [B, L, D]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        if self.use_checkpoint and self.training:
            hidden_states = checkpoint(
                self.resnet_1, hidden_states, temb, use_reentrant=False
            )
            hidden_states = checkpoint(
                self.attention, hidden_states, encoder_hidden_states, use_reentrant=False
            )
            hidden_states = checkpoint(
                self.resnet_2, hidden_states, temb, use_reentrant=False
            )
        else:
            hidden_states = self.resnet_1(hidden_states, temb)
            hidden_states = self.attention(hidden_states, encoder_hidden_states)
            hidden_states = self.resnet_2(hidden_states, temb)
        
        return hidden_states


class UNet2DConditionModel(nn.Module):
    """
    U-Net with cross-attention for text-conditioned image generation.
    
    Architecture follows Stable Diffusion 1.x design:
    - Encoder: 4 resolution levels with downsampling
    - Bottleneck: Transformer blocks
    - Decoder: 4 resolution levels with upsampling and skip connections
    
    Conditioning:
    - Time: Sinusoidal embeddings + MLP
    - Text: Cross-attention to CLIP embeddings
    """
    
    def __init__(self, config: Union[UNetConfig, dict]):
        super().__init__()
        
        if isinstance(config, dict):
            config = UNetConfig(**config)
        
        self.config = config
        
        # Determine number of heads
        if config.num_head_channels != -1:
            num_heads = config.model_channels // config.num_head_channels
        else:
            num_heads = config.num_heads
        
        # Time embedding
        time_embed_dim = config.model_channels * 4
        
        self.time_proj = Timesteps(config.model_channels)
        self.time_embedding = TimestepEmbedding(
            config.model_channels, time_embed_dim
        )
        
        # Input convolution
        self.conv_in = nn.Conv2d(
            config.in_channels, config.model_channels,
            kernel_size=3, padding=1
        )
        
        # Encoder (down blocks)
        self.down_blocks = nn.ModuleList()
        
        output_channels = config.model_channels
        for i, mult in enumerate(config.channel_mult):
            input_channels = output_channels
            output_channels = config.model_channels * mult
            is_final = i == len(config.channel_mult) - 1
            
            # Check if this resolution should have attention
            ds = 2 ** i  # Downsampling factor
            has_attention = ds in config.attention_resolutions
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    time_emb_dim=time_embed_dim,
                    num_res_blocks=config.num_res_blocks,
                    dropout=config.dropout,
                    add_downsample=not is_final,
                    has_attention=has_attention,
                    num_heads=num_heads,
                    num_head_channels=config.num_head_channels,
                    use_spatial_transformer=config.use_spatial_transformer,
                    transformer_depth=config.transformer_depth,
                    context_dim=config.context_dim,
                    use_checkpoint=config.use_checkpoint,
                )
            )
        
        # Middle block
        self.mid_block = MidBlock(
            in_channels=output_channels,
            time_emb_dim=time_embed_dim,
            dropout=config.dropout,
            num_heads=num_heads,
            num_head_channels=config.num_head_channels,
            use_spatial_transformer=config.use_spatial_transformer,
            transformer_depth=config.transformer_depth,
            context_dim=config.context_dim,
            use_checkpoint=config.use_checkpoint,
        )
        
        # Decoder (up blocks)
        self.up_blocks = nn.ModuleList()
        
        reversed_mult = list(reversed(config.channel_mult))
        output_channels = config.model_channels * reversed_mult[0]
        
        for i, mult in enumerate(reversed_mult):
            prev_output_channels = output_channels
            output_channels = config.model_channels * mult
            input_channels = config.model_channels * (
                reversed_mult[min(i + 1, len(reversed_mult) - 1)]
            )
            
            is_final = i == len(reversed_mult) - 1
            
            # Check if this resolution should have attention
            ds = 2 ** (len(reversed_mult) - 1 - i)
            has_attention = ds in config.attention_resolutions
            
            self.up_blocks.append(
                UpBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    prev_out_channels=prev_output_channels,
                    time_emb_dim=time_embed_dim,
                    num_res_blocks=config.num_res_blocks + 1,
                    dropout=config.dropout,
                    add_upsample=not is_final,
                    has_attention=has_attention,
                    num_heads=num_heads,
                    num_head_channels=config.num_head_channels,
                    use_spatial_transformer=config.use_spatial_transformer,
                    transformer_depth=config.transformer_depth,
                    context_dim=config.context_dim,
                    use_checkpoint=config.use_checkpoint,
                )
            )
        
        # Output
        self.conv_norm_out = GroupNorm32(32, config.model_channels)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            config.model_channels, config.out_channels,
            kernel_size=3, padding=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Zero out output conv for residual connection
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, torch.Tensor]:
        """
        Forward pass of U-Net.
        
        Args:
            sample: Noisy latent tensor [B, 4, H, W]
            timestep: Diffusion timesteps [B] or scalar
            encoder_hidden_states: CLIP text embeddings [B, L, D]
            return_dict: Whether to return UNet2DConditionOutput
        
        Returns:
            Predicted noise tensor [B, 4, H, W]
        """
        # Handle scalar timesteps
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)
        
        # Broadcast to batch dimension
        if timestep.shape[0] == 1:
            timestep = timestep.expand(sample.shape[0])
        
        # Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)
        
        # Input
        hidden_states = self.conv_in(sample)
        
        # Encoder
        down_block_res_samples = (hidden_states,)
        for down_block in self.down_blocks:
            hidden_states, res_samples = down_block(
                hidden_states, t_emb, encoder_hidden_states
            )
            down_block_res_samples += tuple(res_samples)
        
        # Middle
        hidden_states = self.mid_block(
            hidden_states, t_emb, encoder_hidden_states
        )
        
        # Decoder
        for up_block in self.up_blocks:
            # Get skip connections for this block
            res_samples = down_block_res_samples[-(self.config.num_res_blocks + 1):]
            down_block_res_samples = down_block_res_samples[:-(self.config.num_res_blocks + 1)]
            
            hidden_states = up_block(
                hidden_states, res_samples, t_emb, encoder_hidden_states
            )
        
        # Output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        output = self.conv_out(hidden_states)
        
        if not return_dict:
            return output
        
        return UNet2DConditionOutput(sample=output)
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
