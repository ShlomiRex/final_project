"""
Autoregressive Latent Transformer (LatentGPT)

A GPT-2 style transformer that generates VQ-VAE latent tokens autoregressively,
with optional text conditioning via cross-attention to CLIP embeddings.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for sequence positions."""
    
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input."""
        return x + self.pe[:, :x.size(1)]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional cross-attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.cross_attention = cross_attention
        
        # Self-attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Cross-attention projections (if enabled)
        if cross_attention:
            cross_dim = cross_attention_dim or hidden_size
            self.cross_q_proj = nn.Linear(hidden_size, hidden_size)
            self.cross_k_proj = nn.Linear(cross_dim, hidden_size)
            self.cross_v_proj = nn.Linear(cross_dim, hidden_size)
            self.cross_out_proj = nn.Linear(hidden_size, hidden_size)
            self.cross_ln = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        B, H, T, D = q.shape
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        
        if causal:
            # Create causal mask
            mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        return torch.matmul(attn, v)
    
    def forward(
        self,
        x: torch.Tensor,
        cross_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with self-attention and optional cross-attention.
        
        Args:
            x: Input tensor [B, T, D]
            cross_context: Optional context for cross-attention [B, S, D_cross]
            
        Returns:
            Output tensor [B, T, D]
        """
        B, T, D = x.shape
        
        # Self-attention
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        out = self._attention(q, k, v, causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        # Cross-attention (if context provided and layer supports it)
        if self.cross_attention and cross_context is not None:
            x_norm = self.cross_ln(x + out)
            
            S = cross_context.size(1)
            cq = self.cross_q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            ck = self.cross_k_proj(cross_context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            cv = self.cross_v_proj(cross_context).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            
            cross_out = self._attention(cq, ck, cv, causal=False)
            cross_out = cross_out.transpose(1, 2).contiguous().view(B, T, D)
            out = out + self.cross_out_proj(cross_out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention, optional cross-attention, and FFN."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            cross_attention=cross_attention,
            cross_attention_dim=cross_attention_dim,
        )
        
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cross_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block."""
        x = x + self.attn(self.ln1(x), cross_context)
        x = x + self.ffn(self.ln2(x))
        return x


class LatentGPT(nn.Module):
    """
    Autoregressive transformer for VQ-VAE latent token generation.
    
    Generates discrete tokens autoregressively, optionally conditioned on
    text embeddings via cross-attention.
    
    Args:
        vocab_size: Size of VQ-VAE codebook
        hidden_size: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length (H' * W' for latent grid)
        ffn_dim: FFN intermediate dimension (default: 4 * hidden_size)
        dropout: Dropout rate
        cross_attention_dim: Dimension of cross-attention context (e.g., CLIP hidden size)
        use_cross_attention: Whether to use cross-attention for conditioning
        
    Example:
        >>> model = LatentGPT(
        ...     vocab_size=16384,
        ...     hidden_size=1024,
        ...     num_layers=24,
        ...     num_heads=16,
        ...     max_seq_len=256,  # 16x16 latent grid
        ...     cross_attention_dim=512,  # CLIP hidden size
        ... )
        >>> 
        >>> # Training: predict next token
        >>> tokens = torch.randint(0, 16384, (B, 256))
        >>> text_emb = clip_encoder(text)  # [B, 77, 512]
        >>> logits = model(tokens, text_emb)  # [B, 256, 16384]
        >>> 
        >>> # Inference: autoregressive generation
        >>> generated = model.generate(text_emb, max_len=256)
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        max_seq_len: int = 256,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        cross_attention_dim: Optional[int] = None,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_cross_attention = use_cross_attention
        
        if ffn_dim is None:
            ffn_dim = 4 * hidden_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_size)  # +1 for BOS token
        self.bos_token_id = vocab_size  # Use last index as BOS
        
        # Positional embedding
        self.pos_embedding = SinusoidalPositionalEmbedding(hidden_size, max_seq_len + 1)
        
        # Null embedding for unconditional generation (CFG)
        if use_cross_attention:
            self.null_embedding = nn.Parameter(torch.randn(1, 1, cross_attention_dim or hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                cross_attention=use_cross_attention,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.LongTensor,
        cross_context: Optional[torch.Tensor] = None,
        use_null_context: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            tokens: Input token indices [B, T]
            cross_context: Context for cross-attention [B, S, D] (e.g., CLIP embeddings)
            use_null_context: If True, use null embedding for unconditional generation
            
        Returns:
            Logits [B, T, vocab_size]
        """
        B, T = tokens.shape
        
        # Token + positional embeddings
        x = self.token_embedding(tokens)
        x = self.pos_embedding(x)
        
        # Handle conditioning
        if self.use_cross_attention:
            if use_null_context or cross_context is None:
                cross_context = self.null_embedding.expand(B, -1, -1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, cross_context)
        
        # Output
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        cross_context: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg_scale: float = 1.0,
    ) -> torch.LongTensor:
        """
        Autoregressive token generation.
        
        Args:
            cross_context: Context for cross-attention [B, S, D]
            max_len: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Generated token indices [B, max_len]
        """
        if max_len is None:
            max_len = self.max_seq_len
        
        B = cross_context.size(0) if cross_context is not None else 1
        device = next(self.parameters()).device
        
        # Start with BOS token
        tokens = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Get logits for last position
            logits = self.forward(tokens, cross_context)[:, -1, :]
            
            # CFG: combine conditional and unconditional predictions
            if cfg_scale > 1.0 and cross_context is not None:
                uncond_logits = self.forward(tokens, use_null_context=True)[:, -1, :]
                logits = uncond_logits + cfg_scale * (logits - uncond_logits)
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            tokens = torch.cat([tokens, next_token], dim=1)
        
        # Remove BOS token
        return tokens[:, 1:]
    
    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_mlflow_params(self) -> dict:
        """Return parameters for MLflow logging."""
        return {
            "transformer_layers": self.num_layers,
            "transformer_hidden": self.hidden_size,
            "transformer_heads": self.blocks[0].attn.num_heads,
            "transformer_params": f"{self.get_num_params() / 1e6:.1f}M",
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "use_cross_attention": self.use_cross_attention,
        }
