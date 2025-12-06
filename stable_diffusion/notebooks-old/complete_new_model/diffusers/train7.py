"""
stable_diffusion_from_scratch_unet.py

This script demonstrates training a Stable-Diffusion-like pipeline where the UNet
is trained entirely from scratch (no pretrained UNet). Everything is self-contained
inside the script: dataset selection, hyperparameters, saving paths.

Key choices made in the script:
- Dataset: CIFAR-10 (automatically downloaded), images resized to 128x128 for faster training.
- VAE: uses a pretrained AutoencoderKL for encoding/decoding latents (training a VAE from scratch
  is possible but expensive; the user's explicit restriction was only "DO NOT USE PRETRAINED UNET MODEL").
- Text encoder & tokenizer: pretrained CLIP (from transformers) used to obtain text embeddings.
- UNet: a custom lightweight conditional UNet implemented from scratch in this file. It
  supports conditioning via simple cross-attention blocks that take CLIP text embeddings.
- Scheduler: DDPMScheduler from diffusers.
- Training: small demo training loop (single GPU / CPU if no GPU). Uses AdamW optimizer.
- Everything (dataset, checkpoints, example outputs) is written to paths defined inside the file.

WARNING / NOTES
- This is a pedagogical demo: training a full high-quality SD model from scratch requires
  large datasets, many GPUs, and weeks of training. This script is intentionally small so
  you can run it as a prototype and iterate.
- If you want *no* pretrained components at all (text encoder or VAE), tell me and I will
  adapt the script, but training will be much heavier.

Run:
    python stable_diffusion_from_scratch_unet.py

"""

import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

import matplotlib.pyplot as plt
import numpy as np

# ------------------ Configuration (all in-code, as requested) ------------------
ROOT_DIR = Path("./train7_output")
ROOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = ROOT_DIR / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ROOT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Create log file with timestamp
LOG_FILE = ROOT_DIR / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ------------------ Logger Setup ------------------
class TeeLogger:
    """Logger that writes to both console and file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

class TeeLoggerStderr:
    """Logger for stderr that writes to both console and file."""
    def __init__(self, filepath):
        self.terminal = sys.stderr
        self.log = open(filepath, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Initialize logging
stdout_logger = TeeLogger(LOG_FILE)
stderr_logger = TeeLoggerStderr(LOG_FILE)
sys.stdout = stdout_logger
sys.stderr = stderr_logger

print(f"📁 Output directories created:")
print(f"   ROOT_DIR: {ROOT_DIR.absolute()}")
print(f"   CHECKPOINT_DIR: {CHECKPOINT_DIR.absolute()}")
print(f"   EXAMPLES_DIR: {EXAMPLES_DIR.absolute()}")
print(f"   PLOTS_DIR: {PLOTS_DIR.absolute()}")
print(f"   LOG_FILE: {LOG_FILE.absolute()}")
print(f"\n{'='*80}")
print(f"📝 All console output is being logged to: {LOG_FILE.name}")
print(f"{'='*80}\n")

# ------------------ Configuration Variables ------------------
SEED = 42
IMAGE_SIZE = 32  # Keep at 32x32 as requested - results in 8x8 latents
BATCH_SIZE = 16  # Reduced batch size for larger model
NUM_EPOCHS = 100  # Increased from 10 - diffusion models need much more training
LEARNING_RATE = 1e-4
GRAD_ACCUM_STEPS = 2  # Increased gradient accumulation to maintain effective batch size
MAX_TRAIN_STEPS = 10_000_000  # Increased cap - let it train longer
SAVE_CHECKPOINT_EVERY_EPOCHS = 10  # Save checkpoint every 10 epochs
GENERATE_IMAGES_EVERY_EPOCHS = 1  # Generate images EVERY epoch to monitor progress
NUM_TRAIN_TIMESTEPS = 1000
CLASSIFIER_FREE_GUIDANCE_DROPOUT = 0.1  # Randomly drop text conditioning 10% of the time

PRETRAINED_VAE = "stabilityai/sdxl-vae"  # we will try to load a VAE from HF; fallback handled
PRETRAINED_CLIP = "openai/clip-vit-base-patch32"

# ------------------ GPU and Device Setup ------------------

def check_gpu_setup():
    """Check GPU availability and print detailed information."""
    print("=" * 80)
    print("🖥️  HARDWARE SETUP CHECK")
    print("=" * 80)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ ERROR: No CUDA-capable GPU detected!")
        print("This script requires at least 1 GPU for training.")
        print("Please ensure you have:")
        print("  - NVIDIA GPU with CUDA support")
        print("  - Proper CUDA drivers installed")
        print("  - PyTorch with CUDA support")
        exit(1)
    
    # Get GPU count and information
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    if gpu_count == 0:
        print("❌ ERROR: No GPUs detected even though CUDA is available!")
        exit(1)
    
    print(f"✅ Found {gpu_count} GPU(s) - Training can proceed!")
    print()
    
    # Print detailed GPU information
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name}")
        print(f"  - Memory: {memory_gb:.1f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Check memory usage
        if torch.cuda.is_available():
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  - Memory Used: {allocated:.2f} GB / {memory_gb:.1f} GB")
            print(f"  - Memory Cached: {cached:.2f} GB")
        print()
    
    # Print CUDA version info
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print()
    
    return gpu_count

def setup_accelerator():
    """Setup Accelerator for distributed training."""
    print("🚀 SETTING UP ACCELERATOR")
    print("=" * 80)
    
    # Configure DDP kwargs for better performance
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="fp16",  # Use mixed precision for better performance
        kwargs_handlers=[ddp_kwargs]
    )
    
    print(f"Accelerator Device: {accelerator.device}")
    print(f"Process Index: {accelerator.process_index}")
    print(f"Local Process Index: {accelerator.local_process_index}")
    print(f"Number of Processes: {accelerator.num_processes}")
    print(f"Is Main Process: {accelerator.is_main_process}")
    print(f"Is Local Main Process: {accelerator.is_local_main_process}")
    print(f"Mixed Precision: {accelerator.mixed_precision}")
    print()
    
    return accelerator

# ------------------ Utilities ------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def print_debug_info(
    global_step: int,
    images: torch.Tensor,
    latents: torch.Tensor,
    noise: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    captions: list,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_embeddings: torch.Tensor,
    pred: torch.Tensor,
    loss: torch.Tensor,
    batch_idx: int
):
    """
    Print comprehensive debugging information about tensor dimensions and values.
    
    Args:
        global_step: Current global training step
        images: Original images from dataloader
        latents: VAE-encoded latents
        noise: Random noise added
        noisy_latents: Latents with noise added
        timesteps: Diffusion timesteps
        captions: List of text captions
        input_ids: Tokenized caption IDs
        attention_mask: Attention mask for captions
        text_embeddings: CLIP text embeddings
        pred: UNet prediction
        loss: Current loss value
        batch_idx: Index of current batch in epoch
    """
    print("\n" + "="*100)
    print(f"🔍 DEBUG INFO - Global Step {global_step} (Batch {batch_idx})")
    print("="*100)
    
    # Image information
    print("\n📸 IMAGE INFORMATION:")
    print(f"   Images shape: {images.shape}")
    print(f"   Images dtype: {images.dtype}")
    print(f"   Images device: {images.device}")
    print(f"   Image size (H x W): {images.shape[2]} x {images.shape[3]}")
    print(f"   Expected image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"   Images min/max: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"   Images mean/std: [{images.mean().item():.4f}, {images.std().item():.4f}]")
    
    # VAE Latents information
    print("\n🎨 VAE LATENTS INFORMATION:")
    print(f"   Latents shape: {latents.shape}")
    print(f"   Latents dtype: {latents.dtype}")
    print(f"   Latents device: {latents.device}")
    print(f"   Latents channels: {latents.shape[1]}")
    print(f"   Latents spatial size: {latents.shape[2]}x{latents.shape[3]}")
    print(f"   Expected latent size: {IMAGE_SIZE // 4} x {IMAGE_SIZE // 4}")
    print(f"   VAE downsampling factor: {images.shape[2] / latents.shape[2]:.2f}x")
    print(f"   Latents min/max: [{latents.min().item():.4f}, {latents.max().item():.4f}]")
    print(f"   Latents mean/std: [{latents.mean().item():.4f}, {latents.std().item():.4f}]")
    
    # Noise information
    print("\n🎲 NOISE INFORMATION:")
    print(f"   Noise shape: {noise.shape}")
    print(f"   Noise dtype: {noise.dtype}")
    print(f"   Noise min/max: [{noise.min().item():.4f}, {noise.max().item():.4f}]")
    print(f"   Noise mean/std: [{noise.mean().item():.4f}, {noise.std().item():.4f}]")
    
    # Noisy latents information
    print("\n🌀 NOISY LATENTS INFORMATION:")
    print(f"   Noisy latents shape: {noisy_latents.shape}")
    print(f"   Noisy latents min/max: [{noisy_latents.min().item():.4f}, {noisy_latents.max().item():.4f}]")
    print(f"   Noisy latents mean/std: [{noisy_latents.mean().item():.4f}, {noisy_latents.std().item():.4f}]")
    
    # Timestep information
    print("\n⏰ TIMESTEP INFORMATION:")
    print(f"   Timesteps shape: {timesteps.shape}")
    print(f"   Timesteps dtype: {timesteps.dtype}")
    print(f"   Timesteps values: {timesteps.tolist()}")
    print(f"   Timesteps min/max: [{timesteps.min().item()}, {timesteps.max().item()}]")
    print(f"   Timesteps mean: {timesteps.float().mean().item():.2f}")
    
    # Caption information
    print("\n📝 CAPTION INFORMATION:")
    print(f"   Number of captions: {len(captions)}")
    print(f"   Captions:")
    for i, caption in enumerate(captions):
        print(f"      [{i}]: '{caption}'")
    
    # Tokenization information
    print("\n🔤 TOKENIZATION INFORMATION:")
    print(f"   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs dtype: {input_ids.dtype}")
    print(f"   Input IDs device: {input_ids.device}")
    print(f"   Max sequence length: {input_ids.shape[1]}")
    print(f"   First caption tokens: {input_ids[0].tolist()}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    print(f"   Attention mask dtype: {attention_mask.dtype}")
    print(f"   First caption attention mask: {attention_mask[0].tolist()}")
    print(f"   Num attended tokens (first sample): {attention_mask[0].sum().item()}")
    
    # Text embeddings information
    print("\n🧠 TEXT EMBEDDINGS INFORMATION:")
    print(f"   Text embeddings shape: {text_embeddings.shape}")
    print(f"   Text embeddings dtype: {text_embeddings.dtype}")
    print(f"   Text embeddings device: {text_embeddings.device}")
    print(f"   Batch size: {text_embeddings.shape[0]}")
    print(f"   Sequence length: {text_embeddings.shape[1]}")
    print(f"   Embedding dimension: {text_embeddings.shape[2]}")
    print(f"   Text embeddings min/max: [{text_embeddings.min().item():.4f}, {text_embeddings.max().item():.4f}]")
    print(f"   Text embeddings mean/std: [{text_embeddings.mean().item():.4f}, {text_embeddings.std().item():.4f}]")
    
    # UNet prediction information
    print("\n🤖 UNET PREDICTION INFORMATION:")
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Prediction dtype: {pred.dtype}")
    print(f"   Prediction min/max: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"   Prediction mean/std: [{pred.mean().item():.4f}, {pred.std().item():.4f}]")
    
    # Loss information
    print("\n📉 LOSS INFORMATION:")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss dtype: {loss.dtype}")
    
    # Memory information (if CUDA is available)
    if torch.cuda.is_available():
        print("\n💾 GPU MEMORY INFORMATION:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"   GPU {i}:")
            print(f"      Allocated: {allocated:.2f} GB")
            print(f"      Reserved: {reserved:.2f} GB")
    
    print("\n" + "="*100)
    print(f"✅ DEBUG INFO COMPLETE - Step {global_step}")
    print("="*100 + "\n")


# ------------------ Simple Cross-Attention Block ------------------
class CrossAttention(nn.Module):
    """
    Cross-attention block: queries from spatial features, keys/values from text token embeddings.
    
    This allows each spatial position (H*W) to attend to all text tokens (T), enabling the model
    to selectively focus on different parts of the text prompt for different spatial regions.
    
    Input:
    - x: spatial features (B, C, H, W) 
    - text_embeds: text token embeddings (B, T, D) - preserves token dimension!
    - attention_mask: optional mask (B, T) to ignore padding tokens
    
    Output: 
    - attended features (B, C, H, W) conditioned on relevant text tokens
    """

    def __init__(self, spatial_channels: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        assert spatial_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = spatial_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # project spatial features (queries)
        self.to_q = nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1)
        # project text embeddings to k/v
        self.to_k = nn.Linear(text_dim, spatial_channels)
        self.to_v = nn.Linear(text_dim, spatial_channels)

        self.out = nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, text_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: (B, C, H, W)
        # text_embeds: (B, T, D) where T is number of text tokens
        # attention_mask: (B, T) where 1 = attend, 0 = ignore
        
        b, c, h, w = x.shape
        q = self.to_q(x)  # (B, C, H, W)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)  # (B, heads, head_dim, N)
        q = q.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # Handle text embeddings - ensure batch dimension consistency
        if text_embeds.dim() == 3:
            # text_embeds: (B, T, D)
            assert text_embeds.shape[0] == b, f"Batch size mismatch: x={b}, text_embeds={text_embeds.shape[0]}"
            _, T, D = text_embeds.shape
            
            k = self.to_k(text_embeds)  # (B, T, spatial_channels)
            v = self.to_v(text_embeds)  # (B, T, spatial_channels)
            
            # Reshape for multi-head attention: (B, T, spatial_channels) -> (B, heads, T, head_dim)
            k = k.reshape(b, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, T, head_dim)
            v = v.reshape(b, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, T, head_dim)
        else:
            # Fallback for 2D text embeddings - treat as single token
            assert text_embeds.shape[0] == b, f"Batch size mismatch: x={b}, text_embeds={text_embeds.shape[0]}"
            T = 1
            k = self.to_k(text_embeds).reshape(b, self.num_heads, self.head_dim).unsqueeze(2)  # (B, heads, 1, head_dim)
            v = self.to_v(text_embeds).reshape(b, self.num_heads, self.head_dim).unsqueeze(2)  # (B, heads, 1, head_dim)

        # Compute attention: q @ k^T -> (B, heads, N, T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, T)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T) - ensure it matches the token dimension
            assert attention_mask.shape == (b, T), f"Mask shape {attention_mask.shape} doesn't match (B={b}, T={T})"
            
            # Reshape mask to (B, 1, 1, T) to broadcast over (B, heads, N, T)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            # Set attention to very negative for masked tokens
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Softmax over the token dimension (T)
        attn = torch.softmax(attn, dim=-1)  # (B, heads, N, T)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        
        # Reshape back to spatial format
        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, c, h, w)  # (B, C, H, W)
        out = self.out(out)
        
        return out


# ------------------ Small UNet built from scratch ------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim=None, dropout=0.1):
        super().__init__()
        
        # First convolution path
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        
        # Second convolution path
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Timestep embedding projection - projects to 2x channels for scale and shift
        if time_embed_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, out_ch * 2)  # 2x for scale and shift (FiLM)
            )
        else:
            self.time_mlp = None
            
        # Residual connection - project input channels to output channels if different
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x, time_embed=None):
        residual = self.nin_shortcut(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply timestep conditioning using FiLM (Feature-wise Linear Modulation)
        if time_embed is not None and self.time_mlp is not None:
            time_embed_proj = self.time_mlp(time_embed)  # (B, 2*C)
            scale, shift = time_embed_proj.chunk(2, dim=1)  # (B, C), (B, C)
            scale = scale[:, :, None, None]  # (B, C, 1, 1)
            shift = shift[:, :, None, None]  # (B, C, 1, 1)
            h = h * (1 + scale) + shift  # FiLM conditioning
            
        h = F.silu(h)  # Use SiLU activation instead of ReLU
        h = self.dropout(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Residual connection
        h = h + residual
        h = F.silu(h)  # Final activation
        
        return h


class SimpleUNet(nn.Module):
    """Enhanced conditional UNet with cross-attention to text embeddings and timestep embeddings."""

    def __init__(self, in_channels=4, base_channels=128, channel_mults=(1, 2, 4), text_dim=512):
        super().__init__()
        
        # Input conv
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Time embedding
        time_embed_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        self.down_channels = []  # Track channels for skip connections
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleDict({
                'res1': ResidualBlock(ch, out_ch, time_embed_dim),
                'res2': ResidualBlock(out_ch, out_ch, time_embed_dim),
                'attn': CrossAttention(out_ch, text_dim, num_heads=8),
                'res3': ResidualBlock(out_ch, out_ch, time_embed_dim),
                'downsample': self._make_downsample(out_ch) if i < len(channel_mults) - 1 else nn.Identity()
            }))
            self.down_channels.append(out_ch)
            ch = out_ch

        # Middle block
        self.mid = nn.ModuleDict({
            'res1': ResidualBlock(ch, ch, time_embed_dim),
            'attn1': CrossAttention(ch, text_dim, num_heads=8),
            'res2': ResidualBlock(ch, ch, time_embed_dim),
            'attn2': CrossAttention(ch, text_dim, num_heads=8),
            'res3': ResidualBlock(ch, ch, time_embed_dim)
        })

        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = self.down_channels[-(i+1)]  # Get corresponding skip channel
            
            self.up_blocks.append(nn.ModuleDict({
                'upsample': self._make_upsample(ch, out_ch) if i > 0 else nn.Identity(),
                'res1': ResidualBlock((ch if i == 0 else out_ch) + skip_ch, out_ch, time_embed_dim),  # Concatenated input
                'res2': ResidualBlock(out_ch, out_ch, time_embed_dim),
                'attn': CrossAttention(out_ch, text_dim, num_heads=8),
                'res3': ResidualBlock(out_ch, out_ch, time_embed_dim),
            }))
            ch = out_ch

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)
        )
    
    def _make_downsample(self, channels):
        """Create safe downsampling layer that works with small dimensions"""
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=1)  # 1x1 conv to maintain channels
        )
    
    def _make_upsample(self, in_ch, out_ch):
        """Create safe upsampling layer"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        From Fairseq. Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, timesteps, text_embeds):
        # x: (B, C, H, W)
        # timesteps: (B,)
        # text_embeds: (B, seq_len, text_dim)
        
        # Get timestep embeddings
        time_embed = self.get_timestep_embedding(timesteps, self.time_embed[0].in_features)  # Use first layer input size
        time_embed = self.time_embed(time_embed)  # Project to time_embed_dim
        
        # Initial convolution
        h = self.in_conv(x)
        skips = []
        
        # Down path
        for i, block in enumerate(self.down_blocks):
            # Apply residual blocks
            h = block['res1'](h, time_embed)
            h = block['res2'](h, time_embed)
            
            # Apply cross-attention
            h = block['attn'](h, text_embeds)
            h = block['res3'](h, time_embed)
            skips.append(h)
            
            # Downsample (except for the last block)
            h = block['downsample'](h)

        # Middle blocks with dual attention
        h = self.mid['res1'](h, time_embed)
        h = self.mid['attn1'](h, text_embeds)
        h = self.mid['res2'](h, time_embed)
        h = self.mid['attn2'](h, text_embeds)
        h = self.mid['res3'](h, time_embed)

        # Up path
        for i, block in enumerate(self.up_blocks):
            # Upsample
            h = block['upsample'](h)
            
            # Concatenate skip connection
            if skips:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
            
            # Apply residual blocks
            h = block['res1'](h, time_embed)
            h = block['attn'](h, text_embeds)
            h = block['res2'](h, time_embed)
            h = block['res3'](h, time_embed)

        # Final output
        out = self.out_conv(h)
        return out


# ------------------ Dataset loader ------------------

class CIFAR10WithCaptions:
    """Wrapper for CIFAR-10 dataset that includes class names and caption generation."""
    
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, root: str, image_size: int, download: bool = True):
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset = CIFAR10(root=root, download=download, transform=transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        caption = f"a photo of a {self.CLASS_NAMES[label]}"
        return image, label, caption
    
    def get_class_name(self, label: int) -> str:
        """Get class name for a given label."""
        return self.CLASS_NAMES[label]
    
    def get_caption(self, label: int) -> str:
        """Get caption for a given label."""
        return f"a photo of a {self.CLASS_NAMES[label]}"

def get_dataloader(batch_size: int, image_size: int):
    dataset = CIFAR10WithCaptions(root=str(ROOT_DIR / "data"), image_size=image_size, download=True)
    # Reduce num_workers for smaller batch size to avoid overhead
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dl


# ------------------ Main training and inference ------------------

def save_checkpoint(unet, optimizer, global_step, epoch):
    """Save checkpoint without generating images."""
    # Create step-specific directories
    step_dir = CHECKPOINT_DIR / f"epoch_{epoch}_step_{global_step}"
    step_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = step_dir / f"ckpt_epoch_{epoch}_step_{global_step}.pt"
    torch.save({
        'unet_state_dict': unet.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'step': global_step,
        'epoch': epoch
    }, checkpoint_path)
    print(f"💾 Saved checkpoint at epoch {epoch}, step {global_step} to {checkpoint_path}")


def plot_loss_curve(losses, epoch, global_step, save_path=None):
    """
    Plot the training loss curve and save it to disk.
    
    Args:
        losses: List of all loss values recorded during training
        epoch: Current epoch number
        global_step: Current global step number
        save_path: Optional custom save path. If None, uses PLOTS_DIR
    """
    if len(losses) == 0:
        print("⚠️  No losses to plot")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 PLOTTING LOSS CURVE")
    print(f"   Epoch: {epoch}")
    print(f"   Global Step: {global_step}")
    print(f"   Total Loss Values: {len(losses)}")
    print(f"{'='*80}\n")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Full loss curve
    ax1.plot(losses, alpha=0.6, linewidth=0.5, color='blue', label='Raw Loss')
    
    # Add moving average (window of 100 steps)
    if len(losses) >= 100:
        moving_avg_window = 100
        moving_avg = np.convolve(losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        ax1.plot(range(moving_avg_window-1, len(losses)), moving_avg, 
                color='red', linewidth=2, label=f'Moving Avg (window={moving_avg_window})')
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'Training Loss Curve - Epoch {epoch}, Step {global_step}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add statistics text box
    stats_text = f'Current Loss: {losses[-1]:.6f}\n'
    stats_text += f'Min Loss: {min(losses):.6f}\n'
    stats_text += f'Max Loss: {max(losses):.6f}\n'
    stats_text += f'Mean Loss: {np.mean(losses):.6f}\n'
    stats_text += f'Std Loss: {np.std(losses):.6f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Recent loss (last 1000 steps or all if less)
    recent_window = min(1000, len(losses))
    recent_losses = losses[-recent_window:]
    
    ax2.plot(range(len(losses) - recent_window, len(losses)), recent_losses, 
            alpha=0.7, linewidth=1, color='green', label='Recent Loss')
    
    # Add moving average for recent losses
    if len(recent_losses) >= 50:
        recent_moving_avg_window = 50
        recent_moving_avg = np.convolve(recent_losses, 
                                       np.ones(recent_moving_avg_window)/recent_moving_avg_window, 
                                       mode='valid')
        ax2.plot(range(len(losses) - recent_window + recent_moving_avg_window - 1, len(losses)), 
                recent_moving_avg, color='orange', linewidth=2, 
                label=f'Moving Avg (window={recent_moving_avg_window})')
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title(f'Recent Training Loss (Last {recent_window} Steps)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add recent statistics text box
    recent_stats_text = f'Recent Avg: {np.mean(recent_losses):.6f}\n'
    recent_stats_text += f'Recent Min: {min(recent_losses):.6f}\n'
    recent_stats_text += f'Recent Max: {max(recent_losses):.6f}'
    ax2.text(0.02, 0.98, recent_stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = PLOTS_DIR / f"loss_curve_epoch_{epoch}_step_{global_step}.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{'='*80}")
    print(f"✅ LOSS CURVE SAVED")
    print(f"   Path: {save_path}")
    print(f"   File exists: {save_path.exists()}")
    print(f"   Full path: {save_path.absolute()}")
    print(f"{'='*80}\n")


def generate_sample_images(unet, vae, tokenizer, text_encoder, noise_scheduler, global_step, epoch, device):
    """Generate sample images and reconstructions without saving checkpoint."""
    # Create images directory
    images_dir = EXAMPLES_DIR / f"epoch_{epoch}_step_{global_step}"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🎨 GENERATING SAMPLE IMAGES AND RECONSTRUCTIONS")
    print(f"   Epoch: {epoch}")
    print(f"   Global Step: {global_step}")
    print(f"   Output Directory: {images_dir}")
    print(f"   Directory exists: {images_dir.exists()}")
    print(f"   Expected output image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"{'='*80}\n")
    
    # Generate sample images
    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    # Use all dataset class names to generate prompts
    dataset_class_names = CIFAR10WithCaptions.CLASS_NAMES
    prompts = [f"a photo of a {class_name}" for class_name in dataset_class_names]
    
    print(f"🎨 Generating images for all {len(prompts)} CIFAR-10 classes: {dataset_class_names}")
    
    # Move models to the same device for inference
    device = next(unet.parameters()).device
    
    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])[0]
    
    # Generate latents
    bsz = len(prompts)
    latent_shape = (bsz, 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
    latents = torch.randn(latent_shape, device=device)
    
    print(f"\n🔧 INFERENCE DIMENSIONS:")
    print(f"   Initial latents shape: {latents.shape}")
    print(f"   Expected latent size: {IMAGE_SIZE // 4} x {IMAGE_SIZE // 4}")
    
    # Set scheduler for inference
    scheduler_copy = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    scheduler_copy.set_timesteps(50)
    timesteps = scheduler_copy.timesteps
    
    # Denoising loop
    for t in timesteps:
        t_batch = torch.tensor([int(t)] * bsz, device=device)
        with torch.no_grad():
            pred_noise = unet(latents, t_batch, text_embeddings)
        latents = scheduler_copy.step(pred_noise, t, latents).prev_sample
    
    print(f"   Final latents shape after denoising: {latents.shape}")
    
    # Decode to images - use the same scaling factor as during training
    vae_scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
    print(f"   VAE scaling factor: {vae_scaling_factor}")
    print(f"   Latents before VAE decode: {latents.shape}")
    
    with torch.no_grad():
        try:
            decoded_images = vae.decode(latents / vae_scaling_factor).sample
            print(f"   Images after VAE decode: {decoded_images.shape}")
            print(f"   VAE upsampling factor: {decoded_images.shape[2] / latents.shape[2]:.2f}x")
        except Exception as e:
            print(f"Warning: VAE decode failed with scaling, trying without: {e}")
            decoded_images = vae.decode(latents).sample
            print(f"   Images after VAE decode (no scaling): {decoded_images.shape}")
    
    # Convert to [0,1] range
    images = (decoded_images.clamp(-1, 1) + 1) / 2
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    print(f"   Final numpy image shape: {images.shape}")
    print(f"   Expected: ({bsz}, {IMAGE_SIZE}, {IMAGE_SIZE}, 3)")
    print(f"   Actual image size: {images.shape[1]} x {images.shape[2]}")
    
    if images.shape[1] != IMAGE_SIZE or images.shape[2] != IMAGE_SIZE:
        print(f"\n⚠️  WARNING: Output image size mismatch!")
        print(f"   Expected: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"   Got: {images.shape[1]}x{images.shape[2]}")
        print(f"   This is likely due to the VAE's internal upsampling behavior.")
        print(f"   The SDXL VAE may not preserve exact dimensions for small images.")
        print(f"   Consider resizing images back to {IMAGE_SIZE}x{IMAGE_SIZE} or using a different VAE.")
    
    # Save individual images
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        img_filename = f"epoch_{epoch}_step_{global_step}_sample_{i}_{prompt.replace(' ', '_').replace(',', '')}.png"
        img_path = images_dir / img_filename
        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(f"Epoch {epoch}, Step {global_step}: {prompt}\nSize: {img.shape[0]}x{img.shape[1]}", fontsize=8)
        plt.tight_layout()
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save grid image
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = np.array(axs).reshape(-1)
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        axs[i].imshow(np.clip(img, 0, 1))
        axs[i].set_title(prompt, fontsize=8)
        axs[i].axis('off')
    for j in range(len(images), len(axs)):
        axs[j].axis('off')
    
    grid_path = images_dir / f"epoch_{epoch}_step_{global_step}_grid.png"
    plt.suptitle(f"Generated Images at Epoch {epoch}, Step {global_step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"✅ GENERATED IMAGES SAVED SUCCESSFULLY")
    print(f"   Total images: {len(images)}")
    print(f"   Grid image: {grid_path}")
    print(f"   Grid exists: {grid_path.exists()}")
    print(f"   Full path: {grid_path.absolute()}")
    print(f"{'='*80}\n")
    
    # ========== PART 2: RECONSTRUCTION FROM DATASET ==========
    print(f"\n{'='*80}")
    print(f"🔄 GENERATING RECONSTRUCTIONS FROM DATASET")
    print(f"{'='*80}\n")
    
    # Get one sample from each CIFAR-10 class for reconstruction
    dataset = CIFAR10WithCaptions(root=str(ROOT_DIR / "data"), image_size=IMAGE_SIZE, download=True)
    
    # Collect one image per class
    class_samples = {}
    original_images_list = []
    captions_list = []
    labels_list = []
    
    print("📦 Collecting samples from dataset (one per class)...")
    for i in range(len(dataset)):
        image, label, caption = dataset[i]
        if label not in class_samples:
            class_samples[label] = (image, caption)
            original_images_list.append(image)
            captions_list.append(caption)
            labels_list.append(label)
            
        if len(class_samples) == 10:  # Got all 10 classes
            break
    
    # Stack images into a batch
    original_images = torch.stack(original_images_list).to(device)
    bsz_recon = original_images.shape[0]
    
    print(f"✅ Collected {bsz_recon} samples (one per class)")
    print(f"   Original images shape: {original_images.shape}")
    
    # Encode original images to latents
    print("🎨 Encoding original images to latents...")
    vae_scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
    with torch.no_grad():
        encoded = vae.encode(original_images)
        if hasattr(encoded, 'latent_dist'):
            clean_latents = encoded.latent_dist.sample() * vae_scaling_factor
        else:
            clean_latents = encoded.latent * vae_scaling_factor
    
    print(f"   Clean latents shape: {clean_latents.shape}")
    
    # Add noise to latents (moderate noise level for reconstruction test)
    reconstruction_noise_timestep = 500  # Medium noise level
    print(f"🌀 Adding noise to latents (t={reconstruction_noise_timestep})...")
    noise_recon = torch.randn_like(clean_latents)
    timesteps_noise = torch.tensor([reconstruction_noise_timestep] * bsz_recon, device=device).long()
    noisy_latents_recon = noise_scheduler.add_noise(clean_latents, noise_recon, timesteps_noise)
    
    # Get text embeddings
    print("🧠 Getting text embeddings...")
    tokenized_recon = tokenizer(captions_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings_recon = text_encoder(input_ids=tokenized_recon['input_ids'], 
                                            attention_mask=tokenized_recon['attention_mask'])[0]
    
    # Denoise using the trained UNet
    print(f"🤖 Denoising with trained UNet...")
    scheduler_copy_recon = DDPMScheduler(
        beta_start=0.0001, 
        beta_end=0.02, 
        beta_schedule="linear", 
        num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )
    scheduler_copy_recon.set_timesteps(50)  # 50 denoising steps
    
    # Start denoising from the noisy latents
    latents_recon = noisy_latents_recon.clone()
    
    # Only denoise from the specified timestep down to 0
    inference_timesteps_recon = [t for t in scheduler_copy_recon.timesteps if t <= reconstruction_noise_timestep]
    print(f"   Denoising steps: {len(inference_timesteps_recon)} (from t={reconstruction_noise_timestep} to t=0)")
    
    for t in inference_timesteps_recon:
        t_batch = torch.tensor([int(t)] * bsz_recon, device=device)
        with torch.no_grad():
            pred_noise_recon = unet(latents_recon, t_batch, text_embeddings_recon)
        latents_recon = scheduler_copy_recon.step(pred_noise_recon, t, latents_recon).prev_sample
    
    # Decode reconstructed latents to images
    print("🖼️  Decoding reconstructed latents to images...")
    with torch.no_grad():
        reconstructed_images = vae.decode(latents_recon / vae_scaling_factor).sample
    
    # Convert to [0,1] range for visualization
    original_display = (original_images.clamp(-1, 1) + 1) / 2
    reconstructed_display = (reconstructed_images.clamp(-1, 1) + 1) / 2
    
    # Convert to numpy
    original_np = original_display.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_np = reconstructed_display.cpu().permute(0, 2, 3, 1).numpy()
    
    # Save reconstruction comparison grid
    print("💾 Saving reconstruction comparison grid...")
    fig, axs = plt.subplots(2, bsz_recon, figsize=(bsz_recon * 2.5, 2 * 2.5))
    
    for i in range(bsz_recon):
        class_name = CIFAR10WithCaptions.CLASS_NAMES[labels_list[i]]
        
        # Row 1: Original
        axs[0, i].imshow(np.clip(original_np[i], 0, 1))
        axs[0, i].set_title(f"Original\n{class_name}", fontsize=8)
        axs[0, i].axis('off')
        
        # Row 2: Reconstructed
        axs[1, i].imshow(np.clip(reconstructed_np[i], 0, 1))
        axs[1, i].set_title(f"Reconstructed\n(from t={reconstruction_noise_timestep})", fontsize=8)
        axs[1, i].axis('off')
    
    # Add row labels
    axs[0, 0].text(-0.5, 0.5, 'Original\nImages', transform=axs[0, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[1, 0].text(-0.5, 0.5, 'Reconstructed\nImages', transform=axs[1, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    
    plt.suptitle(f"Reconstruction at Epoch {epoch}, Step {global_step}\n"
                 f"(Original → Encode → Add Noise (t={reconstruction_noise_timestep}) → Denoise → Decode)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    reconstruction_grid_path = images_dir / f"epoch_{epoch}_step_{global_step}_reconstruction_grid.png"
    plt.savefig(reconstruction_grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"✅ RECONSTRUCTION IMAGES SAVED SUCCESSFULLY")
    print(f"   Total reconstructions: {len(reconstructed_np)}")
    print(f"   Reconstruction grid: {reconstruction_grid_path}")
    print(f"   Grid exists: {reconstruction_grid_path.exists()}")
    print(f"   Full path: {reconstruction_grid_path.absolute()}")
    print(f"{'='*80}\n")
    
    # ========== PART 3: COMBINED GRID ==========
    print(f"\n{'='*80}")
    print(f"📊 CREATING COMBINED GRID (Generation + Reconstruction)")
    print(f"{'='*80}\n")
    
    # Create a large combined grid showing both generation and reconstruction
    # 3 rows: Generated from noise, Original from dataset, Reconstructed from dataset
    fig, axs = plt.subplots(3, 10, figsize=(25, 3 * 2.5))
    
    # Row 1: Generated images (from pure noise)
    for i in range(10):
        axs[0, i].imshow(np.clip(images[i], 0, 1))
        axs[0, i].set_title(dataset_class_names[i], fontsize=8)
        axs[0, i].axis('off')
    
    # Row 2: Original images from dataset
    for i in range(10):
        axs[1, i].imshow(np.clip(original_np[i], 0, 1))
        axs[1, i].set_title(dataset_class_names[i], fontsize=8)
        axs[1, i].axis('off')
    
    # Row 3: Reconstructed images
    for i in range(10):
        axs[2, i].imshow(np.clip(reconstructed_np[i], 0, 1))
        axs[2, i].set_title(dataset_class_names[i], fontsize=8)
        axs[2, i].axis('off')
    
    # Add row labels
    axs[0, 0].text(-0.5, 0.5, 'Generated\n(from noise)', transform=axs[0, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[1, 0].text(-0.5, 0.5, 'Original\n(dataset)', transform=axs[1, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[2, 0].text(-0.5, 0.5, 'Reconstructed\n(denoised)', transform=axs[2, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    
    plt.suptitle(f"Generation vs Reconstruction at Epoch {epoch}, Step {global_step}\n"
                 f"Top: Pure generation (random noise → denoise), Middle: Original dataset images, "
                 f"Bottom: Reconstruction (encode → noise t={reconstruction_noise_timestep} → denoise)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    combined_grid_path = images_dir / f"epoch_{epoch}_step_{global_step}_combined_grid.png"
    plt.savefig(combined_grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"✅ COMBINED GRID SAVED SUCCESSFULLY")
    print(f"   Combined grid: {combined_grid_path}")
    print(f"   Grid exists: {combined_grid_path.exists()}")
    print(f"   Full path: {combined_grid_path.absolute()}")
    print(f"{'='*80}\n")
    
    # Set model back to training mode
    unet.train()

def save_checkpoint_and_generate_images(unet, vae, tokenizer, text_encoder, noise_scheduler, optimizer, global_step, epoch, device):
    """Save checkpoint and generate sample images at the current training step."""
    save_checkpoint(unet, optimizer, global_step, epoch)
    generate_sample_images(unet, vae, tokenizer, text_encoder, noise_scheduler, global_step, epoch, device)

def train():
    # Run GPU checks before any other setup
    print("🚂 STARTING TRAINING")
    print("=" * 80)
    
    gpu_count = check_gpu_setup()
    accelerator = setup_accelerator()
    device = accelerator.device
    
    # load components
    print("=" * 80)
    print("🤖 LOADING MODELS")
    print("=" * 80)
    print("Device:", device)
    print(f"Using {gpu_count} GPU(s)")
    
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_CLIP)
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_CLIP)

    # VAE: try to load a pretrained VAE for latents
    try:
        vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE)
        vae_scaling = getattr(vae.config, "scaling_factor", 0.13025)  # SDXL VAE uses 0.13025
        print(f"✅ Loaded pretrained VAE with scaling factor: {vae_scaling}")
    except Exception as e:
        print("❌ Could not load pretrained VAE, Error:", e)
        exit(1)

    # create scheduler
    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # create UNet from scratch - enhanced but with proper dimensions for CIFAR-10
    latent_channels = 4
    unet = SimpleUNet(
        in_channels=latent_channels, 
        base_channels=128,  # Increased from 64
        channel_mults=(1, 2),  # Only 1 downsampling level for 32x32 images to avoid spatial collapse
        text_dim=text_encoder.config.hidden_size
    )
    
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"✅ Created enhanced UNet with {total_params:,} total parameters ({trainable_params:,} trainable)")
    print(f"📈 Parameter breakdown:")
    print(f"   - Base channels: {128}")
    print(f"   - Channel progression: {[128 * mult for mult in (1, 2)]}")
    print(f"   - 3 residual blocks per down/up level")
    print(f"   - Attention heads: 8 (increased from 4)")
    print(f"   - Dual attention in middle block")
    print(f"   - Latent dimensions: {IMAGE_SIZE}x{IMAGE_SIZE} → {IMAGE_SIZE//4}x{IMAGE_SIZE//4} after VAE encoding")
    print(f"   - After downsampling: {IMAGE_SIZE//4}x{IMAGE_SIZE//4} → {IMAGE_SIZE//8}x{IMAGE_SIZE//8} (only 1 level to preserve spatial info)")

    # optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # dataloader
    dataloader = get_dataloader(BATCH_SIZE, IMAGE_SIZE)
    print(f"✅ Created dataloader with batch size {BATCH_SIZE}")

    # Prepare everything with accelerator
    print("\n🔧 PREPARING MODELS WITH ACCELERATOR")
    print("=" * 80)
    unet, text_encoder, vae, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, vae, optimizer, dataloader
    )
    
    # Set models to appropriate modes
    text_encoder.eval()
    vae.eval()
    
    print("✅ All models prepared with Accelerator")
    print(f"Models are on device: {next(unet.parameters()).device}")
    print()

    global_step = 0
    losses = []

    print("🚂 STARTING TRAINING")
    print("=" * 80)
    
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs", disable=not accelerator.is_local_main_process):
        epoch_losses = []
        
        # Create progress bar for batches within the epoch
        batch_pbar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", 
            leave=False,
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(batch_pbar):
            images, labels, captions = batch

            # encode with VAE to latents
            with torch.no_grad():
                if hasattr(vae, 'encode'):
                    encoded = vae.encode(images)
                    if hasattr(encoded, 'latent_dist'):
                        latents = encoded.latent_dist.sample() * vae_scaling
                    else:
                        latents = encoded.latent
                else:
                    raise RuntimeError("VAE has no encode().")

            # sample random noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # text conditioning: use captions from dataset
            # Classifier-free guidance: randomly drop text conditioning
            if torch.rand(1).item() < CLASSIFIER_FREE_GUIDANCE_DROPOUT:
                # Use unconditional (empty) text embeddings
                unconditional_captions = [""] * len(captions)
                tokenized = tokenizer(unconditional_captions, padding='max_length', truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            else:
                tokenized = tokenizer(captions, padding='max_length', truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            
            input_ids = tokenized['input_ids'].to(accelerator.device)
            attention_mask = tokenized['attention_mask'].to(accelerator.device)
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

            # predict noise with UNet
            unet.train()
            
            # Use accelerator's autocast for mixed precision
            with accelerator.autocast():
                pred = unet(noisy_latents, timesteps, text_embeddings)
                loss = F.mse_loss(pred, noise)

            # Backward pass with accelerator
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            current_loss = loss.item()
            losses.append(current_loss)
            epoch_losses.append(current_loss)
            global_step += 1

            # Print debug info every 1000 steps (only on main process)
            if global_step % 1000 == 0 and accelerator.is_local_main_process:
                print_debug_info(
                    global_step=global_step,
                    images=images,
                    latents=latents,
                    noise=noise,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    captions=captions,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    text_embeddings=text_embeddings,
                    pred=pred,
                    loss=loss,
                    batch_idx=batch_idx
                )

            # Update progress bar with current loss and moving average (only on main process)
            if accelerator.is_local_main_process:
                recent_avg = sum(losses[-20:]) / min(len(losses), 20)
                batch_pbar.set_postfix({
                    'loss': f'{current_loss:.6f}',
                    'avg_loss': f'{recent_avg:.6f}',
                    'step': global_step
                })

            if global_step >= MAX_TRAIN_STEPS:
                break
                
        # End of epoch - check if we need to save checkpoint or generate images
        # Wait for all processes to sync before saving
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            # Unwrap models for saving/inference
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_vae = accelerator.unwrap_model(vae)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            
            # Check if we should save checkpoint (every N epochs)
            should_save_checkpoint = (epoch + 1) % SAVE_CHECKPOINT_EVERY_EPOCHS == 0
            
            # Check if we should generate images (every N epochs)
            should_generate_images = (epoch + 1) % GENERATE_IMAGES_EVERY_EPOCHS == 0
            
            if accelerator.is_local_main_process:
                print(f"\n{'='*80}")
                print(f"📋 End of Epoch {epoch+1}:")
                print(f"   - Should save checkpoint: {should_save_checkpoint} (every {SAVE_CHECKPOINT_EVERY_EPOCHS} epochs)")
                print(f"   - Should generate images: {should_generate_images} (every {GENERATE_IMAGES_EVERY_EPOCHS} epochs)")
                print(f"{'='*80}\n")
                
                # Plot loss curve after every epoch
                plot_loss_curve(losses, epoch + 1, global_step)
            
            if should_save_checkpoint and should_generate_images:
                # Both checkpoint and images
                if accelerator.is_local_main_process:
                    print(f"📸 Epoch {epoch+1}: Saving checkpoint AND generating images...")
                save_checkpoint_and_generate_images(
                    unwrapped_unet, unwrapped_vae, tokenizer, 
                    unwrapped_text_encoder, noise_scheduler, 
                    optimizer, global_step, epoch + 1, device
                )
            elif should_save_checkpoint:
                # Only checkpoint
                if accelerator.is_local_main_process:
                    print(f"💾 Epoch {epoch+1}: Saving checkpoint...")
                save_checkpoint(unwrapped_unet, optimizer, global_step, epoch + 1)
            elif should_generate_images:
                # Only images
                if accelerator.is_local_main_process:
                    print(f"🎨 Epoch {epoch+1}: Generating sample images...")
                generate_sample_images(
                    unwrapped_unet, unwrapped_vae, tokenizer, 
                    unwrapped_text_encoder, noise_scheduler, 
                    global_step, epoch + 1, device
                )
        
        # Print epoch summary (only on main process)
        if accelerator.is_local_main_process:
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            epoch_min_loss = min(epoch_losses) if epoch_losses else 0
            epoch_max_loss = max(epoch_losses) if epoch_losses else 0
            tqdm.write(f"📊 Epoch {epoch+1} completed - Avg loss: {epoch_avg_loss:.6f} | Min: {epoch_min_loss:.6f} | Max: {epoch_max_loss:.6f}")
            
            # Warning if loss is not decreasing or showing signs of collapse
            if epoch > 10 and epoch_avg_loss > 0.5:
                tqdm.write(f"⚠️  Warning: Loss is still high ({epoch_avg_loss:.6f}) after {epoch+1} epochs. Check generated images for quality.")
        
        if global_step >= MAX_TRAIN_STEPS:
            break

    # Final save (only on main process)
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_vae = accelerator.unwrap_model(vae)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        
        # Final loss plot
        if accelerator.is_local_main_process:
            plot_loss_curve(losses, epoch + 1, global_step, 
                          save_path=PLOTS_DIR / "final_loss_curve.png")
        
        save_checkpoint_and_generate_images(
            unwrapped_unet, unwrapped_vae, tokenizer, 
            unwrapped_text_encoder, noise_scheduler, 
            optimizer, global_step, epoch, device
        )
        print("✅ Training finished. Final checkpoint, images, and loss plot saved.")


def reconstruct_from_dataset(unet: nn.Module, vae: nn.Module, tokenizer: CLIPTokenizer, 
                             text_encoder: CLIPTextModel, scheduler: DDPMScheduler, 
                             device: torch.device, num_samples: int = 10, noise_timestep: int = 0):
    """
    Reconstruct images from the dataset by:
    1. Taking real images from the dataset
    2. Encoding them to latents with VAE
    3. Optionally adding noise to the latents (simulating the forward diffusion process)
    4. Denoising with the trained UNet to reconstruct the original (if noise was added)
    
    This helps compare:
    - Pure random noise generation (starting from N(0,1))
    - Reconstruction from learned latent distribution (starting from VAE-encoded real images)
    
    Args:
        unet: Trained UNet model
        vae: VAE encoder/decoder
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        scheduler: Noise scheduler
        device: Device to run on
        num_samples: Number of samples to reconstruct (one per class)
        noise_timestep: Timestep for adding noise (0 = no noise, just VAE encode/decode test)
                       Higher values add more noise (max = NUM_TRAIN_TIMESTEPS)
    """
    print(f"\n{'='*80}")
    print(f"🔄 RECONSTRUCTION FROM DATASET")
    print(f"{'='*80}\n")
    
    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    # Get one sample from each CIFAR-10 class
    dataset = CIFAR10WithCaptions(root=str(ROOT_DIR / "data"), image_size=IMAGE_SIZE, download=True)
    
    # Collect one image per class
    class_samples = {}
    original_images_list = []
    captions_list = []
    labels_list = []
    
    print("📦 Collecting samples from dataset (one per class)...")
    for i in range(len(dataset)):
        image, label, caption = dataset[i]
        if label not in class_samples:
            class_samples[label] = (image, caption)
            original_images_list.append(image)
            captions_list.append(caption)
            labels_list.append(label)
            
        if len(class_samples) == 10:  # Got all 10 classes
            break
    
    # Stack images into a batch
    original_images = torch.stack(original_images_list).to(device)
    bsz = original_images.shape[0]
    
    print(f"✅ Collected {bsz} samples (one per class)")
    print(f"   Original images shape: {original_images.shape}")
    print(f"   Classes: {[CIFAR10WithCaptions.CLASS_NAMES[label] for label in labels_list]}")
    
    # Step 1: Encode original images to latents
    print("\n🎨 Step 1: Encoding original images to latents...")
    vae_scaling_factor = getattr(vae.config, 'scaling_factor', 0.13025)
    with torch.no_grad():
        encoded = vae.encode(original_images)
        if hasattr(encoded, 'latent_dist'):
            clean_latents = encoded.latent_dist.sample() * vae_scaling_factor
        else:
            clean_latents = encoded.latent * vae_scaling_factor
    
    print(f"   Clean latents shape: {clean_latents.shape}")
    print(f"   Latents min/max: [{clean_latents.min().item():.4f}, {clean_latents.max().item():.4f}]")
    
    # Step 2: Optionally add noise to simulate forward diffusion
    if noise_timestep > 0:
        print(f"\n🌀 Step 2: Adding noise to latents (simulating forward diffusion)...")
        noise = torch.randn_like(clean_latents)
        timesteps_noise = torch.tensor([noise_timestep] * bsz, device=device).long()
        
        # Use the scheduler to add noise properly
        noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps_noise)
        
        print(f"   Noise timestep: {noise_timestep}/{scheduler.config.num_train_timesteps}")
        print(f"   Noisy latents shape: {noisy_latents.shape}")
        print(f"   Noisy latents min/max: [{noisy_latents.min().item():.4f}, {noisy_latents.max().item():.4f}]")
    else:
        print(f"\n🌀 Step 2: Skipping noise addition (noise_timestep=0)")
        print(f"   Will perform pure VAE encode/decode test (no diffusion)")
        noisy_latents = clean_latents.clone()
    
    # Step 3: Get text embeddings for conditioning
    print("\n🧠 Step 3: Getting text embeddings...")
    tokenized = tokenizer(captions_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=tokenized['input_ids'], 
                                       attention_mask=tokenized['attention_mask'])[0]
    print(f"   Text embeddings shape: {text_embeddings.shape}")
    
    # Step 4: Denoise using the trained UNet (only if noise was added)
    if noise_timestep > 0:
        print(f"\n🤖 Step 4: Denoising with trained UNet...")
        scheduler_copy = DDPMScheduler(
            beta_start=0.0001, 
            beta_end=0.02, 
            beta_schedule="linear", 
            num_train_timesteps=NUM_TRAIN_TIMESTEPS
        )
        scheduler_copy.set_timesteps(50)  # 50 denoising steps
        
        # Start denoising from the noisy latents
        latents = noisy_latents.clone()
        
        # Only denoise from the specified timestep down to 0
        inference_timesteps = [t for t in scheduler_copy.timesteps if t <= noise_timestep]
        print(f"   Denoising steps: {len(inference_timesteps)} (from t={noise_timestep} to t=0)")
        
        for t in tqdm(inference_timesteps, desc="Denoising", leave=False):
            t_batch = torch.tensor([int(t)] * bsz, device=device)
            with torch.no_grad():
                pred_noise = unet(latents, t_batch, text_embeddings)
            latents = scheduler_copy.step(pred_noise, t, latents).prev_sample
        
        print(f"   Final denoised latents shape: {latents.shape}")
    else:
        print(f"\n🤖 Step 4: Skipping denoising (noise_timestep=0)")
        print(f"   Using clean latents directly (no UNet inference)")
        latents = noisy_latents  # which is just clean_latents.clone()
    
    # Step 5: Decode latents back to images
    print(f"\n🖼️  Step 5: Decoding latents to images...")
    with torch.no_grad():
        reconstructed_images = vae.decode(latents / vae_scaling_factor).sample
    
    print(f"   Reconstructed images shape: {reconstructed_images.shape}")
    
    # Also decode the original clean latents for comparison
    with torch.no_grad():
        vae_reconstructed = vae.decode(clean_latents / vae_scaling_factor).sample
    
    # Convert all to [0,1] range for visualization
    original_display = (original_images.clamp(-1, 1) + 1) / 2
    vae_reconstructed_display = (vae_reconstructed.clamp(-1, 1) + 1) / 2
    reconstructed_display = (reconstructed_images.clamp(-1, 1) + 1) / 2
    
    # Convert to numpy
    original_np = original_display.cpu().permute(0, 2, 3, 1).numpy()
    vae_reconstructed_np = vae_reconstructed_display.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed_np = reconstructed_display.cpu().permute(0, 2, 3, 1).numpy()
    
    # Step 6: Save comparison images
    print("\n💾 Step 6: Saving comparison images...")
    comparison_dir = EXAMPLES_DIR / "reconstruction_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Create a comprehensive comparison grid
    # 3 rows: Original, VAE Reconstruction (no diffusion), Full Reconstruction (with/without diffusion)
    # 10 columns: One per class
    fig, axs = plt.subplots(3, bsz, figsize=(bsz * 2.5, 3 * 2.5))
    
    for i in range(bsz):
        class_name = CIFAR10WithCaptions.CLASS_NAMES[labels_list[i]]
        
        # Row 1: Original
        axs[0, i].imshow(np.clip(original_np[i], 0, 1))
        axs[0, i].set_title(f"Original\n{class_name}", fontsize=8)
        axs[0, i].axis('off')
        
        # Row 2: VAE only (always no diffusion for comparison baseline)
        axs[1, i].imshow(np.clip(vae_reconstructed_np[i], 0, 1))
        axs[1, i].set_title(f"VAE Only\n(no diffusion)", fontsize=8)
        axs[1, i].axis('off')
        
        # Row 3: Full reconstruction (with or without diffusion depending on noise_timestep)
        axs[2, i].imshow(np.clip(reconstructed_np[i], 0, 1))
        if noise_timestep > 0:
            axs[2, i].set_title(f"VAE + Diffusion\n(t={noise_timestep}→0)", fontsize=8)
        else:
            axs[2, i].set_title(f"Clean Latents\n(t=0, no noise)", fontsize=8)
        axs[2, i].axis('off')
    
    # Add row labels
    axs[0, 0].text(-0.5, 0.5, 'Original\nImages', transform=axs[0, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    axs[1, 0].text(-0.5, 0.5, 'VAE\nReconstruction', transform=axs[1, 0].transAxes,
                   fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    if noise_timestep > 0:
        axs[2, 0].text(-0.5, 0.5, 'Diffusion\nReconstruction', transform=axs[2, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    else:
        axs[2, 0].text(-0.5, 0.5, 'Clean Latents\n(no diffusion)', transform=axs[2, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right', rotation=0)
    
    if noise_timestep > 0:
        title = f"Reconstruction Comparison: Dataset → Latents → Denoising\n(Started from noisy latents at timestep {noise_timestep})"
    else:
        title = f"Reconstruction Comparison: Dataset → Latents (No Noise Added)\n(Pure VAE encode/decode test - timestep {noise_timestep})"
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    grid_path = comparison_dir / f"reconstruction_comparison_t{noise_timestep}.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved comparison grid: {grid_path}")
    
    # Also save individual comparisons
    for i in range(bsz):
        class_name = CIFAR10WithCaptions.CLASS_NAMES[labels_list[i]]
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        axs[0].imshow(np.clip(original_np[i], 0, 1))
        axs[0].set_title(f"Original: {class_name}", fontsize=12, fontweight='bold')
        axs[0].axis('off')
        
        axs[1].imshow(np.clip(vae_reconstructed_np[i], 0, 1))
        axs[1].set_title("VAE Reconstruction\n(no diffusion)", fontsize=12, fontweight='bold')
        axs[1].axis('off')
        
        axs[2].imshow(np.clip(reconstructed_np[i], 0, 1))
        if noise_timestep > 0:
            axs[2].set_title(f"Diffusion Reconstruction\n(from t={noise_timestep})", fontsize=12, fontweight='bold')
        else:
            axs[2].set_title(f"Clean Latents\n(t=0, no noise)", fontsize=12, fontweight='bold')
        axs[2].axis('off')
        
        plt.tight_layout()
        individual_path = comparison_dir / f"reconstruction_{class_name}_t{noise_timestep}.png"
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✅ Saved {bsz} individual comparison images")
    
    print(f"\n{'='*80}")
    print(f"✅ RECONSTRUCTION COMPLETE")
    print(f"   Comparison grid: {grid_path}")
    print(f"   Individual comparisons: {comparison_dir}")
    print(f"\n📊 Summary:")
    print(f"   - Original images: Real CIFAR-10 samples")
    print(f"   - VAE Reconstruction: Encode → Decode (tests VAE quality)")
    if noise_timestep > 0:
        print(f"   - Diffusion Reconstruction: Encode → Add noise (t={noise_timestep}) → Denoise → Decode")
        print(f"   - This shows how well the UNet can recover images from the learned latent distribution")
    else:
        print(f"   - Clean Latents: Encode → Decode (t=0, no noise or diffusion)")
        print(f"   - This is identical to VAE Reconstruction (tests that latents pass through correctly)")
    print(f"{'='*80}\n")


def infer(unet: nn.Module, vae: nn.Module, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, scheduler: DDPMScheduler, device: torch.device):
    unet.eval()
    
    # Use all CIFAR-10 class captions for inference
    dataset_class_names = CIFAR10WithCaptions.CLASS_NAMES
    prompts = [f"a photo of a {class_name}" for class_name in dataset_class_names]
    
    print(f"🖼️  Generating images for all {len(prompts)} CIFAR-10 classes...")
    print(f"Classes: {dataset_class_names}")
    print(f"Expected output size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])[0]

    # start from pure noise in latent space and denoise using the trained UNet
    bsz = len(prompts)
    latent_shape = (bsz, 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4)  # assume VAE downsamples by 4
    latents = torch.randn(latent_shape, device=device)
    
    print(f"Initial latents shape: {latents.shape}")

    scheduler.set_timesteps(50)  # number of denoising steps during inference
    timesteps = scheduler.timesteps

    for t in timesteps:
        t_batch = torch.tensor([int(t)] * bsz, device=device)
        with torch.no_grad():
            pred_noise = unet(latents, t_batch, text_embeddings)
        # predict previous noisy sample -> simple DDPM step
        latents = scheduler.step(pred_noise, t, latents).prev_sample

    print(f"Final latents shape: {latents.shape}")

    # decode latents to images
    with torch.no_grad():
        try:
            decoded = vae.decode(latents / getattr(vae.config, 'scaling_factor', 1.0)).sample
            print(f"Decoded images shape: {decoded.shape}")
        except Exception:
            # tiny VAE path
            decoded = vae.decode(latents)
            print(f"Decoded images shape (no scaling): {decoded.shape}")
        
        images = decoded

    # images are in [-1,1], convert to [0,1]
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    print(f"Final numpy images shape: {images.shape}")
    print(f"Individual image size: {images.shape[1]}x{images.shape[2]}")
    
    if images.shape[1] != IMAGE_SIZE or images.shape[2] != IMAGE_SIZE:
        print(f"\n⚠️  WARNING: Output image size is {images.shape[1]}x{images.shape[2]}, expected {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"This mismatch is caused by the SDXL VAE which has fixed upsampling behavior.")
        print(f"The VAE expects specific input dimensions and may produce unexpected output sizes for 32x32 inputs.")

    # Save individual images for each class
    for i, (img, prompt, class_name) in enumerate(zip(images, prompts, dataset_class_names)):
        img_filename = f"infer_final_{class_name}.png"
        img_path = EXAMPLES_DIR / img_filename
        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(f"Final Inference: {prompt}")
        plt.tight_layout()
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {img_filename}")

    # Create a comprehensive grid (2 rows of 5 images each for 10 classes)
    cols = 5
    rows = 2
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    for i, (img, prompt, class_name) in enumerate(zip(images, prompts, dataset_class_names)):
        row = i // cols
        col = i % cols
        axs[row, col].imshow(np.clip(img, 0, 1))
        axs[row, col].set_title(class_name, fontsize=10, fontweight='bold')
        axs[row, col].axis('off')
    
    out_path = EXAMPLES_DIR / "infer_final_all_classes_grid.png"
    plt.suptitle("Final Inference: All CIFAR-10 Classes", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎯 Generated images for all {len(prompts)} CIFAR-10 classes")
    print(f"Individual images and grid saved to {EXAMPLES_DIR}")
    print(f"Final grid: {out_path}")


if __name__ == '__main__':
    try:
        train()
    finally:
        # Restore stdout/stderr and close log files
        sys.stdout = stdout_logger.terminal
        sys.stderr = stderr_logger.terminal
        stdout_logger.close()
        stderr_logger.close()
        print(f"\n{'='*80}")
        print(f"✅ Training log saved to: {LOG_FILE}")
        print(f"{'='*80}")
