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
CHECKPOINT_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = ROOT_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# ------------------ Configuration Variables ------------------
SEED = 42
IMAGE_SIZE = 32  # resize CIFAR10
BATCH_SIZE = 256
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
GRAD_ACCUM_STEPS = 1
MAX_TRAIN_STEPS = 200_000  # small demo cap
SAVE_CHECKPOINT_EVERY_EPOCHS = 5  # Save checkpoint every 5 epochs
GENERATE_IMAGES_EVERY_EPOCHS = 1  # Generate images every 1 epoch
NUM_TRAIN_TIMESTEPS = 1000

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


# ------------------ Simple Cross-Attention Block ------------------
class CrossAttention(nn.Module):
    """A minimal cross-attention block: queries from spatial features, keys/values from text embeddings."""

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
        b, c, h, w = x.shape
        q = self.to_q(x)  # (B, C, H, W)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)  # (B, heads, head_dim, N)
        q = q.permute(0, 1, 3, 2)  # (B, heads, N, head_dim)

        # text_embeds: (B, T, D) -> collapse T by mean pooling
        if text_embeds.dim() == 3:
            k_v_src = text_embeds.mean(dim=1)  # (B, D)
        else:
            k_v_src = text_embeds
        k = self.to_k(k_v_src).reshape(b, self.num_heads, self.head_dim).unsqueeze(2)  # (B, heads, 1, head_dim)
        v = self.to_v(k_v_src).reshape(b, self.num_heads, self.head_dim).unsqueeze(2)  # (B, heads, 1, head_dim)

        # attention: q @ k^T -> (B, heads, N, 1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, 1)
        if attention_mask is not None:
            # attention_mask: (B, T) -> reduce to (B,1,1) after mean, applied as multiplier
            mask_val = attention_mask.float().mean(dim=1).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask_val == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, c, h, w)
        out = self.out(out)
        return out


# ------------------ Small UNet built from scratch ------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(h + self.nin_shortcut(x))


class SimpleUNet(nn.Module):
    """A lightweight conditional UNet with cross-attention to text embeddings."""

    def __init__(self, in_channels=4, base_channels=64, channel_mults=(1, 2, 4), text_dim=512):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # down blocks
        downs = []
        ch = base_channels
        self.down_blocks = nn.ModuleList()
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleDict({
                'res1': ResidualBlock(ch, out_ch),
                'attn': CrossAttention(out_ch, text_dim, num_heads=4),
                'res2': ResidualBlock(out_ch, out_ch),
                'downsample': nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            }))
            ch = out_ch

        # bottleneck
        self.mid = nn.ModuleDict({
            'res1': ResidualBlock(ch, ch),
            'attn': CrossAttention(ch, text_dim, num_heads=4),
            'res2': ResidualBlock(ch, ch)
        })

        # up blocks
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_blocks.append(nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                'res1': ResidualBlock(out_ch * 2, out_ch),  # *2 because of skip connection concatenation
                'attn': CrossAttention(out_ch, text_dim, num_heads=4),
                'res2': ResidualBlock(out_ch, out_ch),
            }))
            ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.ReLU(),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps, text_embeds):
        # x: (B, C, H, W)
        # store skips
        h = self.in_conv(x)
        skips = []
        for block in self.down_blocks:
            h = block['res1'](h)
            h = block['attn'](h, text_embeds)
            h = block['res2'](h)
            skips.append(h)
            h = block['downsample'](h)

        # mid
        h = self.mid['res1'](h)
        h = self.mid['attn'](h, text_embeds)
        h = self.mid['res2'](h)

        # up
        for block in self.up_blocks:
            h = block['upsample'](h)
            skip = skips.pop()
            # concatenate along channels
            h = torch.cat([h, skip], dim=1)
            h = block['res1'](h)
            h = block['attn'](h, text_embeds)
            h = block['res2'](h)

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
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
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

def generate_sample_images(unet, vae, tokenizer, text_encoder, noise_scheduler, global_step, epoch, device):
    """Generate sample images without saving checkpoint."""
    # Create images directory
    images_dir = EXAMPLES_DIR / f"epoch_{epoch}_step_{global_step}"
    images_dir.mkdir(exist_ok=True)
    
    # Generate sample images
    unet.eval()
    
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
    
    # Decode to images
    with torch.no_grad():
        try:
            images = vae.decode(latents / getattr(vae.config, 'scaling_factor', 1.0)).sample
        except Exception:
            images = vae.decode(latents)
    
    # Convert to [0,1] range
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    # Save individual images
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        img_filename = f"epoch_{epoch}_step_{global_step}_sample_{i}_{prompt.replace(' ', '_').replace(',', '')}.png"
        img_path = images_dir / img_filename
        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        plt.title(f"Epoch {epoch}, Step {global_step}: {prompt}")
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
    
    print(f"🖼️  Generated {len(images)} sample images at epoch {epoch}, step {global_step}")
    print(f"Images saved to {images_dir}")
    
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
        vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)
        print("✅ Loaded pretrained VAE.")
    except Exception as e:
        print("❌ Could not load pretrained VAE, Error:", e)
        exit(1)

    # create scheduler
    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # create UNet from scratch
    latent_channels = 4
    unet = SimpleUNet(in_channels=latent_channels, base_channels=64, channel_mults=(1, 2), text_dim=text_encoder.config.hidden_size)
    print(f"✅ Created UNet with {sum(p.numel() for p in unet.parameters()):,} parameters")

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
        
        for batch in batch_pbar:
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
        if accelerator.is_main_process:
            # Wait for all processes to sync before saving
            accelerator.wait_for_everyone()
            
            # Unwrap models for saving/inference
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_vae = accelerator.unwrap_model(vae)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            
            # Check if we should save checkpoint (every N epochs)
            should_save_checkpoint = (epoch + 1) % SAVE_CHECKPOINT_EVERY_EPOCHS == 0
            
            # Check if we should generate images (every N epochs)
            should_generate_images = (epoch + 1) % GENERATE_IMAGES_EVERY_EPOCHS == 0
            
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
            tqdm.write(f"Epoch {epoch+1} completed - Average loss: {epoch_avg_loss:.6f}")
        
        if global_step >= MAX_TRAIN_STEPS:
            break

    # Final save (only on main process)
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_vae = accelerator.unwrap_model(vae)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        
        save_checkpoint_and_generate_images(
            unwrapped_unet, unwrapped_vae, tokenizer, 
            unwrapped_text_encoder, noise_scheduler, 
            optimizer, global_step, epoch, device
        )
        print("✅ Training finished. Final checkpoint and images saved.")


def infer(unet: nn.Module, vae: nn.Module, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, scheduler: DDPMScheduler):
    unet.eval()
    
    # Use all CIFAR-10 class captions for inference
    dataset_class_names = CIFAR10WithCaptions.CLASS_NAMES
    prompts = [f"a photo of a {class_name}" for class_name in dataset_class_names]
    
    print(f"🖼️  Generating images for all {len(prompts)} CIFAR-10 classes...")
    print(f"Classes: {dataset_class_names}")

    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])[0]

    # start from pure noise in latent space and denoise using the trained UNet
    bsz = len(prompts)
    latent_shape = (bsz, 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4)  # assume VAE downsamples by 4
    latents = torch.randn(latent_shape, device=device)

    scheduler.set_timesteps(50)  # number of denoising steps during inference
    timesteps = scheduler.timesteps

    for t in timesteps:
        t_batch = torch.tensor([int(t)] * bsz, device=device)
        with torch.no_grad():
            pred_noise = unet(latents, t_batch, text_embeddings)
        # predict previous noisy sample -> simple DDPM step
        latents = scheduler.step(pred_noise, t, latents).prev_sample

    # decode latents to images
    with torch.no_grad():
        try:
            images = vae.decode(latents / getattr(vae.config, 'scaling_factor', 1.0)).sample
        except Exception:
            # tiny VAE path
            images = vae.decode(latents)

    # images are in [-1,1], convert to [0,1]
    images = (images.clamp(-1, 1) + 1) / 2
    images = images.cpu().permute(0, 2, 3, 1).numpy()

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
    train()
