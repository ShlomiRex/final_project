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

import matplotlib.pyplot as plt
import numpy as np

# ------------------ Configuration (all in-code, as requested) ------------------
ROOT_DIR = Path("./sd_scratch_demo")
ROOT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = ROOT_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
IMAGE_SIZE = 128  # resize CIFAR10 to 128x128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
GRAD_ACCUM_STEPS = 1
MAX_TRAIN_STEPS = 500  # small demo cap
SAVE_EVERY = 200
NUM_INFERENCE_IMAGES = 4
NUM_TRAIN_TIMESTEPS = 1000

PRETRAINED_VAE = "stabilityai/sdxl-vae"  # we will try to load a VAE from HF; fallback handled
PRETRAINED_CLIP = "openai/clip-vit-base-patch32"

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

def get_dataloader(batch_size: int, image_size: int):
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = CIFAR10(root=str(ROOT_DIR / "data"), download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dl


# ------------------ Main training and inference ------------------

def train():
    # load components
    print("Device:", DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_CLIP)
    text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_CLIP).to(DEVICE)

    # VAE: try to load a pretrained VAE for latents. If unavailable, create a small placeholder VAE
    try:
        vae = AutoencoderKL.from_pretrained(PRETRAINED_VAE).to(DEVICE)
        vae.eval()
        vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)
        print("Loaded pretrained VAE.")
    except Exception as e:
        print("Could not load pretrained VAE, creating a tiny identity-like VAE. Error:", e)

        class TinyVAE(nn.Module):
            def __init__(self, in_channels=3, latent_channels=4):
                super().__init__()
                self.enc = nn.Conv2d(in_channels, latent_channels, 4, 4)
                self.dec = nn.ConvTranspose2d(latent_channels, in_channels, 4, 4)
                self.latent_scale = 1.0

            def encode(self, x):
                # n, c, h, w -> n, latent, h/4, w/4
                z = self.enc(x)
                class Dummy:
                    def __init__(self, latent):
                        self.latent = latent
                        self.latent_dist = self
                    def sample(self):
                        return self.latent
                return Dummy(z)

            def decode(self, z):
                return self.dec(z)

        vae = TinyVAE(in_channels=3, latent_channels=4).to(DEVICE)
        vae.eval()
        vae_scaling = 1.0

    # create scheduler
    noise_scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear", num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # create UNet from scratch
    # UNet input channels: VAE latent channels (assume 4) -> we'll set to 4 if using AutoencoderKL
    latent_channels = 4
    unet = SimpleUNet(in_channels=latent_channels, base_channels=64, channel_mults=(1, 2), text_dim=text_encoder.config.hidden_size).to(DEVICE)
    print("UNet parameters:", sum(p.numel() for p in unet.parameters()))

    # optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # dataloader
    dataloader = get_dataloader(BATCH_SIZE, IMAGE_SIZE)

    global_step = 0
    losses = []

    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            images, labels = batch
            images = images.to(DEVICE)

            # encode with VAE to latents
            with torch.no_grad():
                # AutoencoderKL.encode returns object with .latent_dist.sample() in HF VAE
                if hasattr(vae, 'encode'):
                    encoded = vae.encode(images.to(dtype=next(vae.parameters()).dtype if any(p.requires_grad for p in vae.parameters()) else images.dtype))
                    if hasattr(encoded, 'latent_dist'):
                        latents = encoded.latent_dist.sample() * vae_scaling
                    else:
                        # tiny VAE path
                        latents = encoded.latent
                else:
                    raise RuntimeError("VAE has no encode().")

            # sample random noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # fake text conditioning: use labels->text (CIFAR classes)
            class_names = [
                'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
            ]
            captions = [f"a photo of a {class_names[int(l)]}" for l in labels]
            tokenized = tokenizer(captions, padding='max_length', truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(DEVICE)
            attention_mask = tokenized['attention_mask'].to(DEVICE)
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

            # predict noise with UNet
            unet.train()
            pred = unet(noisy_latents, timesteps, text_embeddings)

            loss = F.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            global_step += 1

            if global_step % 20 == 0:
                avg = sum(losses[-20:]) / min(len(losses), 20)
                print(f"Epoch {epoch} Step {global_step} | loss: {avg:.6f}")

            if global_step % SAVE_EVERY == 0:
                torch.save({'unet_state_dict': unet.state_dict(), 'optimizer': optimizer.state_dict(), 'step': global_step}, CHECKPOINT_DIR / f"ckpt_{global_step}.pt")
                print("Saved checkpoint at step", global_step)

            if global_step >= MAX_TRAIN_STEPS:
                break
        if global_step >= MAX_TRAIN_STEPS:
            break

    # final save
    torch.save({'unet_state_dict': unet.state_dict(), 'optimizer': optimizer.state_dict(), 'step': global_step}, CHECKPOINT_DIR / f"ckpt_final.pt")
    print("Training finished. Final checkpoint saved.")

    # run a small inference and save images
    infer(unet, vae, tokenizer, text_encoder, noise_scheduler)


def infer(unet: nn.Module, vae: nn.Module, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, scheduler: DDPMScheduler):
    unet.eval()
    prompts = [
        "a photo of a cat",
        "a photo of a truck",
        "a painting of a frog",
        "an artistic photo of a ship",
    ][:NUM_INFERENCE_IMAGES]

    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])[0]

    # start from pure noise in latent space and denoise using the trained UNet
    bsz = len(prompts)
    latent_shape = (bsz, 4, IMAGE_SIZE // 4, IMAGE_SIZE // 4)  # assume VAE downsamples by 4
    latents = torch.randn(latent_shape, device=DEVICE)

    scheduler.set_timesteps(50)  # number of denoising steps during inference
    timesteps = scheduler.timesteps

    for t in timesteps:
        t_batch = torch.tensor([int(t)] * bsz, device=DEVICE)
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

    # save grid
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = np.array(axs).reshape(-1)
    for i, img in enumerate(images):
        axs[i].imshow(np.clip(img, 0, 1))
        axs[i].axis('off')
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    out_path = EXAMPLES_DIR / "infer_grid.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved inference grid to", out_path)


if __name__ == '__main__':
    train()
