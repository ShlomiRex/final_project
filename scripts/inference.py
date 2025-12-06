#!/usr/bin/env python
"""
Inference script for LatentGPT.

Generates images from text prompts using a trained model.

Usage:
    python scripts/inference.py --checkpoint path/to/model --prompt "A dog running"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.models.latent_gpt import LatentGPT
from src.models.clip_encoder import CLIPEncoder
from src.models.vqvae import VQVAEWrapper
from src.data.transforms import denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LatentGPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional, inferred from checkpoint if not provided)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        # Try to find config next to checkpoint
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            config = Config.from_yaml(config_path)
        else:
            print("No config found, using defaults")
            config = Config()
    
    print(f"Loading models...")
    
    # Load VQ-VAE
    vqvae = VQVAEWrapper.from_pretrained(config.vqvae.checkpoint, device=device)
    
    # Load CLIP encoder
    clip_encoder = CLIPEncoder(
        model_name=config.text_encoder.model_name,
        device=device,
    )
    
    # Load transformer
    model = LatentGPT(
        vocab_size=config.vqvae.codebook_size,
        hidden_size=config.transformer.hidden_size,
        num_layers=config.transformer.num_layers,
        num_heads=config.transformer.num_heads,
        max_seq_len=config.transformer.max_seq_len,
        cross_attention_dim=config.text_encoder.hidden_size,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Generating {args.num_samples} images for prompt: '{args.prompt}'")
    
    # Encode prompt
    prompts = [args.prompt] * args.num_samples
    text_embeddings = clip_encoder(prompts)
    
    # Generate tokens
    with torch.no_grad():
        tokens = model.generate(
            cross_context=text_embeddings,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
        )
    
    # Decode tokens to images
    # Reshape to 2D grid
    latent_size = int(tokens.size(1) ** 0.5)
    tokens_2d = tokens.view(-1, latent_size, latent_size)
    
    images = vqvae.decode(tokens_2d)
    
    # Denormalize and convert to PIL
    images = denormalize(images)
    images = images.clamp(0, 1).cpu().numpy()
    images = (images * 255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))  # NCHW -> NHWC
    
    # Create grid
    n = args.num_samples
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    h, w = images.shape[1:3]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
    
    # Save
    output_path = Path(args.output)
    Image.fromarray(grid).save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
