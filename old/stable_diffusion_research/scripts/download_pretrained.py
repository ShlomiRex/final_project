#!/usr/bin/env python3
"""
Download pretrained models for Stable Diffusion training.

Downloads:
- VAE: stabilityai/sd-vae-ft-mse
- Text Encoder: openai/clip-vit-large-patch14

Usage:
    python scripts/download_pretrained.py --output_dir pretrained_models
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pretrained_models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE model to download",
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Text encoder model to download",
    )
    
    return parser.parse_args()


def download_vae(model_name: str, output_dir: Path):
    """Download VAE model."""
    print(f"Downloading VAE: {model_name}")
    
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(model_name)
    
    save_path = output_dir / "vae"
    vae.save_pretrained(save_path)
    
    print(f"Saved VAE to {save_path}")
    
    return save_path


def download_text_encoder(model_name: str, output_dir: Path):
    """Download text encoder model."""
    print(f"Downloading text encoder: {model_name}")
    
    from transformers import CLIPTextModel, CLIPTokenizer
    
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name)
    
    save_path = output_dir / "text_encoder"
    tokenizer.save_pretrained(save_path)
    text_encoder.save_pretrained(save_path)
    
    print(f"Saved text encoder to {save_path}")
    
    return save_path


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Downloading Pretrained Models ===")
    print()
    
    # Download VAE
    vae_path = download_vae(args.vae_model, output_dir)
    print()
    
    # Download text encoder
    text_encoder_path = download_text_encoder(args.text_encoder_model, output_dir)
    print()
    
    print("=== Download Complete ===")
    print(f"VAE: {vae_path}")
    print(f"Text Encoder: {text_encoder_path}")
    print()
    print("Update your config to use local paths:")
    print(f'  vae.pretrained_model: "{vae_path}"')
    print(f'  text_encoder.pretrained_model: "{text_encoder_path}"')


if __name__ == "__main__":
    main()
