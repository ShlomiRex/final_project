#!/usr/bin/env python3
"""
Image generation script for Stable Diffusion.

Generate images from text prompts using a trained model.

Usage:
    # Single prompt
    python scripts/generate.py --checkpoint outputs/checkpoints/checkpoint_100000.pt \
        --config configs/base.yaml \
        --prompt "A photo of a cat"
    
    # Multiple prompts from file
    python scripts/generate.py --checkpoint outputs/checkpoints/checkpoint_100000.pt \
        --config configs/base.yaml \
        --prompts_file prompts.txt \
        --num_images_per_prompt 4
    
    # Interactive mode
    python scripts/generate.py --checkpoint outputs/checkpoints/checkpoint_100000.pt \
        --config configs/base.yaml \
        --interactive
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from src.models.unet import UNet2DConditionModel, UNetConfig
from src.models.vae import VAEWrapper
from src.models.text_encoder import CLIPTextEncoderWrapper
from src.diffusion.noise_scheduler import DDIMScheduler, SchedulerConfig
from src.diffusion.sampler import CFGSampler
from src.evaluation.sample_generator import create_image_grid
from src.utils.config import load_config, parse_cli_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Output directory",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of DDIM steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Image resolution (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights if available",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for continuous generation",
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate denoising steps",
    )
    
    args, unknown = parser.parse_known_args()
    args.overrides = parse_cli_overrides(unknown)
    
    return args


def load_model(checkpoint_path: str, config: dict, device: torch.device, use_ema: bool = False):
    """Load model from checkpoint."""
    
    # Build model
    model_config = config.get("model", {})
    
    unet_config = UNetConfig(
        in_channels=model_config.get("in_channels", 4),
        out_channels=model_config.get("out_channels", 4),
        block_out_channels=model_config.get("block_out_channels", [320, 640, 1280, 1280]),
        down_block_types=model_config.get("down_block_types"),
        up_block_types=model_config.get("up_block_types"),
        layers_per_block=model_config.get("layers_per_block", 2),
        attention_head_dim=model_config.get("attention_head_dim", 8),
        cross_attention_dim=model_config.get("cross_attention_dim", 768),
    )
    
    model = UNet2DConditionModel(unet_config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if use_ema and "ema_state_dict" in checkpoint:
        print("Loading EMA weights")
        model.load_state_dict(checkpoint["ema_state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def generate_images(
    sampler: CFGSampler,
    text_encoder: CLIPTextEncoderWrapper,
    prompts: list,
    num_images_per_prompt: int,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = None,
    save_intermediate: bool = False,
):
    """Generate images for prompts."""
    
    all_images = []
    all_intermediates = []
    
    for prompt in prompts:
        # Encode text
        text_emb = text_encoder.encode([prompt] * num_images_per_prompt)
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate
        with torch.no_grad():
            if save_intermediate:
                images, intermediates = sampler.sample_with_intermediate(
                    prompt_embeds=text_emb,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                all_intermediates.append(intermediates)
            else:
                images = sampler.sample(
                    prompt_embeds=text_emb,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
        
        all_images.extend(images)
    
    return all_images, all_intermediates if save_intermediate else None


def interactive_mode(
    sampler: CFGSampler,
    text_encoder: CLIPTextEncoderWrapper,
    resolution: int,
    num_inference_steps: int,
    guidance_scale: float,
    output_dir: Path,
):
    """Run in interactive mode."""
    
    print("\n=== Interactive Mode ===")
    print("Enter prompts to generate images. Type 'quit' to exit.")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print()
    
    image_count = 0
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        
        if not prompt:
            continue
        
        print("Generating...")
        
        # Generate
        images, _ = generate_images(
            sampler=sampler,
            text_encoder=text_encoder,
            prompts=[prompt],
            num_images_per_prompt=1,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Save
        for img in images:
            save_path = output_dir / f"image_{image_count:04d}.png"
            img.save(save_path)
            print(f"Saved: {save_path}")
            image_count += 1
        
        # Also save prompt
        with open(output_dir / f"image_{image_count-1:04d}.txt", "w") as f:
            f.write(prompt)
    
    print(f"\nGenerated {image_count} images. Goodbye!")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, overrides=args.overrides)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device, use_ema=args.use_ema)
    
    # Load VAE and text encoder
    vae_config = config.get("vae", {})
    vae = VAEWrapper(
        model_name=vae_config.get("pretrained_model", "stabilityai/sd-vae-ft-mse"),
        device=device,
    )
    
    text_encoder_config = config.get("text_encoder", {})
    text_encoder = CLIPTextEncoderWrapper(
        model_name=text_encoder_config.get("pretrained_model", "openai/clip-vit-large-patch14"),
        device=device,
    )
    
    # Build scheduler
    diffusion_config = config.get("diffusion", {})
    scheduler_config = SchedulerConfig(
        num_train_timesteps=diffusion_config.get("num_train_timesteps", 1000),
        beta_start=diffusion_config.get("beta_start", 0.00085),
        beta_end=diffusion_config.get("beta_end", 0.012),
        beta_schedule=diffusion_config.get("beta_schedule", "scaled_linear"),
        prediction_type=diffusion_config.get("prediction_type", "epsilon"),
    )
    scheduler = DDIMScheduler(scheduler_config)
    
    # Create sampler
    sampler = CFGSampler(
        unet=model,
        scheduler=scheduler,
        vae=vae,
    )
    
    # Get resolution
    data_config = config.get("data", {})
    resolution = args.resolution or data_config.get("resolution", 256)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(
            sampler=sampler,
            text_encoder=text_encoder,
            resolution=resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_dir=output_dir,
        )
        return
    
    # Get prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        print("Error: Please provide --prompt or --prompts_file")
        sys.exit(1)
    
    print(f"Generating {len(prompts)} prompts × {args.num_images_per_prompt} images each")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Steps: {args.num_inference_steps}, Guidance: {args.guidance_scale}")
    
    # Generate
    images, intermediates = generate_images(
        sampler=sampler,
        text_encoder=text_encoder,
        prompts=prompts,
        num_images_per_prompt=args.num_images_per_prompt,
        resolution=resolution,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_intermediate=args.save_intermediate,
    )
    
    # Save images
    for i, img in enumerate(images):
        prompt_idx = i // args.num_images_per_prompt
        img_idx = i % args.num_images_per_prompt
        
        save_path = output_dir / f"prompt_{prompt_idx:03d}_image_{img_idx:02d}.png"
        img.save(save_path)
        
        # Save prompt
        with open(output_dir / f"prompt_{prompt_idx:03d}.txt", "w") as f:
            f.write(prompts[prompt_idx])
    
    print(f"Saved {len(images)} images to {output_dir}")
    
    # Save grid if multiple images
    if len(images) > 1:
        grid = create_image_grid(images)
        grid.save(output_dir / "grid.png")
        print(f"Saved grid to {output_dir / 'grid.png'}")
    
    # Save intermediate steps if requested
    if intermediates:
        intermediates_dir = output_dir / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
        
        for prompt_idx, steps in enumerate(intermediates):
            # Create progress grid
            progress_grid = create_image_grid(steps)
            progress_grid.save(intermediates_dir / f"prompt_{prompt_idx:03d}_progress.png")
        
        print(f"Saved intermediate steps to {intermediates_dir}")


if __name__ == "__main__":
    main()
