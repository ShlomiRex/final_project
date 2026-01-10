#!/usr/bin/env python3
"""
Evaluation script for Stable Diffusion.

Runs comprehensive evaluation on a trained model:
- Generate samples from validation prompts
- Compute FID score
- Compute CLIP score

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/checkpoint_100000.pt \
        --config configs/base.yaml \
        --output_dir evaluation_results
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.models.unet import UNet2DConditionModel, UNetConfig
from src.models.vae import VAEWrapper
from src.models.text_encoder import CLIPTextEncoderWrapper
from src.diffusion.noise_scheduler import DDPMScheduler, DDIMScheduler, SchedulerConfig
from src.diffusion.sampler import CFGSampler
from src.training.ema import EMAModel
from src.evaluation.evaluator import Evaluator
from src.evaluation.sample_generator import SampleGenerator, create_image_grid
from src.evaluation.fid import FIDCalculator
from src.evaluation.clip_score import CLIPScoreCalculator
from src.data.dataset import get_dataloader
from src.utils.config import load_config, parse_cli_overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion")
    
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
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples for FID calculation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
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
        "--use_ema",
        action="store_true",
        help="Use EMA weights if available",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to text file with prompts (one per line)",
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
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


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
    
    # Build scheduler (use DDIM for faster inference)
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
    
    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts for evaluation
        prompts = [
            "A photo of a cat sitting on a windowsill",
            "A beautiful sunset over the ocean",
            "A mountain landscape with snow peaks",
            "A colorful flower garden in spring",
            "A city skyline at night with lights",
            "A portrait of a person smiling",
            "An abstract painting with vibrant colors",
            "A forest path in autumn",
        ]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Generate sample grid
    print("Generating sample grid...")
    sample_images = []
    sample_prompts = prompts[:16]  # Use first 16 prompts for grid
    
    data_config = config.get("data", {})
    resolution = data_config.get("resolution", 256)
    
    for prompt in tqdm(sample_prompts, desc="Generating samples"):
        # Encode text
        text_emb = text_encoder.encode([prompt])
        
        # Generate
        with torch.no_grad():
            images = sampler.sample(
                prompt_embeds=text_emb,
                height=resolution,
                width=resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        
        sample_images.extend(images)
    
    # Save sample grid
    grid = create_image_grid(sample_images, cols=4)
    grid.save(output_dir / "sample_grid.png")
    print(f"Saved sample grid to {output_dir / 'sample_grid.png'}")
    
    # Save individual samples with prompts
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    for i, (img, prompt) in enumerate(zip(sample_images, sample_prompts)):
        img.save(samples_dir / f"sample_{i:03d}.png")
        with open(samples_dir / f"sample_{i:03d}.txt", "w") as f:
            f.write(prompt)
    
    # Calculate FID score
    print(f"\nCalculating FID score with {args.num_samples} samples...")
    fid_calculator = FIDCalculator(device=device)
    
    # Generate samples for FID
    generated_images = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating for FID"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.num_samples)
        batch_prompts = [prompts[i % len(prompts)] for i in range(start_idx, end_idx)]
        
        # Encode text
        text_emb = text_encoder.encode(batch_prompts)
        
        # Generate
        with torch.no_grad():
            images = sampler.sample(
                prompt_embeds=text_emb,
                height=resolution,
                width=resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
        
        generated_images.extend(images)
    
    # Get real images for FID
    print("Loading real images for FID comparison...")
    real_dataloader = get_dataloader(
        config=data_config,
        split="test",
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    real_images = []
    for batch in tqdm(real_dataloader, desc="Loading real images"):
        if len(real_images) >= args.num_samples:
            break
        
        # Convert tensors to PIL images
        images = batch["image"]
        for i in range(images.shape[0]):
            if len(real_images) >= args.num_samples:
                break
            img_tensor = images[i]
            # Denormalize
            img_tensor = (img_tensor + 1) / 2
            img_tensor = img_tensor.clamp(0, 1)
            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            real_images.append(img_array)
    
    # Calculate FID
    fid_score = fid_calculator.calculate_fid_from_images(
        real_images=real_images,
        generated_images=generated_images,
    )
    print(f"FID Score: {fid_score:.4f}")
    
    # Calculate CLIP score
    print("\nCalculating CLIP score...")
    clip_calculator = CLIPScoreCalculator(device=device)
    
    # Use the same prompts as generated images
    eval_prompts = [prompts[i % len(prompts)] for i in range(len(generated_images))]
    clip_score = clip_calculator.calculate_clip_score(
        images=generated_images,
        prompts=eval_prompts,
    )
    print(f"CLIP Score: {clip_score:.4f}")
    
    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": resolution,
        "use_ema": args.use_ema,
        "fid_score": fid_score,
        "clip_score": clip_score,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")
    print("\n=== Summary ===")
    print(f"FID Score: {fid_score:.4f}")
    print(f"CLIP Score: {clip_score:.4f}")


if __name__ == "__main__":
    main()
