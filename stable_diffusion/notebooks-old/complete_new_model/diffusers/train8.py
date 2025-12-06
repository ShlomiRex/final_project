"""
Train a conditional UNet2D (Diffusers) on MNIST captions using a pretrained VAE (encoder/decoder)
and CLIP text encoder, then sample images from text prompts.

Key design choices
- Use AutoencoderKL from Stable Diffusion to encode images to latents (in_channels=4).
- Upscale MNIST (28x28, grayscale) to 256x256, 3 channels, normalized to [-1,1] for VAE.
- Conditional UNet2DConditionModel with cross-attention on CLIP text embeddings (dim=512).
- DDPM noise schedule for training and inference. Classifier-free guidance for sampling.

Notes
- Training on MNIST is fast but small; model architecture is kept light to fit GPU/CPU.
- The pretrained VAE and CLIP are frozen. Only the UNet is trained.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler
from diffusers.models import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from transformers import CLIPTextModel, CLIPTokenizer

def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def build_transforms(image_size: int = 256) -> transforms.Compose:
	# MNIST -> 3x256x256 in [-1,1]
	return transforms.Compose(
		[
			transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


@dataclass
class TrainConfig:
	dataset_root: str
	output_dir: str = "./outputs/train8"
	batch_size: int = 64
	num_epochs: int = 1
	lr: float = 1e-4
	num_train_timesteps: int = 1000
	image_size: int = 256
	tokenizer_max_length: int = 16
	cfg_dropout_p: float = 0.1  # classifier-free guidance dropout during training
	seed: int = 42
	mixed_precision: bool = False
	# UNet size: we keep it small for MNIST
	unet_block_out_channels: tuple[int, ...] = (128, 256, 256)
	layers_per_block: int = 2


def create_models(device: torch.device, config: TrainConfig):
	# Pretrained VAE and CLIP text encoder/tokenizer
	vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
	vae.requires_grad_(False)
	vae.eval()
	vae.to(device)

	text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
	tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
	text_encoder.requires_grad_(False)
	text_encoder.eval()
	text_encoder.to(device)

	# Conditional UNet operating in latent space (4 channels), size 32x32 for 256x256 images
	unet = UNet2DConditionModel(
		sample_size=config.image_size // 8,  # 32 for 256x256
		in_channels=4,
		out_channels=4,
		layers_per_block=config.layers_per_block,
		block_out_channels=config.unet_block_out_channels,
		down_block_types=(
			"DownBlock2D",
			"CrossAttnDownBlock2D",
			"DownBlock2D",
		),
		up_block_types=(
			"UpBlock2D",
			"CrossAttnUpBlock2D",
			"UpBlock2D",
		),
		cross_attention_dim=512,  # CLIP ViT-B/32 hidden size
	).to(device)

	return vae, tokenizer, text_encoder, unet


def make_dataloader(config: TrainConfig) -> DataLoader:
	tfms = build_transforms(config.image_size)
	ds = datasets.MNIST(root=config.dataset_root, train=True, download=True, transform=tfms)
	return DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)


def seed_everything(seed: int):
	torch.manual_seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def train(config: TrainConfig):
	os.makedirs(config.output_dir, exist_ok=True)
	seed_everything(config.seed)
	device = get_device()

	vae, tokenizer, text_encoder, unet = create_models(device, config)

	# Noise scheduler for training
	noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule="squaredcos_cap_v2")

	optimizer = torch.optim.AdamW(unet.parameters(), lr=config.lr)

	dataloader = make_dataloader(config)

	scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")

	# Loss tracking
	batch_losses = []
	epoch_losses = []

	global_step = 0
	unet.train()
	for epoch in range(config.num_epochs):
		epoch_loss_sum = 0.0
		epoch_batch_count = 0
		pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
		for images, labels in pbar:
			images = images.to(device, non_blocking=True)

			# Create text captions from labels
			captions: List[str] = [f"A handwritten digit {int(l)}" for l in labels]

			# Tokenize (with classifier-free guidance dropout during training)
			if np.random.rand() < config.cfg_dropout_p:
				captions_input = [""] * len(captions)
			else:
				captions_input = captions

			text_inputs = tokenizer(
				captions_input,
				padding="max_length",
				max_length=config.tokenizer_max_length,
				truncation=True,
				return_tensors="pt",
			)
			text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

			with torch.no_grad():
				# Encode images to latents using frozen VAE
				latents = vae.encode(images).latent_dist.sample() * 0.18215
				# Prepare text embeddings
				text_embeddings = text_encoder(text_inputs["input_ids"]).last_hidden_state

			# Sample noise and timestep; add noise
			noise = torch.randn_like(latents)
			bsz = latents.shape[0]
			timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
			noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

			# Predict noise
			with torch.autocast(device_type=device.type, dtype=torch.float16 if (config.mixed_precision and device.type == "cuda") else torch.float32, enabled=config.mixed_precision):
				noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
				loss = nn.functional.mse_loss(noise_pred, noise)

			optimizer.zero_grad(set_to_none=True)
			if scaler.is_enabled():
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				loss.backward()
				optimizer.step()

			# Record loss
			loss_value = loss.item()
			batch_losses.append(loss_value)
			epoch_loss_sum += loss_value
			epoch_batch_count += 1

			global_step += 1
			pbar.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})

			# Save periodic checkpoints
			if global_step % 500 == 0:
				ckpt_path = os.path.join(config.output_dir, f"unet_step_{global_step}.pt")
				torch.save({"unet": unet.state_dict(), "step": global_step}, ckpt_path)

		# Record epoch average loss
		avg_epoch_loss = epoch_loss_sum / epoch_batch_count if epoch_batch_count > 0 else 0.0
		epoch_losses.append(avg_epoch_loss)
		print(f"Epoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")

		# Save per-epoch checkpoint
		ckpt_path = os.path.join(config.output_dir, f"unet_epoch_{epoch+1}.pt")
		torch.save({"unet": unet.state_dict(), "epoch": epoch + 1, "step": global_step}, ckpt_path)

	# Save final
	final_path = os.path.join(config.output_dir, "unet_final.pt")
	torch.save(unet.state_dict(), final_path)

	# Plot and save loss graphs
	plot_losses(batch_losses, epoch_losses, config.output_dir)


def plot_losses(batch_losses: List[float], epoch_losses: List[float], output_dir: str):
	"""Generate and save loss plots for per-batch and per-epoch losses."""
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))

	# Per-batch loss
	axes[0].plot(batch_losses, linewidth=0.8, alpha=0.7)
	axes[0].set_xlabel("Batch")
	axes[0].set_ylabel("Loss")
	axes[0].set_title("Training Loss per Batch")
	axes[0].grid(True, alpha=0.3)

	# Per-epoch loss
	if len(epoch_losses) > 0:
		axes[1].plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linewidth=2)
		axes[1].set_xlabel("Epoch")
		axes[1].set_ylabel("Average Loss")
		axes[1].set_title("Training Loss per Epoch")
		axes[1].grid(True, alpha=0.3)

	plt.tight_layout()
	loss_plot_path = os.path.join(output_dir, "training_loss.png")
	plt.savefig(loss_plot_path, dpi=150)
	plt.close()
	print(f"Loss plots saved to {loss_plot_path}")


@torch.no_grad()
def sample(
	prompt: str,
	output_path: Optional[str],
	guidance_scale: float,
	num_inference_steps: int,
	image_size: int,
	device: torch.device,
	vae: AutoencoderKL,
	tokenizer: CLIPTokenizer,
	text_encoder: CLIPTextModel,
	unet: UNet2DConditionModel,
	seed: Optional[int] = None,
):
	if seed is not None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	scheduler = DDPMScheduler(beta_schedule="squaredcos_cap_v2", num_train_timesteps=1000)
	scheduler.set_timesteps(num_inference_steps)

	# Get conditioned and unconditioned text embeddings for classifier-free guidance
	text_inputs = tokenizer(
		[prompt], padding="max_length", max_length=16, truncation=True, return_tensors="pt"
	).to(device)
	text_embeddings = text_encoder(text_inputs.input_ids).last_hidden_state

	uncond_inputs = tokenizer([""], padding="max_length", max_length=16, return_tensors="pt").to(device)
	uncond_embeddings = text_encoder(uncond_inputs.input_ids).last_hidden_state

	encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0)

	# Init random latents in latent space
	latents = torch.randn((1, 4, image_size // 8, image_size // 8), device=device)

	for t in tqdm(scheduler.timesteps, desc="Sampling"):
		# Expand for classifier-free guidance
		latent_model_input = torch.cat([latents] * 2, dim=0)
		latent_model_input = scheduler.scale_model_input(latent_model_input, t)

		# Predict noise
		noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
		noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
		noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

		# Step
		latents = scheduler.step(noise_pred, t, latents).prev_sample

	# Decode latents to image
	latents = latents / 0.18215
	image = vae.decode(latents).sample
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.detach().cpu()

	if output_path:
		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
		# save as PNG via torchvision
		from torchvision.utils import save_image

		save_image(image, output_path)

	return image


def load_for_sampling(
	output_dir: str,
	image_size: int,
	device: torch.device,
	config: TrainConfig,
	ckpt: Optional[str] = None,
):
	vae, tokenizer, text_encoder, unet = create_models(device, config)

	# Load the most recent checkpoint or provided path
	state_dict = None
	if ckpt is not None and os.path.isfile(ckpt):
		state = torch.load(ckpt, map_location=device)
		state_dict = state.get("unet", state)
	else:
		# Try final, epoch, or latest step
		candidates = [
			os.path.join(output_dir, "unet_final.pt"),
		]
		# Add epoch files if present
		for f in sorted(os.listdir(output_dir)):
			if f.startswith("unet_epoch_") and f.endswith(".pt"):
				candidates.append(os.path.join(output_dir, f))
		# Add step files
		for f in sorted(os.listdir(output_dir)):
			if f.startswith("unet_step_") and f.endswith(".pt"):
				candidates.append(os.path.join(output_dir, f))

		for path in candidates:
			if os.path.isfile(path):
				state = torch.load(path, map_location=device)
				state_dict = state.get("unet", state)
				break

	if state_dict is None:
		raise FileNotFoundError("No UNet checkpoint found for sampling. Provide --ckpt or train first.")

	unet.load_state_dict(state_dict, strict=True)
	unet.eval()
	return vae, tokenizer, text_encoder, unet


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train/sample conditional UNet on MNIST with VAE+CLIP")

	# Common
	parser.add_argument("--output_dir", type=str, default="./outputs/train8")
	parser.add_argument("--dataset_root", type=str, default=os.path.normpath(os.path.join(os.path.dirname(__file__), "../../datasets")))
	parser.add_argument("--image_size", type=int, default=256)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--mixed_precision", action="store_true")

	# Train
	parser.add_argument("--train", action="store_true", help="Run training")
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--cfg_dropout_p", type=float, default=0.1, help="Probability to drop text during training")

	# Sample
	parser.add_argument("--sample", action="store_true", help="Run sampling")
	parser.add_argument("--prompt", type=str, default="A handwritten digit 3")
	parser.add_argument("--guidance_scale", type=float, default=7.5)
	parser.add_argument("--inference_steps", type=int, default=50)
	parser.add_argument("--save", type=str, default="./outputs/train8/sample.png")
	parser.add_argument("--ckpt", type=str, default=None, help="Path to UNet checkpoint to load for sampling")

	return parser.parse_args()


def main():
	args = parse_args()

	if not args.train and not args.sample:
		# Default to training if neither specified
		args.train = True

	config = TrainConfig(
		dataset_root=args.dataset_root,
		output_dir=args.output_dir,
		batch_size=args.batch_size,
		num_epochs=args.epochs,
		lr=args.lr,
		image_size=args.image_size,
		cfg_dropout_p=args.cfg_dropout_p,
		mixed_precision=args.mixed_precision,
	)

	if args.train:
		train(config)

	if args.sample:
		device = get_device()
		vae, tokenizer, text_encoder, unet = load_for_sampling(
			output_dir=args.output_dir, image_size=args.image_size, device=device, config=config, ckpt=args.ckpt
		)
		_ = sample(
			prompt=args.prompt,
			output_path=args.save,
			guidance_scale=args.guidance_scale,
			num_inference_steps=args.inference_steps,
			image_size=args.image_size,
			device=device,
			vae=vae,
			tokenizer=tokenizer,
			text_encoder=text_encoder,
			unet=unet,
			seed=args.seed,
		)


if __name__ == "__main__":
	main()

