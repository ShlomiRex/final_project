"""
Sample Generator for Evaluation.

Generates samples from the diffusion model for visualization
and metric calculation.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class SampleGenerator:
    """
    Generates samples from the diffusion model.
    
    Used during training evaluation to:
    - Generate samples with fixed prompts for progress tracking
    - Generate samples for FID/CLIP score calculation
    - Save sample grids for visualization
    """
    
    def __init__(
        self,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        resolution: int = 256,
    ):
        """
        Args:
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            resolution: Image resolution
        """
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.resolution = resolution
    
    @torch.no_grad()
    def generate(
        self,
        unet: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        prompts: List[str],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = True,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.
        
        Args:
            unet: U-Net model
            vae: VAE model
            text_encoder: Text encoder model
            tokenizer: Tokenizer
            scheduler: Diffusion scheduler
            prompts: List of text prompts
            device: Device to run on
            generator: Random generator for reproducibility
            show_progress: Whether to show progress bar
        
        Returns:
            List of generated PIL Images
        """
        from ..diffusion.sampler import CFGSampler
        
        batch_size = len(prompts)
        latent_size = self.resolution // 8  # VAE downsampling factor
        
        # Encode prompts
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))
        if hasattr(text_embeddings, 'last_hidden_state'):
            text_embeddings = text_embeddings.last_hidden_state
        
        # Get unconditional embeddings for CFG
        uncond_inputs = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))
        if hasattr(uncond_embeddings, 'last_hidden_state'):
            uncond_embeddings = uncond_embeddings.last_hidden_state
        
        # Create sampler
        sampler = CFGSampler(scheduler, guidance_scale=self.guidance_scale)
        
        # Generate latents
        latents = sampler.sample(
            unet=unet,
            shape=(batch_size, 4, latent_size, latent_size),
            cond_embeddings=text_embeddings,
            uncond_embeddings=uncond_embeddings,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            show_progress=show_progress,
        )
        
        # Decode latents to images
        images = self._decode_latents(vae, latents)
        
        return images
    
    @torch.no_grad()
    def generate_unconditional(
        self,
        unet: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        """
        Generate unconditional images.
        
        Args:
            unet: U-Net model
            vae: VAE model
            text_encoder: Text encoder
            tokenizer: Tokenizer
            scheduler: Diffusion scheduler
            num_samples: Number of samples to generate
            device: Device to run on
            generator: Random generator
        
        Returns:
            List of generated PIL Images
        """
        # Use empty prompts for unconditional generation
        prompts = [""] * num_samples
        
        return self.generate(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompts=prompts,
            device=device,
            generator=generator,
            show_progress=False,
        )
    
    @torch.no_grad()
    def _decode_latents(self, vae, latents: torch.Tensor) -> List[Image.Image]:
        """Decode latents to PIL Images."""
        # Unscale latents
        if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
            latents = latents / vae.config.scaling_factor
        else:
            latents = latents / 0.18215
        
        # Decode
        if hasattr(vae, 'decode'):
            images = vae.decode(latents)
            if hasattr(images, 'sample'):
                images = images.sample
        else:
            images = vae(latents)
        
        # Convert to PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        
        pil_images = [Image.fromarray(img) for img in images]
        
        return pil_images
    
    def generate_for_fid(
        self,
        unet: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        prompts: List[str],
        num_samples: int,
        device: torch.device,
        batch_size: int = 8,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        """
        Generate many samples for FID calculation.
        
        Args:
            unet: U-Net model
            vae: VAE model
            text_encoder: Text encoder
            tokenizer: Tokenizer
            scheduler: Diffusion scheduler
            prompts: List of prompts to sample from
            num_samples: Total number of samples to generate
            device: Device to run on
            batch_size: Batch size for generation
            generator: Random generator
        
        Returns:
            List of generated PIL Images
        """
        import random
        
        all_images = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating for FID"):
            # Sample prompts for this batch
            current_batch_size = min(batch_size, num_samples - len(all_images))
            batch_prompts = [random.choice(prompts) for _ in range(current_batch_size)]
            
            # Generate
            images = self.generate(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                prompts=batch_prompts,
                device=device,
                generator=generator,
                show_progress=False,
            )
            
            all_images.extend(images)
            
            if len(all_images) >= num_samples:
                break
        
        return all_images[:num_samples]


def create_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 2,
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        rows: Number of rows (computed if not provided)
        cols: Number of columns (computed if not provided)
        padding: Padding between images
    
    Returns:
        Grid image
    """
    import math
    
    n = len(images)
    
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    elif rows is None:
        rows = int(math.ceil(n / cols))
    elif cols is None:
        cols = int(math.ceil(n / rows))
    
    # Get image size
    w, h = images[0].size
    
    # Create grid
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        x = col * (w + padding)
        y = row * (h + padding)
        
        grid.paste(img, (x, y))
    
    return grid


def create_labeled_image_grid(
    images: List[Image.Image],
    labels: List[str],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 10,
    label_height: int = 40,
    font_size: int = 12,
) -> Image.Image:
    """
    Create a grid of images with text labels above each image.
    
    Args:
        images: List of PIL Images
        labels: List of text labels for each image
        rows: Number of rows (computed if not provided)
        cols: Number of columns (computed if not provided)
        padding: Padding between images
        label_height: Height reserved for label text
        font_size: Font size for labels
    
    Returns:
        Grid image with labels
    """
    import math
    from PIL import ImageDraw, ImageFont
    
    n = len(images)
    assert len(labels) == n, "Number of labels must match number of images"
    
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    elif rows is None:
        rows = int(math.ceil(n / cols))
    elif cols is None:
        cols = int(math.ceil(n / rows))
    
    # Get image size
    w, h = images[0].size
    
    # Cell dimensions (image + label)
    cell_w = w
    cell_h = h + label_height
    
    # Create grid
    grid_w = cols * cell_w + (cols - 1) * padding
    grid_h = rows * cell_h + (rows - 1) * padding
    
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        
        x = col * (cell_w + padding)
        y = row * (cell_h + padding)
        
        # Draw label
        label_y = y + 2
        # Truncate label if too long
        max_chars = int(w / (font_size * 0.6))
        display_label = label[:max_chars] + "..." if len(label) > max_chars else label
        
        # Draw text with black color
        draw.text((x + 2, label_y), display_label, fill=(0, 0, 0), font=font)
        
        # Paste image below label
        img_y = y + label_height
        grid.paste(img, (x, img_y))
    
    return grid
