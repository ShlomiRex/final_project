"""
Main Evaluator Class.

Orchestrates evaluation including:
- Sample generation
- FID calculation
- CLIP score calculation
- Logging to MLflow
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

from .fid import FIDCalculator
from .clip_score import CLIPScoreCalculator
from .sample_generator import SampleGenerator, create_image_grid


class Evaluator:
    """
    Main evaluator for diffusion models.
    
    Handles:
    - Generating samples with fixed prompts for progress tracking
    - Calculating FID score
    - Calculating CLIP score
    - Creating visualization grids
    """
    
    def __init__(
        self,
        config: dict,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            config: Evaluation configuration
            device: Device to run evaluation on
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get config values
        self.sample_prompts = config["sample_prompts"]
        self.include_unconditional = config["include_unconditional"]
        self.num_samples = config["num_samples"]
        
        # FID config
        fid_config = config["fid"]
        self.fid_enabled = fid_config["enabled"]
        self.fid_num_samples = fid_config["num_samples"]
        self.fid_batch_size = fid_config["batch_size"]
        
        # CLIP score config
        clip_config = config["clip_score"]
        self.clip_enabled = clip_config["enabled"]
        self.clip_num_samples = clip_config["num_samples"]
        self.clip_batch_size = clip_config["batch_size"]
        
        # Diffusion config
        diffusion_config = config["diffusion"]
        self.num_inference_steps = diffusion_config["num_inference_steps"]
        self.guidance_scale = diffusion_config["guidance_scale"]
        
        # Resolution
        self.resolution = config["resolution"]
        
        # Initialize components (lazy loading)
        self._sample_generator = None
        self._fid_calculator = None
        self._clip_calculator = None
        
        # Reference statistics for FID (optional)
        self.reference_stats = None
        reference_path = fid_config.get("reference_stats")
        if reference_path:
            self._load_reference_stats(reference_path)
    
    @property
    def sample_generator(self) -> SampleGenerator:
        """Lazy load sample generator."""
        if self._sample_generator is None:
            self._sample_generator = SampleGenerator(
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                resolution=self.resolution,
            )
        return self._sample_generator
    
    @property
    def fid_calculator(self) -> FIDCalculator:
        """Lazy load FID calculator."""
        if self._fid_calculator is None:
            self._fid_calculator = FIDCalculator(
                device=self.device,
                batch_size=self.fid_batch_size,
            )
        return self._fid_calculator
    
    @property
    def clip_calculator(self) -> CLIPScoreCalculator:
        """Lazy load CLIP score calculator."""
        if self._clip_calculator is None:
            self._clip_calculator = CLIPScoreCalculator(
                device=self.device,
            )
        return self._clip_calculator
    
    def _load_reference_stats(self, path: str):
        """Load pre-computed reference statistics for FID."""
        import numpy as np
        data = np.load(path)
        self.reference_stats = (data["mu"], data["sigma"])
    
    @torch.no_grad()
    def generate_samples(
        self,
        unet: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        device: torch.device,
        seed: Optional[int] = None,
    ) -> Dict[str, Image.Image]:
        """
        Generate samples for visualization.
        
        Uses fixed prompts for consistent progress tracking.
        
        Args:
            unet: U-Net model
            vae: VAE model
            text_encoder: Text encoder
            tokenizer: Tokenizer
            scheduler: Diffusion scheduler
            device: Device to run on
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping prompts to generated images
        """
        unet.eval()
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        
        samples = {}
        
        # Generate for each prompt
        for prompt in self.sample_prompts:
            images = self.sample_generator.generate(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                prompts=[prompt],
                device=device,
                generator=generator,
                show_progress=False,
            )
            samples[prompt] = images[0]
        
        # Generate unconditional samples
        if self.include_unconditional:
            uncond_images = self.sample_generator.generate_unconditional(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                num_samples=4,
                device=device,
                generator=generator,
            )
            for i, img in enumerate(uncond_images):
                samples[f"unconditional_{i}"] = img
        
        return samples
    
    @torch.no_grad()
    def calculate_metrics(
        self,
        unet: nn.Module,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        device: torch.device,
        real_images: Optional[List[Image.Image]] = None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics (FID, CLIP score).
        
        Args:
            unet: U-Net model
            vae: VAE model
            text_encoder: Text encoder
            tokenizer: Tokenizer
            scheduler: Diffusion scheduler
            device: Device to run on
            real_images: Real images for FID (optional if reference_stats set)
            prompts: Prompts for generation
        
        Returns:
            Dictionary of metrics
        """
        unet.eval()
        metrics = {}
        
        if prompts is None:
            prompts = self.sample_prompts * (self.fid_num_samples // len(self.sample_prompts) + 1)
        
        # Calculate FID
        if self.fid_enabled:
            print("Calculating FID score...")
            
            # Generate samples for FID
            fid_images = self.sample_generator.generate_for_fid(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                prompts=prompts,
                num_samples=self.fid_num_samples,
                device=device,
                batch_size=self.fid_batch_size,
            )
            
            # Calculate FID
            fid_score = self.fid_calculator.calculate_fid(
                generated_images=fid_images,
                real_images=real_images,
                real_statistics=self.reference_stats,
            )
            
            metrics["fid"] = fid_score
            print(f"FID Score: {fid_score:.2f}")
        
        # Calculate CLIP score
        if self.clip_enabled:
            print("Calculating CLIP score...")
            
            # Generate samples with their prompts
            clip_prompts = prompts[:self.clip_num_samples]
            
            clip_images = self.sample_generator.generate_for_fid(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=scheduler,
                prompts=clip_prompts,
                num_samples=len(clip_prompts),
                device=device,
                batch_size=self.clip_batch_size,
            )
            
            # Calculate CLIP score
            clip_score = self.clip_calculator.calculate_clip_score_batched(
                images=clip_images,
                texts=clip_prompts,
                batch_size=self.clip_batch_size,
            )
            
            metrics["clip_score"] = clip_score
            print(f"CLIP Score: {clip_score:.2f}")
        
        return metrics
    
    def create_sample_grid(
        self,
        samples: Dict[str, Image.Image],
    ) -> Image.Image:
        """
        Create a grid of samples for visualization.
        
        Args:
            samples: Dictionary mapping prompts to images
        
        Returns:
            Grid image
        """
        images = list(samples.values())
        return create_image_grid(images)
