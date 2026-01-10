"""
FID (Fréchet Inception Distance) Calculator.

Measures the quality of generated images by comparing
feature statistics with real images.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from tqdm import tqdm


class FIDCalculator:
    """
    Calculator for FID (Fréchet Inception Distance).
    
    FID measures the distance between the feature distributions
    of real and generated images using an InceptionV3 model.
    
    Lower FID = Better quality
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ):
        """
        Args:
            device: Device to run model on
            batch_size: Batch size for feature extraction
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load InceptionV3 model
        self.model = self._load_inception_model()
        self.model.eval()
        self.model.to(self.device)
    
    def _load_inception_model(self) -> nn.Module:
        """Load InceptionV3 model for feature extraction."""
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        # Load pretrained model
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        
        # Remove final classification layer to get features
        model.fc = nn.Identity()
        
        # Set to eval mode
        model.eval()
        
        return model
    
    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess images for InceptionV3."""
        from torchvision import transforms
        
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        tensors = [preprocess(img.convert("RGB")) for img in images]
        return torch.stack(tensors)
    
    @torch.no_grad()
    def extract_features(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract InceptionV3 features from images.
        
        Args:
            images: List of PIL Images or tensor of images
            show_progress: Whether to show progress bar
        
        Returns:
            Feature array [N, 2048]
        """
        if isinstance(images, list):
            # Preprocess PIL images
            all_features = []
            
            iterator = range(0, len(images), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Extracting features")
            
            for i in iterator:
                batch_images = images[i:i + self.batch_size]
                batch_tensor = self._preprocess_images(batch_images)
                batch_tensor = batch_tensor.to(self.device)
                
                features = self.model(batch_tensor)
                all_features.append(features.cpu().numpy())
            
            return np.concatenate(all_features, axis=0)
        
        else:
            # Already a tensor
            all_features = []
            
            iterator = range(0, len(images), self.batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Extracting features")
            
            for i in iterator:
                batch = images[i:i + self.batch_size].to(self.device)
                features = self.model(batch)
                all_features.append(features.cpu().numpy())
            
            return np.concatenate(all_features, axis=0)
    
    def calculate_statistics(self, features: np.ndarray) -> tuple:
        """
        Calculate mean and covariance of features.
        
        Args:
            features: Feature array [N, D]
        
        Returns:
            Tuple of (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(
        self,
        generated_images: Union[List[Image.Image], np.ndarray],
        real_images: Optional[Union[List[Image.Image], np.ndarray]] = None,
        real_statistics: Optional[tuple] = None,
        show_progress: bool = True,
    ) -> float:
        """
        Calculate FID score between generated and real images.
        
        Args:
            generated_images: Generated images
            real_images: Real images (if real_statistics not provided)
            real_statistics: Pre-computed (mu, sigma) for real images
            show_progress: Whether to show progress bar
        
        Returns:
            FID score
        """
        # Extract features from generated images
        gen_features = self.extract_features(generated_images, show_progress)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        # Get real image statistics
        if real_statistics is not None:
            mu_real, sigma_real = real_statistics
        elif real_images is not None:
            real_features = self.extract_features(real_images, show_progress)
            mu_real, sigma_real = self.calculate_statistics(real_features)
        else:
            raise ValueError("Either real_images or real_statistics must be provided")
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        
        return fid
    
    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Calculate Fréchet distance between two Gaussian distributions.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        
        Args:
            mu1, sigma1: Statistics of first distribution
            mu2, sigma2: Statistics of second distribution
            eps: Small value for numerical stability
        
        Returns:
            Fréchet distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Handle imaginary component (numerical error)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fid)
    
    def save_statistics(
        self,
        images: List[Image.Image],
        save_path: Union[str, Path],
    ):
        """
        Pre-compute and save statistics for a dataset.
        
        Args:
            images: List of images
            save_path: Path to save statistics
        """
        features = self.extract_features(images)
        mu, sigma = self.calculate_statistics(features)
        
        np.savez(save_path, mu=mu, sigma=sigma)
    
    def load_statistics(self, load_path: Union[str, Path]) -> tuple:
        """
        Load pre-computed statistics.
        
        Args:
            load_path: Path to load statistics from
        
        Returns:
            Tuple of (mu, sigma)
        """
        data = np.load(load_path)
        return data["mu"], data["sigma"]
