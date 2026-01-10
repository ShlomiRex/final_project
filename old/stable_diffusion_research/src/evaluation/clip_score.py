"""
CLIP Score Calculator.

Measures image-text alignment using CLIP embeddings.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class CLIPScoreCalculator:
    """
    Calculator for CLIP Score.
    
    CLIP Score measures how well generated images match their text prompts
    using the CLIP model to compute image-text similarity.
    
    Higher CLIP Score = Better image-text alignment
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: CLIP model to use
            device: Device to run model on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load CLIP model
        self.model, self.processor = self._load_clip_model()
        self.model.eval()
        self.model.to(self.device)
    
    def _load_clip_model(self):
        """Load CLIP model and processor."""
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained(self.model_name)
        processor = CLIPProcessor.from_pretrained(self.model_name)
        
        return model, processor
    
    @torch.no_grad()
    def get_image_features(
        self,
        images: List[Image.Image],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract CLIP image features.
        
        Args:
            images: List of PIL Images
            normalize: Whether to L2 normalize features
        
        Returns:
            Image features [N, D]
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        features = self.model.get_image_features(**inputs)
        
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        
        return features
    
    @torch.no_grad()
    def get_text_features(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract CLIP text features.
        
        Args:
            texts: List of text prompts
            normalize: Whether to L2 normalize features
        
        Returns:
            Text features [N, D]
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        features = self.model.get_text_features(**inputs)
        
        if normalize:
            features = F.normalize(features, p=2, dim=-1)
        
        return features
    
    def calculate_clip_score(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> float:
        """
        Calculate CLIP Score for image-text pairs.
        
        CLIP Score = mean(cos_sim(image_features, text_features)) * 100
        
        Args:
            images: List of images
            texts: List of corresponding text prompts
        
        Returns:
            CLIP Score (0-100 scale)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        
        # Get features
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        
        # Compute cosine similarity for each pair
        similarities = (image_features * text_features).sum(dim=-1)
        
        # Average and scale to 100
        clip_score = similarities.mean().item() * 100
        
        return clip_score
    
    def calculate_clip_score_batched(
        self,
        images: List[Image.Image],
        texts: List[str],
        batch_size: int = 32,
    ) -> float:
        """
        Calculate CLIP Score with batching for large datasets.
        
        Args:
            images: List of images
            texts: List of corresponding text prompts
            batch_size: Batch size for processing
        
        Returns:
            CLIP Score (0-100 scale)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        
        all_similarities = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            image_features = self.get_image_features(batch_images)
            text_features = self.get_text_features(batch_texts)
            
            similarities = (image_features * text_features).sum(dim=-1)
            all_similarities.append(similarities)
        
        all_similarities = torch.cat(all_similarities)
        clip_score = all_similarities.mean().item() * 100
        
        return clip_score
    
    def calculate_image_image_similarity(
        self,
        images1: List[Image.Image],
        images2: List[Image.Image],
    ) -> float:
        """
        Calculate similarity between two sets of images.
        
        Useful for measuring diversity or comparing with reference images.
        
        Args:
            images1: First set of images
            images2: Second set of images
        
        Returns:
            Average cosine similarity
        """
        features1 = self.get_image_features(images1)
        features2 = self.get_image_features(images2)
        
        similarities = (features1 * features2).sum(dim=-1)
        
        return similarities.mean().item()
    
    def rank_images_by_text(
        self,
        images: List[Image.Image],
        text: str,
    ) -> List[tuple]:
        """
        Rank images by their similarity to a text prompt.
        
        Args:
            images: List of images to rank
            text: Text prompt to match
        
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features([text])
        
        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(-1)
        
        # Sort by score
        scores = similarities.cpu().tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        return ranked
