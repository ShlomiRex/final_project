"""
Flickr30k Dataset Module

Dataset loader for Flickr30k with VQ-VAE tokenization support.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Literal
from PIL import Image
import random


class Flickr30kDataset(Dataset):
    """
    Flickr30k dataset for text-to-image training.
    
    Loads images and captions from the HuggingFace datasets library.
    Supports VQ-VAE tokenization and classifier-free guidance dropout.
    
    Args:
        split: Dataset split ("train" or "test")
        transform: Image transform pipeline
        tokenizer: Optional VQ-VAE tokenizer for pre-tokenization
        text_encoder: Optional text encoder for pre-encoding
        cfg_dropout: Probability of dropping text for CFG training
        cache_dir: Directory to cache dataset
        max_samples: Maximum number of samples (for debugging)
        
    Example:
        >>> dataset = Flickr30kDataset(
        ...     split="train",
        ...     transform=transforms.Compose([
        ...         transforms.Resize((256, 256)),
        ...         transforms.ToTensor(),
        ...         transforms.Normalize([0.5], [0.5]),
        ...     ]),
        ...     cfg_dropout=0.1,
        ... )
        >>> image, caption = dataset[0]
    """
    
    DATASET_NAME = "nlphuji/flickr30k"
    
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        text_encoder: Optional[Callable] = None,
        cfg_dropout: float = 0.0,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.cfg_dropout = cfg_dropout
        
        # Default cache directory
        if cache_dir is None:
            cache_dir = "./dataset_cache"
        
        # Load dataset
        from datasets import load_dataset
        
        self.dataset = load_dataset(
            self.DATASET_NAME,
            cache_dir=cache_dir,
            split=split,
        )
        
        # Limit samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with keys:
                - "image": Transformed image tensor [C, H, W]
                - "caption": Selected caption string
                - "all_captions": List of all captions
                - "image_id": Original image ID
        """
        sample = self.dataset[idx]
        
        # Get image
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        # Get captions (Flickr30k has 5 captions per image)
        captions = sample.get("caption", sample.get("captions", []))
        if isinstance(captions, str):
            captions = [captions]
        
        # Select random caption
        caption = random.choice(captions) if captions else ""
        
        # Apply CFG dropout
        if self.training and self.cfg_dropout > 0 and random.random() < self.cfg_dropout:
            caption = ""
        
        return {
            "image": image,
            "caption": caption,
            "all_captions": captions,
            "image_id": sample.get("img_id", idx),
        }
    
    @property
    def training(self) -> bool:
        """Check if dataset is in training mode."""
        return self.split == "train"


def create_dataloader(
    split: Literal["train", "test"] = "train",
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 4,
    cfg_dropout: float = 0.1,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for Flickr30k.
    
    Args:
        split: Dataset split
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        cfg_dropout: CFG dropout probability
        cache_dir: Dataset cache directory
        max_samples: Maximum samples (for debugging)
        
    Returns:
        DataLoader instance
    """
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])
    
    # Create dataset
    dataset = Flickr30kDataset(
        split=split,
        transform=transform,
        cfg_dropout=cfg_dropout,
        cache_dir=cache_dir,
        max_samples=max_samples,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )
    
    return dataloader


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for batching samples.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([sample["image"] for sample in batch])
    captions = [sample["caption"] for sample in batch]
    
    return {
        "images": images,
        "captions": captions,
    }
