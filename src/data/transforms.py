"""
Image transforms for training and inference.
"""

from __future__ import annotations

from torchvision import transforms
from typing import Tuple


def get_train_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Get training transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])


def get_val_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Get validation/inference transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def denormalize(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1].
    
    Args:
        tensor: Image tensor
        
    Returns:
        Denormalized tensor
    """
    return (tensor + 1) / 2
