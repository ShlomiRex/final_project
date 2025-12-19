"""
Image transforms for training and evaluation.

Provides consistent transforms for:
- Training (with augmentation)
- Evaluation (deterministic)
- Multi-resolution training
"""

from typing import List, Optional, Tuple, Union

import torch
from torchvision import transforms
from PIL import Image


def get_train_transforms(
    resolution: int = 256,
    center_crop: bool = True,
    random_flip: bool = True,
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> transforms.Compose:
    """
    Get training transforms.
    
    Args:
        resolution: Target resolution
        center_crop: Whether to center crop (vs random crop)
        random_flip: Whether to apply random horizontal flip
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    
    if center_crop:
        transform_list.append(transforms.CenterCrop(resolution))
    else:
        transform_list.append(transforms.RandomCrop(resolution))
    
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    
    return transforms.Compose(transform_list)


def get_eval_transforms(
    resolution: int = 256,
    normalize_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    normalize_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> transforms.Compose:
    """
    Get evaluation transforms (deterministic).
    
    Args:
        resolution: Target resolution
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    Denormalize a tensor.
    
    Args:
        tensor: Normalized tensor [B, C, H, W] or [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    tensor = tensor * std + mean
    
    return tensor.squeeze(0) if tensor.shape[0] == 1 else tensor


def tensor_to_pil(
    tensor: torch.Tensor,
    denorm: bool = True,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert tensor to PIL Image(s).
    
    Args:
        tensor: Image tensor [B, C, H, W] or [C, H, W]
        denorm: Whether to denormalize first
    
    Returns:
        PIL Image or list of PIL Images
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    if denorm:
        tensor = denormalize(tensor)
    
    # Clamp to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to uint8
    tensor = (tensor * 255).byte()
    
    # Move to CPU and convert to PIL
    images = []
    for i in range(tensor.shape[0]):
        img_array = tensor[i].permute(1, 2, 0).cpu().numpy()
        images.append(Image.fromarray(img_array))
    
    return images[0] if len(images) == 1 else images


def pil_to_tensor(
    image: Union[Image.Image, List[Image.Image]],
    normalize: bool = True,
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
    std: Tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    Convert PIL Image(s) to tensor.
    
    Args:
        image: PIL Image or list of PIL Images
        normalize: Whether to normalize
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Image tensor [B, C, H, W] or [C, H, W]
    """
    if isinstance(image, Image.Image):
        images = [image]
        single = True
    else:
        images = image
        single = False
    
    tensors = []
    for img in images:
        # Convert to tensor
        tensor = transforms.ToTensor()(img)
        
        if normalize:
            tensor = transforms.Normalize(mean, std)(tensor)
        
        tensors.append(tensor)
    
    tensor = torch.stack(tensors, dim=0)
    
    return tensor.squeeze(0) if single else tensor


class RandomResizedCropWithInfo(transforms.RandomResizedCrop):
    """
    Random resized crop that also returns crop parameters.
    
    Useful for reproducibility and debugging.
    """
    
    def forward(self, img: Image.Image) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            img: Input image
        
        Returns:
            Tuple of (cropped image, crop info dict)
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        cropped = transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )
        
        info = {
            "top": i,
            "left": j,
            "height": h,
            "width": w,
            "original_size": img.size,
        }
        
        return cropped, info


class ColorJitterIfEnabled:
    """Color jitter that can be disabled."""
    
    def __init__(
        self,
        enabled: bool = False,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ):
        self.enabled = enabled
        if enabled:
            self.jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if self.enabled:
            return self.jitter(img)
        return img
