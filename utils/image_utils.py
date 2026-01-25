"""
Image Utilities

Functions for saving and loading images to/from folders.
Used for pre-generating images for FID evaluation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def save_image(
    image: Union[torch.Tensor, np.ndarray],
    filepath: Path,
    normalize: bool = True,
) -> None:
    """
    Save a single image to a file.
    
    Args:
        image: Image tensor/array. Expected shapes:
               - (H, W) for grayscale
               - (1, H, W) for grayscale with channel dim
               - (3, H, W) for RGB
               - (C, H, W) where C in [1, 3]
        filepath: Path to save the image
        normalize: If True, assume image is in [0, 1] and convert to [0, 255]
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]
    
    # Handle channel dimension
    if image.ndim == 3:
        if image.shape[0] == 1:
            # Grayscale: (1, H, W) -> (H, W)
            image = image[0]
        elif image.shape[0] == 3:
            # RGB: (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
    
    # Normalize to [0, 255]
    if normalize:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        image = image.clip(0, 255).astype(np.uint8)
    
    # Save
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if image.ndim == 2:
        Image.fromarray(image, mode='L').save(filepath)
    else:
        Image.fromarray(image, mode='RGB').save(filepath)


def save_images_to_folder(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray],
    folder: Path,
    prefix: str = "",
    start_index: int = 0,
    normalize: bool = True,
    extension: str = "png",
) -> int:
    """
    Save multiple images to a folder.
    
    Args:
        images: Batch of images (N, C, H, W) or list of images
        folder: Directory to save images
        prefix: Prefix for filenames (e.g., "img_" -> "img_0000.png")
        start_index: Starting index for filenames
        normalize: If True, assume images are in [0, 1]
        extension: File extension (default: png)
    
    Returns:
        Number of images saved
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    # Handle list of tensors
    if isinstance(images, list):
        images = torch.stack(images)
    
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Ensure 4D
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    
    count = 0
    for i, img in enumerate(images):
        filename = f"{prefix}{start_index + i:04d}.{extension}"
        filepath = folder / filename
        save_image(img, filepath, normalize=normalize)
        count += 1
    
    return count


def load_image(
    filepath: Path,
    normalize: bool = True,
    as_tensor: bool = True,
    grayscale: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Load a single image from a file.
    
    Args:
        filepath: Path to the image file
        normalize: If True, normalize to [0, 1] (otherwise [0, 255])
        as_tensor: If True, return torch.Tensor, else np.ndarray
        grayscale: If True, load as grayscale (1 channel)
    
    Returns:
        Image as tensor (C, H, W) or array (H, W) / (H, W, C)
    """
    filepath = Path(filepath)
    
    mode = 'L' if grayscale else 'RGB'
    img = Image.open(filepath).convert(mode)
    img_array = np.array(img)
    
    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    
    if as_tensor:
        if grayscale:
            # (H, W) -> (1, H, W)
            return torch.from_numpy(img_array).unsqueeze(0)
        else:
            # (H, W, C) -> (C, H, W)
            return torch.from_numpy(img_array).permute(2, 0, 1)
    
    return img_array


def load_images_from_folder(
    folder: Path,
    limit: Optional[int] = None,
    normalize: bool = True,
    as_tensor: bool = True,
    grayscale: bool = True,
    extensions: Tuple[str, ...] = ("png", "jpg", "jpeg"),
) -> Union[torch.Tensor, List[np.ndarray]]:
    """
    Load all images from a folder.
    
    Args:
        folder: Directory containing images
        limit: Maximum number of images to load (None for all)
        normalize: If True, normalize to [0, 1]
        as_tensor: If True, return stacked tensor (N, C, H, W)
        grayscale: If True, load as grayscale
        extensions: Tuple of valid file extensions
    
    Returns:
        Stacked tensor (N, C, H, W) or list of arrays
    """
    folder = Path(folder)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*.{ext}"))
    
    # Sort by filename
    image_files = sorted(image_files)
    
    if limit is not None:
        image_files = image_files[:limit]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder}")
    
    images = []
    for filepath in image_files:
        img = load_image(filepath, normalize=normalize, as_tensor=as_tensor, grayscale=grayscale)
        images.append(img)
    
    if as_tensor:
        return torch.stack(images)
    return images


def count_images_in_folder(
    folder: Path,
    extensions: Tuple[str, ...] = ("png", "jpg", "jpeg"),
) -> int:
    """
    Count the number of images in a folder.
    
    Args:
        folder: Directory to count images in
        extensions: Tuple of valid file extensions
    
    Returns:
        Number of image files
    """
    folder = Path(folder)
    
    if not folder.exists():
        return 0
    
    count = 0
    for ext in extensions:
        count += len(list(folder.glob(f"*.{ext}")))
    
    return count


def folder_has_enough_images(
    folder: Path,
    required: int,
    extensions: Tuple[str, ...] = ("png", "jpg", "jpeg"),
) -> bool:
    """
    Check if a folder contains at least the required number of images.
    
    Args:
        folder: Directory to check
        required: Minimum number of images required
        extensions: Tuple of valid file extensions
    
    Returns:
        True if folder has >= required images
    """
    return count_images_in_folder(folder, extensions) >= required


def save_mnist_samples(
    dataset,
    output_dir: Path,
    images_per_digit: int = 100,
    digits: List[int] = None,
) -> dict:
    """
    Save samples from MNIST dataset organized by digit.
    
    Args:
        dataset: MNIST dataset (torchvision)
        output_dir: Base directory for saving (will create digit_X subdirs)
        images_per_digit: Number of images to save per digit
        digits: List of digits to save (default: 0-9)
    
    Returns:
        Dict with counts per digit: {digit: count_saved}
    """
    if digits is None:
        digits = list(range(10))
    
    output_dir = Path(output_dir)
    
    # Organize dataset by digit
    digit_indices = {d: [] for d in digits}
    for idx, (_, label) in enumerate(dataset):
        if label in digit_indices:
            digit_indices[label].append(idx)
    
    results = {}
    
    for digit in digits:
        digit_dir = output_dir / f"digit_{digit}"
        
        # Skip if already has enough images
        if folder_has_enough_images(digit_dir, images_per_digit):
            existing = count_images_in_folder(digit_dir)
            print(f"Digit {digit}: Already has {existing} images, skipping")
            results[digit] = existing
            continue
        
        digit_dir.mkdir(parents=True, exist_ok=True)
        
        # Get indices for this digit
        indices = digit_indices[digit][:images_per_digit]
        
        if len(indices) < images_per_digit:
            print(f"Warning: Digit {digit} only has {len(indices)} samples available")
        
        # Save images
        count = 0
        for i, idx in enumerate(indices):
            img, _ = dataset[idx]
            
            # Convert to [0, 1] if needed
            if isinstance(img, torch.Tensor):
                if img.min() < 0:
                    # Assume [-1, 1] -> [0, 1]
                    img = (img + 1) / 2
            
            filepath = digit_dir / f"{i:04d}.png"
            save_image(img, filepath, normalize=True)
            count += 1
        
        print(f"Digit {digit}: Saved {count} images")
        results[digit] = count
    
    return results
