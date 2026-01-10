"""
Visualization utilities.

Provides functions for:
- Creating image grids
- Saving visualizations
- Progress visualization
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont


def create_image_grid(
    images: List[Image.Image],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 2,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        rows: Number of rows (computed if not provided)
        cols: Number of columns (computed if not provided)
        padding: Padding between images
        background_color: Background color for grid
    
    Returns:
        Grid image
    """
    if len(images) == 0:
        raise ValueError("No images provided")
    
    n = len(images)
    
    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    elif rows is None:
        rows = int(math.ceil(n / cols))
    elif cols is None:
        cols = int(math.ceil(n / rows))
    
    # Get image size (assume all same size)
    w, h = images[0].size
    
    # Create grid
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    
    grid = Image.new("RGB", (grid_w, grid_h), color=background_color)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        x = col * (w + padding)
        y = row * (h + padding)
        
        # Resize if needed
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        
        grid.paste(img, (x, y))
    
    return grid


def create_labeled_grid(
    images: List[Image.Image],
    labels: List[str],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 2,
    label_height: int = 20,
    font_size: int = 12,
) -> Image.Image:
    """
    Create a grid of images with labels.
    
    Args:
        images: List of PIL Images
        labels: List of labels for each image
        rows: Number of rows
        cols: Number of columns
        padding: Padding between images
        label_height: Height reserved for labels
        font_size: Font size for labels
    
    Returns:
        Labeled grid image
    """
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match")
    
    n = len(images)
    
    # Calculate grid dimensions
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    elif rows is None:
        rows = int(math.ceil(n / cols))
    elif cols is None:
        cols = int(math.ceil(n / rows))
    
    # Get image size
    w, h = images[0].size
    cell_h = h + label_height
    
    # Create grid
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * cell_h + (rows - 1) * padding
    
    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        x = col * (w + padding)
        y = row * (cell_h + padding)
        
        # Resize if needed
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        
        grid.paste(img, (x, y))
        
        # Add label
        # Truncate label if too long
        max_chars = w // (font_size // 2)
        if len(label) > max_chars:
            label = label[:max_chars-3] + "..."
        
        draw.text((x, y + h + 2), label, fill=(0, 0, 0), font=font)
    
    return grid


def save_image_grid(
    images: List[Image.Image],
    save_path: Union[str, Path],
    **kwargs,
):
    """
    Create and save an image grid.
    
    Args:
        images: List of PIL Images
        save_path: Path to save grid
        **kwargs: Arguments passed to create_image_grid
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    grid = create_image_grid(images, **kwargs)
    grid.save(save_path)


def tensor_to_pil(
    tensor: torch.Tensor,
    normalize: bool = True,
    nrow: Optional[int] = None,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert a tensor to PIL Image(s).
    
    Args:
        tensor: Image tensor [B, C, H, W] or [C, H, W]
        normalize: Whether to denormalize from [-1, 1] to [0, 1]
        nrow: If provided, create a grid with this many images per row
    
    Returns:
        PIL Image or list of PIL Images
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Denormalize
    if normalize:
        tensor = (tensor + 1) / 2
    
    # Clamp
    tensor = tensor.clamp(0, 1)
    
    # Convert to uint8
    tensor = (tensor * 255).byte()
    
    # Create images
    images = []
    for i in range(tensor.shape[0]):
        img_array = tensor[i].permute(1, 2, 0).cpu().numpy()
        images.append(Image.fromarray(img_array))
    
    # Create grid if requested
    if nrow is not None:
        return create_image_grid(images, cols=nrow)
    
    return images[0] if len(images) == 1 else images


def create_progress_visualization(
    steps: List[Image.Image],
    labels: Optional[List[str]] = None,
) -> Image.Image:
    """
    Create a visualization of generation progress.
    
    Args:
        steps: List of images at different denoising steps
        labels: Optional labels for each step
    
    Returns:
        Progress visualization image
    """
    if labels is None:
        labels = [f"Step {i}" for i in range(len(steps))]
    
    return create_labeled_grid(
        images=steps,
        labels=labels,
        cols=len(steps),
        rows=1,
    )


def make_comparison_grid(
    image_sets: List[List[Image.Image]],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
) -> Image.Image:
    """
    Create a comparison grid with multiple rows of images.
    
    Useful for comparing different models or settings.
    
    Args:
        image_sets: List of image rows
        row_labels: Labels for each row
        col_labels: Labels for each column
    
    Returns:
        Comparison grid image
    """
    if not image_sets:
        raise ValueError("No image sets provided")
    
    # Flatten images
    all_images = []
    for row in image_sets:
        all_images.extend(row)
    
    num_cols = len(image_sets[0])
    num_rows = len(image_sets)
    
    # Create basic grid
    grid = create_image_grid(all_images, rows=num_rows, cols=num_cols)
    
    return grid
