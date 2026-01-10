"""
Dataset classes for text-image pairs.

Supports multiple datasets:
- Flickr30k
- COCO Captions
- LAION (streaming)
- Custom datasets
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import get_train_transforms


class TextImageDataset(Dataset):
    """
    Generic dataset for text-image pairs.
    
    Works with HuggingFace datasets or custom data.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer=None,  # Optional for unconditional models
        resolution: int = 256,
        image_column: str = "image",
        caption_column: str = "caption",
        center_crop: bool = True,
        random_flip: bool = True,
        max_length: int = 77,
        transform: Optional[Callable] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            dataset: HuggingFace dataset or list of dicts
            tokenizer: CLIP tokenizer (optional, None for unconditional models)
            resolution: Image resolution
            image_column: Column name for images
            caption_column: Column name for captions
            center_crop: Whether to center crop
            random_flip: Whether to randomly flip
            max_length: Max token length
            transform: Optional custom transform
            class_names: Optional mapping from integer labels to text descriptions
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length
        self.class_names = class_names
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms(
                resolution=resolution,
                center_crop=center_crop,
                random_flip=random_flip,
            )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # Get image
        image = item[self.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # Ensure PIL images are converted to RGB (handles grayscale MNIST)
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare return dict
        result = {"pixel_values": image}
        
        # For unconditional models (no tokenizer), skip text processing
        if self.tokenizer is None:
            return result
        
        # Get caption
        caption = item[self.caption_column]
        
        # Handle multiple captions (e.g., Flickr30k has 5 captions per image)
        if isinstance(caption, list):
            caption = random.choice(caption)
        
        # Convert integer labels to text using class names if available
        if isinstance(caption, (int, float)):
            label_idx = int(caption)
            if self.class_names and label_idx in self.class_names:
                caption = self.class_names[label_idx]
            else:
                # Fallback to generic description if class names not available
                caption = f"class {label_idx}"
        
        # Ensure caption is a string
        caption = str(caption)
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        result.update({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
        })
        
        return result


class MultiResolutionDataset(Dataset):
    """
    Dataset that returns images at multiple resolutions.
    
    Useful for training models that need to handle different resolutions.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        resolutions: List[int] = [256, 384, 512],
        resolution_probs: Optional[List[float]] = None,
        image_column: str = "image",
        caption_column: str = "caption",
        center_crop: bool = True,
        random_flip: bool = True,
        max_length: int = 77,
    ):
        """
        Args:
            dataset: HuggingFace dataset
            tokenizer: CLIP tokenizer
            resolutions: List of possible resolutions
            resolution_probs: Probability for each resolution
            image_column: Column name for images
            caption_column: Column name for captions
            center_crop: Whether to center crop
            random_flip: Whether to randomly flip
            max_length: Max token length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.resolutions = resolutions
        self.resolution_probs = resolution_probs or [1.0 / len(resolutions)] * len(resolutions)
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Pre-build transforms for each resolution
        self.transforms = {
            res: get_train_transforms(
                resolution=res,
                center_crop=center_crop,
                random_flip=random_flip,
            )
            for res in resolutions
        }
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # Randomly select resolution
        resolution = random.choices(self.resolutions, weights=self.resolution_probs)[0]
        transform = self.transforms[resolution]
        
        # Get and transform image
        image = item[self.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # Ensure PIL images are converted to RGB (handles grayscale MNIST)
            image = image.convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")
        
        image = transform(image)
        
        # Get caption
        caption = item[self.caption_column]
        if isinstance(caption, list):
            caption = random.choice(caption)
        
        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
            "resolution": resolution,
        }


class AspectRatioBucketDataset(Dataset):
    """
    Dataset with aspect ratio bucketing.
    
    Groups images by aspect ratio to minimize padding/cropping.
    """
    
    def __init__(
        self,
        dataset,
        tokenizer,
        target_area: int = 256 * 256,
        min_bucket_size: int = 256,
        max_bucket_size: int = 512,
        bucket_step: int = 64,
        image_column: str = "image",
        caption_column: str = "caption",
        max_length: int = 77,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.target_area = target_area
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length
        
        # Generate bucket sizes
        self.buckets = self._generate_buckets(
            min_bucket_size, max_bucket_size, bucket_step, target_area
        )
        
        # Build bucket index (expensive - do once)
        self.bucket_indices = self._build_bucket_indices()
    
    def _generate_buckets(
        self,
        min_size: int,
        max_size: int,
        step: int,
        target_area: int,
    ) -> List[Tuple[int, int]]:
        """Generate valid bucket (width, height) pairs."""
        buckets = []
        for w in range(min_size, max_size + 1, step):
            h = int(target_area / w)
            h = (h // step) * step  # Round to step
            if min_size <= h <= max_size:
                buckets.append((w, h))
        return buckets
    
    def _build_bucket_indices(self) -> Dict[Tuple[int, int], List[int]]:
        """Assign each image to a bucket based on aspect ratio."""
        bucket_indices = {bucket: [] for bucket in self.buckets}
        
        for idx in range(len(self.dataset)):
            # Get image size (without loading full image if possible)
            item = self.dataset[idx]
            image = item[self.image_column]
            
            if isinstance(image, Image.Image):
                w, h = image.size
            elif isinstance(image, str):
                with Image.open(image) as img:
                    w, h = img.size
            else:
                # Assume it's an array
                h, w = image.shape[:2]
            
            # Find best bucket
            aspect_ratio = w / h
            best_bucket = min(
                self.buckets,
                key=lambda b: abs(b[0] / b[1] - aspect_ratio)
            )
            bucket_indices[best_bucket].append(idx)
        
        return bucket_indices
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # Get image
        image = item[self.image_column]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        
        # Find bucket for this image
        w, h = image.size
        aspect_ratio = w / h
        bucket = min(
            self.buckets,
            key=lambda b: abs(b[0] / b[1] - aspect_ratio)
        )
        
        # Resize to bucket size
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(bucket, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(bucket),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        image = transform(image)
        
        # Get caption
        caption = item[self.caption_column]
        if isinstance(caption, list):
            caption = random.choice(caption)
        
        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "caption": caption,
            "bucket": bucket,
        }


def get_dataloader(
    config: dict,
    tokenizer,
    accelerator=None,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader from configuration.
    
    Args:
        config: Data configuration dictionary
        tokenizer: CLIP tokenizer
        accelerator: Accelerate accelerator (optional)
        split: Dataset split
    
    Returns:
        DataLoader
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset_name = config.get("dataset_name", config.get("dataset"))
    cache_dir = config.get("cache_dir", None)
    
    if dataset_name == "mnist" or dataset_name.lower() == "mnist":
        # Load MNIST dataset
        hf_dataset = load_dataset(
            "mnist",
            cache_dir=cache_dir,
            split=split,
        )
        image_column = "image"
        caption_column = "label"
        # Create custom class names for MNIST with descriptive text
        class_names = {
            i: f"a photograph of the number {i}" for i in range(10)
        }
        print(f"Loaded MNIST dataset with {len(hf_dataset)} images")
        print(f"Using text prompts: {class_names}")
    
    elif dataset_name == "flickr30k" or dataset_name == "nlphuji/flickr30k":
        # Use Lin-Chen/flickr30k which has a cleaner parquet-based format
        hf_dataset = load_dataset(
            "Lin-Chen/flickr30k",
            cache_dir=cache_dir,
            split="test",
        )
        image_column = "image"
        caption_column = "caption"
    
    elif dataset_name == "coco" or "coco" in dataset_name.lower():
        # Use yerevann/coco-karpathy which has a cleaner format
        hf_dataset = load_dataset(
            "yerevann/coco-karpathy",
            cache_dir=cache_dir,
            split="train" if split == "train" else "val",
        )
        image_column = "filepath"  # yerevann/coco-karpathy uses 'filepath' for images
        caption_column = "sentences"  # and 'sentences' for captions (list of captions)
    
    else:
        # Generic HuggingFace dataset
        hf_dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            split=split,
            streaming=config.get("streaming", False),
            trust_remote_code=True,  # Allow custom dataset loading scripts
        )
        # Debug: print column names
        if len(hf_dataset) > 0:
            first_example = hf_dataset[0]
            print(f"Dataset columns: {list(first_example.keys())}")
        
        # Use standard column names, most HF datasets use these
        image_column = config.get("image_column", "image")
        caption_column = config.get("caption_column", "label")  # Changed default to 'label' which is common
    
    # Extract class names from dataset if it's a classification dataset
    # Note: class_names may already be set for MNIST above
    if 'class_names' not in locals():
        class_names = None
    
    if class_names is None and hasattr(hf_dataset, 'features') and caption_column in hf_dataset.features:
        feature = hf_dataset.features[caption_column]
        # Check if it's a ClassLabel feature
        if hasattr(feature, 'names'):
            # Create mapping from integer to class name
            class_names = {i: name for i, name in enumerate(feature.names)}
            print(f"Loaded {len(class_names)} class names from dataset")
    
    # Create dataset
    if config.get("multi_resolution", {}).get("enabled", False):
        dataset = MultiResolutionDataset(
            dataset=hf_dataset,
            tokenizer=tokenizer,
            resolutions=config["multi_resolution"]["resolutions"],
            resolution_probs=config["multi_resolution"].get("resolution_probs"),
            image_column=image_column,
            caption_column=caption_column,
            center_crop=config.get("center_crop", True),
            random_flip=config.get("random_flip", True),
        )
    else:
        dataset = TextImageDataset(
            dataset=hf_dataset,
            tokenizer=tokenizer,
            resolution=config["resolution"],  # Required - no default
            image_column=image_column,
            caption_column=caption_column,
            center_crop=config.get("center_crop", True),  # center_crop can have default
            random_flip=config.get("random_flip", False),  # random_flip can have default
            class_names=class_names,  # Pass class names for label mapping
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],  # Required - no default
        shuffle=True,
        num_workers=config.get("num_workers", 4),  # num_workers can have default
        pin_memory=config.get("pin_memory", True),  # pin_memory can have default
        drop_last=True,
    )
    
    return dataloader
