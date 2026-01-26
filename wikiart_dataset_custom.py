"""
Custom WikiArt Dataset Loader

This module provides custom PyTorch dataset classes that load WikiArt images
directly from parquet files without using the HuggingFace datasets library.

Two classes:
- WikiArtDatasetCustom: Map-style dataset with LRU cache (slow with shuffle)
- WikiArtIterableDataset: Iterable dataset with file-sequential loading (fast)
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import io
import random
import torch
from torch.utils.data import Dataset, IterableDataset
from functools import lru_cache


class WikiArtIterableDataset(IterableDataset):
    """Fast iterable dataset that loads one parquet file at a time.
    
    This is MUCH faster than random access because:
    1. Loads one parquet file (~400MB) into memory
    2. Yields all samples from it (shuffled within file)
    3. Moves to next file
    4. File order is shuffled each epoch
    
    Trade-off: Shuffling happens within each file (~1100 samples), not globally.
    This is acceptable for training - similar to how WebDataset/TFDS work.
    
    Args:
        data_dir (str or Path): Directory containing the parquet files
        transform (callable, optional): Transform to apply to images
        prompt_template (str): Template string for captions
        styles_list (list): List of art style names for indexing
        shuffle_files (bool): Shuffle file order each epoch (default: True)
        shuffle_samples (bool): Shuffle samples within each file (default: True)
    """
    
    def __init__(self, data_dir, transform=None, prompt_template="{style}", 
                 styles_list=None, shuffle_files=True, shuffle_samples=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.prompt_template = prompt_template
        self.styles_list = styles_list or []
        self.shuffle_files = shuffle_files
        self.shuffle_samples = shuffle_samples
        
        # Find all parquet files
        self.parquet_files = sorted(list(self.data_dir.glob("*.parquet")))
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        # Count total samples (read metadata only)
        print("Indexing parquet files...")
        self.file_sample_counts = []
        self.total_samples = 0
        
        import pyarrow.parquet as pq
        for i, pq_file in enumerate(self.parquet_files):
            parquet_file = pq.ParquetFile(pq_file)
            num_rows = parquet_file.metadata.num_rows
            self.file_sample_counts.append(num_rows)
            self.total_samples += num_rows
            
            if (i + 1) % 10 == 0 or i == len(self.parquet_files) - 1:
                print(f"  Indexed {i+1}/{len(self.parquet_files)} files ({self.total_samples:,} samples)")
        
        print(f"\n✓ WikiArtIterableDataset ready with {self.total_samples:,} samples")
        print(f"  Files: {len(self.parquet_files)} parquet files")
        print(f"  Shuffle files: {shuffle_files}, Shuffle samples: {shuffle_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def _process_row(self, row):
        """Process a single row from the parquet file."""
        # Extract image (stored as bytes in parquet)
        image_data = row['image']
        if isinstance(image_data, dict):
            image_bytes = image_data.get('bytes', image_data)
        else:
            image_bytes = image_data
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get style
        style = row.get('style', row.get('label', row.get('genre', 0)))
        
        # Convert style to index
        if isinstance(style, str):
            style_idx = self.styles_list.index(style) if style in self.styles_list else 0
        else:
            style_idx = int(style)
        
        # Create caption with clean style name
        style_name = self.styles_list[style_idx] if style_idx < len(self.styles_list) else "painting"
        clean_style_name = style_name.replace("_", " ")
        
        try:
            caption = self.prompt_template.format(style_name=clean_style_name)
        except KeyError:
            try:
                caption = self.prompt_template.format(style=clean_style_name)
            except KeyError:
                caption = f"A painting in {clean_style_name} style"
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, caption, style_idx
    
    def __iter__(self):
        """Iterate through all samples, one file at a time."""
        # Get file order (shuffle if enabled)
        file_indices = list(range(len(self.parquet_files)))
        if self.shuffle_files:
            random.shuffle(file_indices)
        
        # Iterate through each file
        for file_idx in file_indices:
            pq_file = self.parquet_files[file_idx]
            
            # Load entire parquet file into memory (one at a time)
            df = pd.read_parquet(pq_file)
            
            # Get row indices (shuffle within file if enabled)
            row_indices = list(range(len(df)))
            if self.shuffle_samples:
                random.shuffle(row_indices)
            
            # Yield each sample from this file
            for row_idx in row_indices:
                row = df.iloc[row_idx]
                try:
                    yield self._process_row(row)
                except Exception as e:
                    # Skip corrupted images
                    continue
            
            # File is done - memory will be freed when df goes out of scope


class WikiArtDatasetCustom(Dataset):
    """Custom dataset that loads WikiArt images directly from parquet files.
    
    Uses lazy loading - only loads individual samples on-demand to avoid OOM.
    
    Args:
        data_dir (str or Path): Directory containing the parquet files
        transform (callable, optional): Transform to apply to images
        prompt_template (str): Template string for captions, e.g. "{style}"
        styles_list (list): List of art style names for indexing
    """
    
    def __init__(self, data_dir, transform=None, prompt_template="{style}", styles_list=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.prompt_template = prompt_template
        self.styles_list = styles_list or []
        
        # Cache for loaded parquet files - keeps most recent files in memory
        self._cache_maxsize = 5  # Keep 5 parquet files in memory (~2GB)
        self._load_parquet_cached = lru_cache(maxsize=self._cache_maxsize)(self._load_parquet)
        
        print("Indexing parquet files (lazy loading)...")
        
        # Find all parquet files
        self.parquet_files = sorted(list(self.data_dir.glob("*.parquet")))
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        print(f"Found {len(self.parquet_files)} parquet files")
        
        # Build index: which file contains which rows (without loading image data)
        # Only read metadata (columns without image bytes)
        self.file_offsets = []  # (file_idx, row_count)
        self.total_samples = 0
        
        for i, pq_file in enumerate(self.parquet_files):
            # Read parquet metadata only (no data loading)
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(pq_file)
            num_rows = parquet_file.metadata.num_rows
            
            self.file_offsets.append((i, num_rows))
            self.total_samples += num_rows
            
            if (i + 1) % 10 == 0 or i == len(self.parquet_files) - 1:
                print(f"  Indexed {i+1}/{len(self.parquet_files)} files ({self.total_samples:,} samples)")
        
        print(f"\n✓ Dataset ready with {self.total_samples:,} samples")
        print(f"✓ Using lazy loading - images loaded on-demand")
    
    def __len__(self):
        return self.total_samples
    
    def _find_file_and_row(self, idx):
        """Find which parquet file contains the given index and the row within that file."""
        if idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range (total samples: {self.total_samples})")
        
        cumulative = 0
        for file_idx, row_count in self.file_offsets:
            if idx < cumulative + row_count:
                # Found the file
                local_idx = idx - cumulative
                return file_idx, local_idx
            cumulative += row_count
        
        raise IndexError(f"Index {idx} not found in file offsets")
    
    def _load_parquet(self, file_path_str):
        """Load a parquet file. This method is cached via LRU cache."""
        return pd.read_parquet(file_path_str)
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx, local_idx = self._find_file_and_row(idx)
        
        # Load parquet file (uses LRU cache - fast if recently accessed)
        pq_file = self.parquet_files[file_idx]
        df = self._load_parquet_cached(str(pq_file))
        row = df.iloc[local_idx]
        
        # Extract image (stored as bytes in parquet)
        image_data = row['image']
        if isinstance(image_data, dict):
            # Image might be stored as dict with 'bytes' key
            image_bytes = image_data.get('bytes', image_data)
        else:
            image_bytes = image_data
        
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get style (might be 'style', 'label', or 'genre')
        style = row.get('style', row.get('label', row.get('genre', 0)))
        
        # Convert style to index if it's a string
        if isinstance(style, str):
            style_idx = self.styles_list.index(style) if style in self.styles_list else 0
        else:
            style_idx = int(style)
        
        # Create caption
        style_name = self.styles_list[style_idx] if style_idx < len(self.styles_list) else "painting"
        
        # Clean style name: replace underscores with spaces for better prompts
        # e.g., "Naive_Art_Primitivism" -> "Naive Art Primitivism"
        clean_style_name = style_name.replace("_", " ")
        
        # Handle different prompt template formats
        try:
            caption = self.prompt_template.format(style_name=clean_style_name)
        except KeyError:
            try:
                caption = self.prompt_template.format(style=clean_style_name)
            except KeyError:
                # Fallback: just use the style name
                caption = f"A painting in {clean_style_name} style"
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, caption, style_idx
