"""
WikiArt Dataset with Text Captions

Custom PyTorch Dataset for WikiArt images with text captions.
Maps style labels to text prompts for text-to-image generation.
"""

from torch.utils.data import Dataset
from config import WIKIART_STYLES


class WikiArtWithCaptions(Dataset):
    """
    WikiArt dataset with text captions for each image.
    Maps style labels to text prompts.
    
    Args:
        hf_dataset: HuggingFace WikiArt dataset
        transform: Optional torchvision transforms to apply to images
        prompt_template: Template string for captions (e.g., "A painting in the style of {style_name}")
    
    Returns:
        Tuple of (image, caption, style_idx)
    """
    def __init__(self, hf_dataset, transform=None, prompt_template="A painting in the style of {style_name}"):
        self.dataset = hf_dataset
        self.transform = transform
        self.prompt_template = prompt_template
        self.style_names = WIKIART_STYLES
        
        # Determine the style column name (varies in different dataset versions)
        sample = hf_dataset[0]
        if 'style' in sample:
            self.style_column = 'style'
        elif 'label' in sample:
            self.style_column = 'label'
        else:
            # If no style column, we'll use random styles (fallback)
            self.style_column = None
            print("Warning: No style column found, using random styles")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get style label
        if self.style_column:
            style_idx = item[self.style_column]
            # Ensure style_idx is within bounds
            if style_idx >= len(self.style_names):
                style_idx = style_idx % len(self.style_names)
        else:
            style_idx = idx % len(self.style_names)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Create caption
        style_name = self.style_names[style_idx].replace('_', ' ')
        caption = self.prompt_template.format(style_name=style_name)
        
        return image, caption, style_idx
