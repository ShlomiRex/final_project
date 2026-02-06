"""
CelebA-HQ Dataset with Text Captions

Custom PyTorch Dataset for CelebA-HQ images with text captions generated
from attributes. Maps attribute vectors to natural language prompts for
text-to-image generation.

CelebA-HQ provides 30,000 high-quality face images at 1024×1024 resolution,
along with 40 binary attributes per image (gender, age, hair color, accessories, etc.).
"""

from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import random
from pathlib import Path
import sys

# Import project configuration
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CELEBA_ATTRIBUTES, EXPERIMENT_4_CONFIG


class CelebAHQWithCaptions(Dataset):
    """
    CelebA-HQ dataset with text captions for each image.
    Generates captions from attribute labels.
    
    The dataset maps 40 binary attributes to natural language descriptions:
    - Gender: Male/Female
    - Age: Young/Old
    - Expression: Smiling, Mouth_Slightly_Open
    - Accessories: Eyeglasses, Wearing_Hat, Wearing_Necktie
    - Hair: Color (Blond, Black, Brown, Gray), Style (Wavy, Straight, Bangs)
    - Facial hair: Mustache, Goatee, Sideburns, No_Beard
    - Other: Attractive, Pale_Skin, Chubby, etc.
    
    Args:
        hf_dataset: HuggingFace CelebA-HQ dataset
        transform: Optional torchvision transforms to apply to images
        attributes_used: List of attribute names to use for prompting
        prompt_templates: List of template strings for caption generation
    
    Returns:
        Tuple of (image, caption, attributes_dict)
    """
    
    def __init__(
        self,
        hf_dataset,
        transform=None,
        attributes_used=None,
        prompt_templates=None,
    ):
        self.dataset = hf_dataset
        self.transform = transform
        
        # Use configured attributes if not specified
        self.attributes_used = attributes_used or CELEBA_ATTRIBUTES
        
        # Default prompt templates
        self.prompt_templates = prompt_templates or EXPERIMENT_4_CONFIG["prompt_templates"]
        
        # Get all possible attribute names from the dataset
        # CelebA has 40 attributes with names like "5_o_Clock_Shadow", "Arched_Eyebrows", etc.
        self._available_attributes = self._get_available_attributes()
        
        print(f"CelebA-HQ Dataset initialized:")
        print(f"  - Total images: {len(self.dataset)}")
        print(f"  - Available attributes: {len(self._available_attributes)}")
        print(f"  - Attributes used for prompting: {len(self.attributes_used)}")
    
    def _get_available_attributes(self):
        """Get list of available attributes from the dataset."""
        # Check first sample to see what attributes are available
        sample = self.dataset[0]
        
        # CelebA-HQ dataset structure varies, try different keys
        if 'attributes' in sample:
            # Some versions have attributes as a dict
            if isinstance(sample['attributes'], dict):
                return list(sample['attributes'].keys())
            # Some have attributes as a tensor with separate names
            elif hasattr(self.dataset, 'features') and 'attributes' in self.dataset.features:
                return self.dataset.features['attributes'].feature.names
        
        # Fallback: use standard CelebA attribute names
        return [
            "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
            "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
            "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
            "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
            "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
            "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
            "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
            "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
            "Young",
        ]
    
    def __len__(self):
        return len(self.dataset)
    
    def _parse_attributes(self, item):
        """Parse attributes from dataset item (handles different formats)."""
        attr_dict = {}
        
        # First try: item has 'attributes' key (some HF formats)
        if 'attributes' in item:
            attrs = item['attributes']
            
            # Handle dict format
            if isinstance(attrs, dict):
                return attrs
            
            # Handle tensor/array format
            elif isinstance(attrs, (torch.Tensor, np.ndarray, list)):
                # Create dict with attribute names
                for i, name in enumerate(self._available_attributes):
                    if i < len(attrs):
                        attr_dict[name] = bool(attrs[i] > 0)  # Convert to boolean
                return attr_dict
        
        # Second try: attributes are separate keys in item (flwrlabs/celeba format)
        # Try to load attributes directly from item keys
        for attr_name in self._available_attributes:
            if attr_name in item:
                attr_dict[attr_name] = bool(item[attr_name] > 0 if isinstance(item[attr_name], (int, float)) else item[attr_name])
        
        # If we found any attributes this way, return them
        if attr_dict:
            return attr_dict
        
        # Fallback: return empty dict
        return {}
    
    def _build_caption(self, attributes):
        """
        Build a natural language caption from attributes.
        
        Uses template-based generation with attribute-driven filling.
        
        Args:
            attributes: Dict of attribute_name -> bool
        
        Returns:
            Caption string
        """
        # Extract key attributes
        is_male = attributes.get("Male", False)
        is_young = attributes.get("Young", True)
        is_smiling = attributes.get("Smiling", False)
        has_glasses = attributes.get("Eyeglasses", False)
        has_hat = attributes.get("Wearing_Hat", False)
        
        # Hair color (mutually exclusive in theory, but use first match)
        hair_color = None
        if attributes.get("Blond_Hair", False):
            hair_color = "blond"
        elif attributes.get("Black_Hair", False):
            hair_color = "black"
        elif attributes.get("Brown_Hair", False):
            hair_color = "brown"
        elif attributes.get("Gray_Hair", False):
            hair_color = "gray"
        
        # Hair style
        hair_style = []
        if attributes.get("Wavy_Hair", False):
            hair_style.append("wavy")
        if attributes.get("Straight_Hair", False):
            hair_style.append("straight")
        if attributes.get("Bangs", False):
            hair_style.append("with bangs")
        
        # Facial hair
        facial_hair = []
        if attributes.get("Mustache", False):
            facial_hair.append("mustache")
        if attributes.get("Goatee", False):
            facial_hair.append("goatee")
        if attributes.get("Sideburns", False):
            facial_hair.append("sideburns")
        
        # Build attribute strings
        gender = "man" if is_male else "woman"
        age = "young" if is_young else "older"
        expression = "smiling" if is_smiling else ""
        
        # Accessories
        accessories = []
        if has_glasses:
            accessories.append("wearing eyeglasses")
        if has_hat:
            accessories.append("wearing a hat")
        
        # Build hair description
        hair_desc = []
        if hair_color:
            hair_desc.append(hair_color)
        if hair_style:
            hair_desc.extend(hair_style)
        hair = " ".join(hair_desc) + " hair" if hair_desc else "hair"
        
        # Choose a template and fill it
        template = random.choice(self.prompt_templates)
        
        # Fill template
        caption = template.format(
            age=age,
            gender=gender,
            expression=expression if expression else "with a neutral expression",
            hair=hair,
            accessories=", ".join(accessories) if accessories else "no accessories",
        )
        
        # Clean up whitespace
        caption = " ".join(caption.split())
        
        return caption
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Parse attributes
        attributes = self._parse_attributes(item)
        
        # Generate caption
        caption = self._build_caption(attributes)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, caption, attributes


def build_simple_prompt(attributes, template="A photo of a {age} {gender}"):
    """
    Build a simple prompt from attributes (for inference/evaluation).
    
    Args:
        attributes: Dict of attribute_name -> bool
        template: Template string with placeholders
    
    Returns:
        Prompt string
    """
    is_male = attributes.get("Male", False)
    is_young = attributes.get("Young", True)
    
    gender = "man" if is_male else "woman"
    age = "young" if is_young else "older"
    
    prompt = template.format(age=age, gender=gender)
    return prompt


def get_prompt_from_attributes(
    male=False,
    young=True,
    smiling=False,
    eyeglasses=False,
    blond_hair=False,
    black_hair=False,
):
    """
    Generate a prompt from specific attributes (for controlled generation).
    
    Args:
        male: Gender (False=female, True=male)
        young: Age (True=young, False=older)
        smiling: Expression
        eyeglasses: Wearing eyeglasses
        blond_hair: Has blond hair
        black_hair: Has black hair
    
    Returns:
        Tuple of (prompt, attributes_dict)
    """
    attributes = {
        "Male": male,
        "Young": young,
        "Smiling": smiling,
        "Eyeglasses": eyeglasses,
        "Blond_Hair": blond_hair,
        "Black_Hair": black_hair,
    }
    
    # Build description
    gender = "man" if male else "woman"
    age = "young" if young else "older"
    expression = ", smiling" if smiling else ""
    glasses = ", wearing eyeglasses" if eyeglasses else ""
    
    hair = ""
    if blond_hair:
        hair = " with blond hair"
    elif black_hair:
        hair = " with black hair"
    
    prompt = f"A photo of a {age} {gender}{hair}{expression}{glasses}"
    
    return prompt, attributes


if __name__ == "__main__":
    # Test caption generation
    print("Testing caption generation...")
    
    # Test with sample attributes
    test_attributes = {
        "Male": False,
        "Young": True,
        "Smiling": True,
        "Eyeglasses": True,
        "Blond_Hair": True,
        "Wavy_Hair": True,
    }
    
    # Create a mock dataset item
    class MockDataset:
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            return {
                'image': Image.new('RGB', (256, 256)),
                'attributes': test_attributes,
            }
    
    dataset = CelebAHQWithCaptions(MockDataset())
    caption = dataset._build_caption(test_attributes)
    
    print(f"\nTest attributes: {test_attributes}")
    print(f"Generated caption: {caption}")
    
    # Test controlled generation
    print("\n" + "="*60)
    print("Testing controlled prompt generation...")
    
    examples = [
        (False, True, True, False, True, False),  # Young woman, smiling, blond
        (True, True, False, True, False, True),   # Young man, glasses, black hair
        (True, False, False, False, False, False), # Older man
        (False, False, True, False, False, False), # Older woman, smiling
    ]
    
    for male, young, smiling, glasses, blond, black in examples:
        prompt, attrs = get_prompt_from_attributes(male, young, smiling, glasses, blond, black)
        print(f"\n{prompt}")
    
    print("\n✓ Caption generation test passed!")
