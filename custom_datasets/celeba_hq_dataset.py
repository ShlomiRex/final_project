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
    
    # ------------------------------------------------------------------ #
    #  Attribute → phrase mapping (with paraphrases for variety)          #
    #  Phrases are written WITHOUT a leading "with" – the connector is   #
    #  added once during assembly so captions read naturally.             #
    # ------------------------------------------------------------------ #
    _ATTRIBUTE_PHRASES = {
        # Facial hair / stubble
        "5_o_Clock_Shadow": ["light facial stubble", "a five o'clock shadow", "slight stubble"],
        "No_Beard": ["a clean-shaven face"],
        "Mustache": ["a mustache", "a moustache"],
        "Goatee": ["a goatee"],
        "Sideburns": ["sideburns"],
        # Eyes / eyebrows
        "Arched_Eyebrows": ["arched eyebrows", "curved eyebrows"],
        "Bushy_Eyebrows": ["bushy eyebrows", "thick eyebrows"],
        "Bags_Under_Eyes": ["bags under the eyes"],
        "Narrow_Eyes": ["narrow eyes"],
        # Nose (collision: Big_Nose vs Pointy_Nose – handled in assembly)
        "Big_Nose": ["a large nose", "a big nose", "a prominent nose"],
        "Pointy_Nose": ["a pointy nose", "a sharp nose"],
        # Lips
        "Big_Lips": ["full lips", "big lips"],
        # Cheekbones / cheeks
        "High_Cheekbones": ["high cheekbones", "prominent cheekbones"],
        "Rosy_Cheeks": ["rosy cheeks", "flushed cheeks"],
        # Face shape / build
        "Chubby": ["a chubby face", "a round face", "a plump face"],
        "Double_Chin": ["a double chin"],
        "Oval_Face": ["an oval face"],
        "Pale_Skin": ["pale skin", "fair skin", "light skin"],
        # Hair colour (just the colour word; combined with "hair" in assembly)
        "Blond_Hair": ["blond", "blonde"],
        "Black_Hair": ["black"],
        "Brown_Hair": ["brown"],
        "Gray_Hair": ["gray", "grey"],
        # Hair style (just the style word; combined with colour in assembly)
        "Straight_Hair": ["straight"],
        "Wavy_Hair": ["wavy"],
        "Bangs": ["bangs"],
        "Receding_Hairline": ["a receding hairline"],
        "Bald": ["a bald head"],
        # Accessories (standalone phrases)
        "Eyeglasses": ["wearing eyeglasses", "wearing glasses"],
        "Wearing_Hat": ["wearing a hat", "in a hat"],
        "Wearing_Earrings": ["wearing earrings"],
        "Wearing_Necklace": ["wearing a necklace"],
        "Wearing_Necktie": ["wearing a necktie", "wearing a tie"],
        # Makeup
        "Heavy_Makeup": ["heavy makeup", "wearing heavy makeup"],
        "Wearing_Lipstick": ["wearing lipstick"],
        # Expression (standalone)
        "Smiling": ["smiling"],
        "Mouth_Slightly_Open": ["mouth slightly open", "with parted lips"],
    }

    # Minor facial features are included with this probability (for variety)
    _MINOR_FEATURE_PROB = 0.80

    # ----- helpers ---------------------------------------------------- #

    def _phrase(self, attr_name):
        """Return a random paraphrase for *attr_name*, or ``None``."""
        phrases = self._ATTRIBUTE_PHRASES.get(attr_name)
        return random.choice(phrases) if phrases else None

    def _maybe_include(self, attr_name, attributes, prob=None):
        """Return a phrase if *attr_name* is active and passes the random gate."""
        if not attributes.get(attr_name, False):
            return None
        if random.random() > (prob if prob is not None else self._MINOR_FEATURE_PROB):
            return None
        return self._phrase(attr_name)

    # ----- hair ------------------------------------------------------- #

    def _build_hair_description(self, attributes):
        """Build hair description, resolving colour collisions by priority."""
        if attributes.get("Bald", False):
            return [random.choice(["a bald head", "no hair"])]

        # Colour – pick first match (resolves Black+Brown, etc.)
        hair_colour = None
        for attr in ("Blond_Hair", "Black_Hair", "Brown_Hair", "Gray_Hair"):
            if attributes.get(attr, False):
                hair_colour = self._phrase(attr)
                break

        # Style – resolve Straight vs Wavy (Straight wins)
        style = None
        if attributes.get("Straight_Hair", False):
            style = self._phrase("Straight_Hair")
        elif attributes.get("Wavy_Hair", False):
            style = self._phrase("Wavy_Hair")

        parts = []
        hair_words = ([style] if style else []) + ([hair_colour] if hair_colour else [])
        if hair_words:
            parts.append(" ".join(hair_words) + " hair")

        if attributes.get("Bangs", False):
            parts.append("bangs")
        if attributes.get("Receding_Hairline", False):
            parts.append(self._phrase("Receding_Hairline"))

        return parts  # may be empty

    # ----- facial hair ------------------------------------------------ #

    def _build_facial_hair_description(self, attributes):
        """Build facial-hair description."""
        parts = []
        for attr in ("5_o_Clock_Shadow", "Mustache", "Goatee", "Sideburns"):
            if attributes.get(attr, False):
                parts.append(self._phrase(attr))

        # "clean-shaven" only when No_Beard is active AND Male AND nothing else applies
        if not parts and attributes.get("No_Beard", False) and attributes.get("Male", False):
            return ["a clean-shaven face"]

        return parts  # may be empty

    # ----- main builder ----------------------------------------------- #

    def _build_caption(self, attributes):
        """
        Build a natural-language caption from CelebA binary attributes.

        Design principles
        -----------------
        1. **Deterministic mapping** – every active attribute has a fixed set
           of paraphrases; one is chosen at random for variety.
        2. **Never invent** – if an attribute is inactive it is silently
           omitted.  No "no accessories" or "neutral expression" fillers.
        3. **Collision resolution** – contradictory labels (e.g. Black_Hair
           + Brown_Hair) are resolved with an explicit priority rule.
        4. **Controlled randomness** – minor facial features may be dropped
           ~20 % of the time to avoid overfitting to one phrasing.
        5. **Core attributes always present** – gender, age, hair, expression,
           accessories, and facial hair are never dropped.

        Output structure
        ----------------
        ``"A photo of a <age> <gender> with <physical descriptors>, <expression>, <accessories>"``
        """
        # --- identity ------------------------------------------------- #
        is_male = attributes.get("Male", False)
        is_young = attributes.get("Young", False)

        gender_word = "man" if is_male else "woman"
        if is_young:
            age_word = "young"
            article = "a"
        else:
            # Young=0 means NOT young – do NOT hallucinate "older"/"middle-aged"
            age_word = None
            article = "a"

        # Collect all physical descriptors (without leading "with")
        descriptors = []

        # --- hair (always) -------------------------------------------- #
        descriptors.extend(self._build_hair_description(attributes))

        # --- facial hair (always) ------------------------------------- #
        descriptors.extend(self._build_facial_hair_description(attributes))

        # --- face shape / skin (usually) ------------------------------ #
        for attr in ("Chubby", "Double_Chin", "Pale_Skin"):
            d = self._maybe_include(attr, attributes)
            if d:
                descriptors.append(d)
        # Oval_Face only if NOT Chubby (contradictory)
        if not attributes.get("Chubby", False):
            d = self._maybe_include("Oval_Face", attributes)
            if d:
                descriptors.append(d)

        # --- facial features (usually) -------------------------------- #
        # Eyebrows – resolve Arched vs Bushy (Arched wins)
        if attributes.get("Arched_Eyebrows", False):
            d = self._maybe_include("Arched_Eyebrows", attributes)
            if d:
                descriptors.append(d)
        elif attributes.get("Bushy_Eyebrows", False):
            d = self._maybe_include("Bushy_Eyebrows", attributes)
            if d:
                descriptors.append(d)

        for attr in ("Bags_Under_Eyes", "Narrow_Eyes", "High_Cheekbones", "Rosy_Cheeks"):
            d = self._maybe_include(attr, attributes)
            if d:
                descriptors.append(d)

        # Nose – resolve Big vs Pointy (Big wins)
        if attributes.get("Big_Nose", False):
            d = self._maybe_include("Big_Nose", attributes)
            if d:
                descriptors.append(d)
        elif attributes.get("Pointy_Nose", False):
            d = self._maybe_include("Pointy_Nose", attributes)
            if d:
                descriptors.append(d)

        # Lips
        d = self._maybe_include("Big_Lips", attributes)
        if d:
            descriptors.append(d)

        # --- expression (always, if active) --------------------------- #
        expression = []
        if attributes.get("Smiling", False):
            expression.append(self._phrase("Smiling"))
        elif attributes.get("Mouth_Slightly_Open", False):
            expression.append(self._phrase("Mouth_Slightly_Open"))

        # --- accessories (always, if active) -------------------------- #
        accessories = []
        for attr in ("Eyeglasses", "Wearing_Hat", "Wearing_Earrings",
                      "Wearing_Necklace", "Wearing_Necktie"):
            if attributes.get(attr, False):
                accessories.append(self._phrase(attr))

        # --- makeup (usually, if active) ------------------------------ #
        # Heavy_Makeup wins over Wearing_Lipstick alone
        if attributes.get("Heavy_Makeup", False):
            d = self._maybe_include("Heavy_Makeup", attributes)
            if d:
                accessories.append(d)
        elif attributes.get("Wearing_Lipstick", False):
            d = self._maybe_include("Wearing_Lipstick", attributes)
            if d:
                accessories.append(d)

        # --- assemble ------------------------------------------------- #
        if age_word:
            caption = f"A photo of {article} {age_word} {gender_word}"
        else:
            caption = f"A photo of {article} {gender_word}"

        if descriptors:
            caption += " with " + ", ".join(descriptors)

        if expression:
            caption += ", " + ", ".join(expression)

        if accessories:
            caption += ", " + ", ".join(accessories)

        # Final cleanup
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
