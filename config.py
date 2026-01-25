"""
Project Configuration

Central configuration for all experiments and notebooks.
All paths should use PROJECT_ROOT for absolute path resolution.
"""

from pathlib import Path


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path("/home/doshlom4/work/final_project")

# Data paths
DATASET_CACHE_DIR = PROJECT_ROOT / "dataset_cache"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Experiment 1: MNIST Text-to-Image with CFG
EXPERIMENT_1_DIR = OUTPUTS_DIR / "experiment_1"
EXPERIMENT_1_DATASET_DIR = EXPERIMENT_1_DIR / "dataset"
EXPERIMENT_1_GENERATED_DIR = EXPERIMENT_1_DIR / "generated"
EXPERIMENT_1_METRICS_DIR = EXPERIMENT_1_DIR / "metrics"

# Experiment 2: CIFAR-10 Text-to-Image with CFG
EXPERIMENT_2_DIR = OUTPUTS_DIR / "experiment_2"
EXPERIMENT_2_DATASET_DIR = EXPERIMENT_2_DIR / "dataset"
EXPERIMENT_2_GENERATED_DIR = EXPERIMENT_2_DIR / "generated"
EXPERIMENT_2_METRICS_DIR = EXPERIMENT_2_DIR / "metrics"

# Experiment 3: WikiArt Text-to-Image with CFG
EXPERIMENT_3_DIR = OUTPUTS_DIR / "experiment_3"
EXPERIMENT_3_DATASET_DIR = EXPERIMENT_3_DIR / "dataset"
EXPERIMENT_3_GENERATED_DIR = EXPERIMENT_3_DIR / "generated"
EXPERIMENT_3_METRICS_DIR = EXPERIMENT_3_DIR / "metrics"

# Research paper
RESEARCH_PAPER_DIR = PROJECT_ROOT / "research-paper"
FIGURES_DIR = RESEARCH_PAPER_DIR / "figures" / "experiments"


# =============================================================================
# Model Configuration
# =============================================================================

# UNet configuration for MNIST
UNET_CONFIG = {
    "sample_size": 28,
    "in_channels": 1,
    "out_channels": 1,
    "layers_per_block": 2,
    "block_out_channels": (32, 64, 64, 32),
    "down_block_types": (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ),
    "cross_attention_dim": 512,  # CLIP embedding dimension
}

# UNet configuration for CIFAR-10 (larger capacity for RGB images)
UNET_CIFAR10_CONFIG = {
    "sample_size": 32,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (64, 128, 256, 256),
    "down_block_types": (
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ),
    "cross_attention_dim": 512,  # CLIP embedding dimension
}

# UNet configuration for WikiArt (128×128 RGB images)
UNET_WIKIART_CONFIG = {
    "sample_size": 128,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (128, 256, 512, 512, 512),  # Larger capacity for complex art
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    "cross_attention_dim": 512,  # CLIP embedding dimension
}

# CLIP configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TOKENIZER_MAX_LENGTH = 8


# =============================================================================
# Training Configuration
# =============================================================================

TRAIN_CONFIG = {
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "num_train_timesteps": 1000,
    "beta_schedule": "squaredcos_cap_v2",
    "checkpoint_every_n_epochs": 5,
}

# Training configuration for CIFAR-10 (smaller batch size due to larger model)
TRAIN_CIFAR10_CONFIG = {
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "num_train_timesteps": 1000,
    "beta_schedule": "squaredcos_cap_v2",
    "checkpoint_every_n_epochs": 10,
}

# Training configuration for WikiArt (even smaller batch size for 128×128 images)
TRAIN_WIKIART_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_train_timesteps": 1000,
    "beta_schedule": "squaredcos_cap_v2",
    "checkpoint_every_n_epochs": 10,
}


# =============================================================================
# Inference Configuration
# =============================================================================

INFERENCE_CONFIG = {
    "num_inference_steps": 50,
    "beta_schedule": "squaredcos_cap_v2",
    "num_train_timesteps": 1000,
}


# =============================================================================
# Experiment 1: MNIST Evaluation Configuration
# =============================================================================

EXPERIMENT_1_CONFIG = {
    # Guidance scales to evaluate
    "guidance_scales": [0, 5, 10, 15, 20, 30, 40, 50, 100],
    
    # Number of images per digit per guidance scale
    "images_per_digit": 100,
    
    # All digits (0-9)
    "digits": list(range(10)),
    
    # Prompt template
    "prompt_template": "A handwritten digit {digit}",
}


# =============================================================================
# Experiment 2: CIFAR-10 Evaluation Configuration
# =============================================================================

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

EXPERIMENT_2_CONFIG = {
    # Guidance scales to evaluate
    "guidance_scales": [0, 5, 10, 15, 20, 30, 40, 50, 100],
    
    # Number of images per class per guidance scale
    "images_per_class": 100,
    
    # All classes (0-9)
    "classes": list(range(10)),
    
    # Class names
    "class_names": CIFAR10_CLASSES,
    
    # Prompt template
    "prompt_template": "A photo of a {class_name}",
}


# =============================================================================
# Experiment 3: WikiArt Evaluation Configuration
# =============================================================================

# WikiArt art styles (27 major styles from WikiArt dataset)
WIKIART_STYLES = [
    "Abstract_Expressionism",
    "Action_painting",
    "Analytical_Cubism",
    "Art_Nouveau_Modern",
    "Baroque",
    "Color_Field_Painting",
    "Contemporary_Realism",
    "Cubism",
    "Early_Renaissance",
    "Expressionism",
    "Fauvism",
    "High_Renaissance",
    "Impressionism",
    "Mannerism_Late_Renaissance",
    "Minimalism",
    "Naive_Art_Primitivism",
    "New_Realism",
    "Northern_Renaissance",
    "Pointillism",
    "Pop_Art",
    "Post_Impressionism",
    "Realism",
    "Rococo",
    "Romanticism",
    "Symbolism",
    "Synthetic_Cubism",
    "Ukiyo_e",
]

EXPERIMENT_3_CONFIG = {
    # Guidance scales to evaluate
    "guidance_scales": [0, 5, 10, 15, 20, 30, 40, 50, 100],
    
    # Number of images per class per guidance scale
    "images_per_class": 100,
    
    # All style classes
    "classes": list(range(len(WIKIART_STYLES))),
    
    # Class names (styles)
    "class_names": WIKIART_STYLES,
    
    # Prompt template
    "prompt_template": "A painting in the style of {style_name}",
}


# =============================================================================
# Classifier Configuration
# =============================================================================

CLASSIFIER_CONFIG = {
    "embedding_dim": 128,
    "num_classes": 10,
    "checkpoint_name": "mnist_classifier.pt",
}


# =============================================================================
# Checkpoint Names
# =============================================================================

# Experiment 1: MNIST
UNET_CHECKPOINT_PREFIX = "train1_unet_checkpoint_epoch_"
CLASSIFIER_CHECKPOINT_NAME = "mnist_classifier.pt"

# Experiment 2: CIFAR-10
UNET_CIFAR10_CHECKPOINT_PREFIX = "cifar10_unet_checkpoint_epoch_"

# Experiment 3: WikiArt
UNET_WIKIART_CHECKPOINT_PREFIX = "wikiart_unet_checkpoint_epoch_"


# =============================================================================
# Helper Functions - Experiment 1 (MNIST)
# =============================================================================

def get_unet_checkpoint_path(epoch: int) -> Path:
    """Get path to UNet checkpoint for a specific epoch."""
    return CHECKPOINTS_DIR / f"{UNET_CHECKPOINT_PREFIX}{epoch}.pt"


def get_latest_unet_checkpoint() -> Path:
    """Find the latest UNet checkpoint by epoch number."""
    checkpoints = list(CHECKPOINTS_DIR.glob(f"{UNET_CHECKPOINT_PREFIX}*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No UNet checkpoints found in {CHECKPOINTS_DIR}")
    return max(checkpoints, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))


def get_classifier_checkpoint_path() -> Path:
    """Get path to MNIST classifier checkpoint."""
    return CHECKPOINTS_DIR / CLASSIFIER_CHECKPOINT_NAME


def get_generated_images_dir(guidance_scale: int) -> Path:
    """Get directory for generated images at a specific guidance scale."""
    return EXPERIMENT_1_GENERATED_DIR / f"guidance_{guidance_scale}"


def get_digit_dir(base_dir: Path, digit: int) -> Path:
    """Get directory for a specific digit within a base directory."""
    return base_dir / f"digit_{digit}"


def ensure_experiment_dirs():
    """Create all experiment 1 directories if they don't exist."""
    dirs = [
        EXPERIMENT_1_DIR,
        EXPERIMENT_1_DATASET_DIR,
        EXPERIMENT_1_GENERATED_DIR,
        EXPERIMENT_1_METRICS_DIR,
        FIGURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create digit subdirs for dataset
    for digit in EXPERIMENT_1_CONFIG["digits"]:
        get_digit_dir(EXPERIMENT_1_DATASET_DIR, digit).mkdir(parents=True, exist_ok=True)
    
    # Create guidance scale subdirs with digit subdirs
    for guidance in EXPERIMENT_1_CONFIG["guidance_scales"]:
        guidance_dir = get_generated_images_dir(guidance)
        for digit in EXPERIMENT_1_CONFIG["digits"]:
            get_digit_dir(guidance_dir, digit).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions - Experiment 2 (CIFAR-10)
# =============================================================================

def get_cifar10_unet_checkpoint_path(epoch: int) -> Path:
    """Get path to CIFAR-10 UNet checkpoint for a specific epoch."""
    return CHECKPOINTS_DIR / f"{UNET_CIFAR10_CHECKPOINT_PREFIX}{epoch}.pt"


def get_latest_cifar10_unet_checkpoint() -> Path:
    """Find the latest CIFAR-10 UNet checkpoint by epoch number."""
    checkpoints = list(CHECKPOINTS_DIR.glob(f"{UNET_CIFAR10_CHECKPOINT_PREFIX}*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No CIFAR-10 UNet checkpoints found in {CHECKPOINTS_DIR}")
    return max(checkpoints, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))


def get_cifar10_generated_images_dir(guidance_scale: int) -> Path:
    """Get directory for generated CIFAR-10 images at a specific guidance scale."""
    return EXPERIMENT_2_GENERATED_DIR / f"guidance_{guidance_scale}"


def get_class_dir(base_dir: Path, class_idx: int) -> Path:
    """Get directory for a specific class within a base directory."""
    class_name = CIFAR10_CLASSES[class_idx]
    return base_dir / f"class_{class_idx}_{class_name}"


def ensure_experiment_2_dirs():
    """Create all experiment 2 directories if they don't exist."""
    dirs = [
        EXPERIMENT_2_DIR,
        EXPERIMENT_2_DATASET_DIR,
        EXPERIMENT_2_GENERATED_DIR,
        EXPERIMENT_2_METRICS_DIR,
        FIGURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create class subdirs for dataset
    for class_idx in EXPERIMENT_2_CONFIG["classes"]:
        get_class_dir(EXPERIMENT_2_DATASET_DIR, class_idx).mkdir(parents=True, exist_ok=True)
    
    # Create guidance scale subdirs with class subdirs
    for guidance in EXPERIMENT_2_CONFIG["guidance_scales"]:
        guidance_dir = get_cifar10_generated_images_dir(guidance)
        for class_idx in EXPERIMENT_2_CONFIG["classes"]:
            get_class_dir(guidance_dir, class_idx).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper Functions - Experiment 3 (WikiArt)
# =============================================================================

def get_wikiart_unet_checkpoint_path(epoch: int) -> Path:
    """Get path to WikiArt UNet checkpoint for a specific epoch."""
    return CHECKPOINTS_DIR / f"{UNET_WIKIART_CHECKPOINT_PREFIX}{epoch}.pt"


def get_latest_wikiart_unet_checkpoint() -> Path:
    """Find the latest WikiArt UNet checkpoint by epoch number."""
    checkpoints = list(CHECKPOINTS_DIR.glob(f"{UNET_WIKIART_CHECKPOINT_PREFIX}*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No WikiArt UNet checkpoints found in {CHECKPOINTS_DIR}")
    return max(checkpoints, key=lambda x: int(str(x).split("_")[-1].split(".")[0]))


def get_wikiart_generated_images_dir(guidance_scale: int) -> Path:
    """Get directory for generated WikiArt images at a specific guidance scale."""
    return EXPERIMENT_3_GENERATED_DIR / f"guidance_{guidance_scale}"


def get_style_dir(base_dir: Path, style_idx: int) -> Path:
    """Get directory for a specific art style within a base directory."""
    style_name = WIKIART_STYLES[style_idx]
    return base_dir / f"style_{style_idx}_{style_name}"


def ensure_experiment_3_dirs():
    """Create all experiment 3 directories if they don't exist."""
    dirs = [
        EXPERIMENT_3_DIR,
        EXPERIMENT_3_DATASET_DIR,
        EXPERIMENT_3_GENERATED_DIR,
        EXPERIMENT_3_METRICS_DIR,
        FIGURES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create style subdirs for dataset
    for style_idx in EXPERIMENT_3_CONFIG["classes"]:
        get_style_dir(EXPERIMENT_3_DATASET_DIR, style_idx).mkdir(parents=True, exist_ok=True)
    
    # Create guidance scale subdirs with style subdirs
    for guidance in EXPERIMENT_3_CONFIG["guidance_scales"]:
        guidance_dir = get_wikiart_generated_images_dir(guidance)
        for style_idx in EXPERIMENT_3_CONFIG["classes"]:
            get_style_dir(guidance_dir, style_idx).mkdir(parents=True, exist_ok=True)
