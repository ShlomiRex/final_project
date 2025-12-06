"""
Configuration management using dataclasses and YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from pathlib import Path
import yaml


@dataclass
class VQVAEConfig:
    """VQ-VAE configuration."""
    source: Literal["pretrained_hf", "pretrained_taming", "custom"] = "pretrained_hf"
    checkpoint: str = "dalle-mini/vqgan_imagenet_f16_16384"
    codebook_size: int = 16384
    downsample_factor: int = 16


@dataclass
class TransformerConfig:
    """Transformer configuration."""
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    ffn_dim: Optional[int] = None  # Default: 4 * hidden_size
    dropout: float = 0.1
    max_seq_len: int = 256  # 16x16 for 256x256 images with f=16


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    model_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77
    hidden_size: int = 512  # CLIP ViT-B/32


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    
    # Checkpointing
    checkpoint_interval: int = 5000
    eval_interval: int = 1000
    log_interval: int = 100
    
    # CFG
    cfg_dropout: float = 0.1


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "flickr30k"
    image_size: int = 256
    num_workers: int = 4
    cache_dir: Optional[str] = None


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    experiment_name: str = "latent-gpt-pretrained-vqvae"
    tracking_uri: str = "http://127.0.0.1:5000"
    run_name: Optional[str] = None
    tags: dict = field(default_factory=dict)


@dataclass
class Config:
    """
    Complete configuration for training.
    
    Example:
        >>> config = Config.from_yaml("configs/base.yaml")
        >>> config.training.learning_rate = 5e-5
        >>> config.save("configs/custom.yaml")
    """
    vqvae: VQVAEConfig = field(default_factory=VQVAEConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    
    # Paths
    output_dir: str = "./outputs"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            vqvae=VQVAEConfig(**data.get("vqvae", {})),
            transformer=TransformerConfig(**data.get("transformer", {})),
            text_encoder=TextEncoderConfig(**data.get("text_encoder", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
            mlflow=MLflowConfig(**data.get("mlflow", {})),
            output_dir=data.get("output_dir", "./outputs"),
            seed=data.get("seed", 42),
        )
    
    def save(self, path: str | Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "vqvae": asdict(self.vqvae),
            "transformer": asdict(self.transformer),
            "text_encoder": asdict(self.text_encoder),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "mlflow": asdict(self.mlflow),
            "output_dir": self.output_dir,
            "seed": self.seed,
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert to flat dictionary for MLflow logging."""
        return {
            # VQ-VAE
            "vqvae_source": self.vqvae.source,
            "vqvae_checkpoint": self.vqvae.checkpoint,
            "codebook_size": self.vqvae.codebook_size,
            "downsample_factor": self.vqvae.downsample_factor,
            
            # Transformer
            "transformer_hidden": self.transformer.hidden_size,
            "transformer_layers": self.transformer.num_layers,
            "transformer_heads": self.transformer.num_heads,
            "max_seq_len": self.transformer.max_seq_len,
            
            # Text encoder
            "text_encoder": self.text_encoder.model_name,
            
            # Training
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "mixed_precision": self.training.mixed_precision,
            "cfg_dropout": self.training.cfg_dropout,
            
            # Data
            "dataset": self.data.dataset,
            "image_size": self.data.image_size,
            
            # General
            "seed": self.seed,
        }
