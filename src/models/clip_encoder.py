"""
CLIP Text Encoder Wrapper

Wraps the HuggingFace CLIP text encoder for use with LatentGPT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Union, List
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPEncoder(nn.Module):
    """
    CLIP text encoder wrapper.
    
    Encodes text prompts to embeddings for cross-attention conditioning.
    
    Args:
        model_name: HuggingFace model name (default: "openai/clip-vit-base-patch32")
        max_length: Maximum token length (default: 77)
        device: Device to load model on
        
    Example:
        >>> encoder = CLIPEncoder()
        >>> embeddings = encoder(["a photo of a cat", "a dog running"])
        >>> print(embeddings.shape)  # [2, 77, 512]
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        
        # Freeze encoder
        self.model.requires_grad_(False)
        self.model.eval()
        
        self._device = device
    
    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the text encoder."""
        return self.model.config.hidden_size
    
    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return self._device
    
    def to(self, device: torch.device) -> "CLIPEncoder":
        """Move model to device."""
        self.model = self.model.to(device)
        self._device = device
        return self
    
    @torch.no_grad()
    def forward(
        self,
        text: Union[str, List[str]],
        return_pooled: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode text to embeddings.
        
        Args:
            text: Text string or list of strings
            return_pooled: If True, also return pooled output
            
        Returns:
            If return_pooled is False:
                Text embeddings [B, max_length, hidden_size]
            If return_pooled is True:
                Tuple of (embeddings, pooled_output)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = tokens.input_ids.to(self._device)
        attention_mask = tokens.attention_mask.to(self._device)
        
        # Encode
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        embeddings = outputs.last_hidden_state
        
        if return_pooled:
            return embeddings, outputs.pooler_output
        
        return embeddings
    
    def encode_with_dropout(
        self,
        text: Union[str, List[str]],
        dropout_prob: float = 0.1,
    ) -> torch.Tensor:
        """
        Encode text with random dropout for classifier-free guidance training.
        
        Args:
            text: Text string or list of strings
            dropout_prob: Probability of replacing text with empty string
            
        Returns:
            Text embeddings [B, max_length, hidden_size]
        """
        if isinstance(text, str):
            text = [text]
        
        # Apply dropout
        if self.training:
            text = [
                "" if torch.rand(1).item() < dropout_prob else t
                for t in text
            ]
        
        return self.forward(text)
    
    def get_empty_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get embedding for empty string (for unconditional generation).
        
        Args:
            batch_size: Number of embeddings to return
            
        Returns:
            Empty text embeddings [batch_size, max_length, hidden_size]
        """
        return self.forward([""] * batch_size)
    
    def get_mlflow_params(self) -> dict:
        """Return parameters for MLflow logging."""
        return {
            "text_encoder": self.model_name,
            "text_encoder_hidden_size": self.hidden_size,
            "text_encoder_max_length": self.max_length,
        }
