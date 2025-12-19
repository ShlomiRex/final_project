"""
CLIP Text Encoder Wrapper for Stable Diffusion.

Provides a consistent interface for encoding text prompts to embeddings
used for cross-attention conditioning in the U-Net.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn


class CLIPTextEncoderWrapper(nn.Module):
    """
    Wrapper for CLIP text encoder.
    
    Encodes text prompts to embeddings for cross-attention
    conditioning in the diffusion model.
    
    Note: The text encoder is frozen during training.
    """
    
    def __init__(
        self,
        pretrained: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.pretrained = pretrained
        self.max_length = max_length
        self._device = device
        
        # Load pretrained model and tokenizer
        self.tokenizer, self.text_encoder = self._load_pretrained(pretrained)
        
        # Move to device if specified
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
        
        # Freeze text encoder
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        # Create null token embedding for unconditional generation
        self._null_tokens = None
        self._null_embedding = None
    
    def _load_pretrained(self, pretrained: str):
        """Load pretrained CLIP text encoder."""
        from transformers import CLIPTextModel, CLIPTokenizer
        
        tokenizer = CLIPTokenizer.from_pretrained(pretrained)
        text_encoder = CLIPTextModel.from_pretrained(pretrained)
        
        return tokenizer, text_encoder
    
    def tokenize(
        self,
        prompts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """
        Tokenize text prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
        
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        tokens = self.tokenizer(
            prompts,
            padding="max_length" if padding else True,
            max_length=self.max_length,
            truncation=truncation,
            return_tensors="pt",
        )
        
        return tokens
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokenized text to embeddings.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
        
        Returns:
            Text embeddings [B, L, D]
        """
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Use last hidden state for cross-attention
            hidden_states = outputs.last_hidden_state
            
            return hidden_states
    
    def encode_prompts(
        self,
        prompts: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        Encode text prompts to embeddings (tokenize + encode).
        
        Args:
            prompts: Single prompt or list of prompts
        
        Returns:
            Text embeddings [B, L, D]
        """
        tokens = self.tokenize(prompts)
        
        # Move to correct device
        device = next(self.text_encoder.parameters()).device
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        return self.encode(input_ids, attention_mask)
    
    def get_null_embedding(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get null embedding for unconditional generation.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Null embeddings [B, L, D]
        """
        device = next(self.text_encoder.parameters()).device
        
        # Cache null embedding
        if self._null_embedding is None or self._null_embedding.device != device:
            null_tokens = self.tokenize([""])
            self._null_tokens = null_tokens["input_ids"].to(device)
            self._null_embedding = self.encode(self._null_tokens)
        
        # Expand to batch size
        return self._null_embedding.expand(batch_size, -1, -1)
    
    def get_embeddings_for_cfg(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Get embeddings for classifier-free guidance.
        
        Returns concatenated [unconditional, conditional] embeddings.
        
        Args:
            prompts: List of prompts
            negative_prompts: Optional list of negative prompts
        
        Returns:
            Embeddings [2*B, L, D] - first half unconditional, second half conditional
        """
        batch_size = len(prompts)
        
        # Get conditional embeddings
        cond_embeddings = self.encode_prompts(prompts)
        
        # Get unconditional embeddings
        if negative_prompts is None:
            uncond_embeddings = self.get_null_embedding(batch_size)
        else:
            uncond_embeddings = self.encode_prompts(negative_prompts)
        
        # Concatenate for batched CFG inference
        return torch.cat([uncond_embeddings, cond_embeddings], dim=0)
    
    @property
    def hidden_size(self) -> int:
        """Hidden size of text embeddings."""
        return self.text_encoder.config.hidden_size
    
    def to(self, device: torch.device):
        """Move encoder to device."""
        self.text_encoder = self.text_encoder.to(device)
        self._device = device
        
        # Invalidate cached null embedding
        self._null_embedding = None
        self._null_tokens = None
        
        return self
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - alias for encode."""
        return self.encode(input_ids, attention_mask)


def load_text_encoder(
    pretrained: str = "openai/clip-vit-large-patch14",
    device: Optional[torch.device] = None,
) -> CLIPTextEncoderWrapper:
    """
    Load a pretrained CLIP text encoder.
    
    Args:
        pretrained: HuggingFace model ID
        device: Device to load model on
    
    Returns:
        CLIPTextEncoderWrapper instance
    """
    encoder = CLIPTextEncoderWrapper(pretrained=pretrained, device=device)
    
    if device is not None:
        encoder = encoder.to(device)
    
    return encoder
