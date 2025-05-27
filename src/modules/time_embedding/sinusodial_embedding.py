import torch
import math

# From: https://github.com/hkproj/pytorch-stable-diffusion/blob/e0cb06de011787cdf13eed7b4287ad8410491149/sd/pipeline.py#L164
def get_time_embedding(timestep_scalar: int, embed_half_dim: int = 160):
    # embed_half_dim is half the embedding dimension. We return tensor of shape (1, embed_half_dim * 2)

    freqs = torch.pow(10000, -torch.arange(start=0, end=embed_half_dim, dtype=torch.float32) / embed_half_dim) # (embed_half_dim,)
    x = torch.tensor([timestep_scalar], dtype=torch.float32)[:, None] * freqs[None] # (1, embed_half_dim)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1) # (1, embed_dim)

# My version
def get_time_embeddings(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    assert embedding_dim % 2 == 0, "embedding_dim must be even"

    # Create the frequency spectrum
    half_dim = embedding_dim // 2
    exponent = -math.log(10000.0) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent)

    # Expand timesteps for broadcasting
    timesteps = timesteps.float().unsqueeze(1)  # (N, 1)
    args = timesteps * freq.unsqueeze(0)        # (N, half_dim)

    # Concatenate sin and cos
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (N, embedding_dim)

    return embedding

class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        assert self.embedding_dim % 2 == 0, "embedding_dim must be even"

        # Create the frequency spectrum
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent)
        freq = freq.to(self.device)

        # Expand timesteps for broadcasting
        timesteps = timesteps.float().unsqueeze(1)  # (N, 1)
        args = timesteps * freq.unsqueeze(0)        # (N, half_dim)

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (N, embedding_dim)

        return embedding