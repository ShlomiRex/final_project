import torch
from torch import nn
from torch.nn import functional as F
from vae_blocks import ResBlock, AttentionBlock
import einops

class Encoder(nn.Sequential):
    def __init__(self):
        super(
            # b c h w -> b 128 h w
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # b 128 h w -> b 128 h w
            ResBlock(128, 128),

            # b 128 h w -> b 128 h w
            ResBlock(128, 128),

            # b 128 h w -> b 128 h/2 w/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # b 128 h/2 w/2 -> b 256 h/2 w/2
            ResBlock(128, 256),

            # b 256 h/2 w/2 -> b 256 h/2 w/2
            ResBlock(256, 256),

            # b 256 h/2 w/2 -> b 256 h/4 w/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # b 256 h/4 w/4 -> b 512 h/4 w/4
            ResBlock(256, 512),

            # b 512 h/4 w/4 -> b 512 h/4 w/4
            ResBlock(512, 512),

            # b 512 h/4 w/4 -> b 512 h/8 w/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # b 512 h/8 w/8 -> b 512 h/8 w/8
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),

            # b 512 h/8 w/8 -> b 512 h/8 w/8
            AttentionBlock(512),

            # b 512 h/8 w/8 -> b 512 h/8 w/8
            ResBlock(512, 512),

            # b 512 h/8 w/8 -> b 512 h/8 w/8
            nn.GroupNorm(32, 512),

            # b 512 h/8 w/8 -> b 512 h/8 w/8
            nn.SiLU(), # The researchers didn't explain why they chose SiLU, it just practically works well

            # b 512 h/8 w/8 -> b 8 h/8 w/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # b 8 h/8 w/8 -> b 8 h/8 w/8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    """
    Encodes an image X
    x: image to encode
    """
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (b c h w)
        # noise: (b c h/8 w/8)

        for module in self:
            # Apply padding on bottom, and right side of image.
            # If stride == (2, 2) [this is the only case] we don't apply padding on bottom or right, and because we want symmetrical padding, we manually do it
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x) # apply layers sequentially
        
        # b 8 h/8 w/8 -> Two tensors of shape: (b 4 h/8 w/8)
        mean, log_var = torch.chunk(x, 2, dim=1)

        # TODO: Check einops is the same output as before, I want to replace .chunk with this
        mean2, log_var2 = einops.rearrange(x, 'b (c p) h w -> p b c h w', p=2)

        assert torch.equal(mean, mean2)
        assert torch.equal(log_var, log_var2)

        # Make log_var bigger
        log_var = torch.clamp(log_var, -30, 20)
        
        # Remove log
        var = log_var.exp()

        # Calculate standard deviation
        stdev = var.sqrt()

        # The VAE formula, the sampling part
        x = mean + stdev * noise

        # Scale by a constant (the repository doesn't explain why, it just works)
        x *= 0.18215

        return x