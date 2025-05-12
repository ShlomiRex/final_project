import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
import einops

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b in_channels h w)
        residue = x
        
        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)



class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super.__init__()
        
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b c h w)

        residue = x 

        b, c, h, w = x.shape()

        # TODO: Check this is correct
        x2 = einops.rearrange(x, 'b c h w -> b c (h w)')

        # b c h w -> b c h*w
        x = x.view(b, c, h*w) # ok

        assert torch.equal(x, x2)



        # TODO: Check this is correct
        x2 = einops.rearrange(x, 'b c (h w) -> b (h w) c')

        # b c h*w -> b h*w c
        x = x.transpose(-1, -2) # ok

        assert torch.equal(x, x2)




        x = self.attention(x)



        # TODO: Check this is correct
        x2 = einops.rearrange(x, 'b (h w) c -> b c (h w)')

        # b h*w c -> b c h*w
        x = x.transpose(-1, -2)

        assert torch.equal(x, x2)


        # TODO: Check this is correct
        x2 = einops.rearrange(x, 'b c (h w) -> b c h w')

        # b c h*w -> b c h w
        x = x.view((b, c, h, w))

        assert torch.equal(x, x2)



        x += residue

        return x
