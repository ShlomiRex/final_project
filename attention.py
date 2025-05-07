import torch
from torch import nn
from torch.nn import functional as F
import einops
import math

class SelfAttention(nn.Module):
    """
    in_proj_bias = If to apply bias before linear projection
    out_proj_bias = If to apply bias after linear projection
    """
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super.__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        input_shape = x.shape
        # x: (b seq_len dim)
        b, seq_len, d_embed = x.shape

        # Instead of applying three different projections we can apply linear projection on all X and then split it using chunk
        # x: (b seq_len dim) -> (b seq_len dim*3) --chunk--> three tensors of shape (b seq_len dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (b seq_len dim) -> (b seq_len n_heads d_head) --transpose--> (b n_heads seq_len d_head)
        q = q.view((b, seq_len, self.n_heads, self.d_head)).transpose(1, 2)
        k = k.view((b, seq_len, self.n_heads, self.d_head)).transpose(1, 2)
        v = v.view((b, seq_len, self.n_heads, self.d_head)).transpose(1, 2)

        # (b n_heads seq_len seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # above principal diagonal, the cells are 1 and others are 0
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # (b n_heads seq_len seq_len) @ (b n_heads seq_len d_head) -> (b n_heads seq_len d_head)
        output = weight @ v

        # (b n_heads seq_len d_head) -> (b seq_len n_heads d_head)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)


        output = self.out_proj(output)

        # b seq_len dim
        return output



