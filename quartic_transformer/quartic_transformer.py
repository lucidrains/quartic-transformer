import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

import einx.nn.torch as einn

# coordinate descent routing

from colt5_attention import topk

# taylor series linear attention

from taylor_series_linear_attention import TaylorSeriesLinearAttn

# helpers

def exists(v):
    return v is not None

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        flash = True,
        causal = False
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.rmsnorm = einn.Norm('b... [d]', mean = False, bias = False)

        self.scale = dim_head ** 0.5
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        x = self.rmsnorm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out)

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        einn.Norm('b... [d]', mean = False, bias = False),
        nn.Linear(dim, dim_inner, bias = False),
        nn.GELU(),
        nn.Linear(dim_inner, dim, bias = False)
    )

# main class

class QuarticTransformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(self, x):
        return x
