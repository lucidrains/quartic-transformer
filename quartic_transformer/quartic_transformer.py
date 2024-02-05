import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

import einx
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

        self.edges_to_attn_bias = nn.Sequential(
            einn.Norm('b... [d]', mean = False, bias = False),
            nn.Linear(dim, heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        mask = None,
        edges = None
    ):
        x = self.rmsnorm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            sim = sim + attn_bias

        if exists(mask):
            sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = einx.softmax('b h i [j]', sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out)

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        einn.Norm('b... [d]', mean = False, bias = False),
        nn.Linear(dim, dim_inner, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim, bias = False)
    )

# main class

class QuarticTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.to_edges = nn.Linear(dim, dim)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ModuleList([
                    Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, causal = causal),
                    FeedForward(dim = dim, mult = ff_mult, dropout = dropout)
                ]),
                ModuleList([
                    FeedForward(dim = dim, mult = ff_mult)
                ])
            ]))

        self.to_logits = nn.Sequential(
            einn.Norm('b... [d]', mean = False, bias = False),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        x = self.token_emb(x)

        edges = einx.add('b i d, b j d -> b i j d', x, x)
        edges = self.to_edges(edges)

        for (attn, ff), (edges_ff,) in self.layers:
            x = attn(x, mask = mask, edges = edges) + x
            x = ff(x) + x

            edges = edges_ff(edges) + edges

        return self.to_logits(x)

