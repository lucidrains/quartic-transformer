import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import einx
import einx.nn.torch as einn

# coordinate descent routing

from colt5_attention import topk

# taylor series linear attention

from taylor_series_linear_attention import TaylorSeriesLinearAttn

# dynamic positional bias from x-transformers

from x_transformers.x_transformers import DynamicPositionBias

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_edges = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        incorporate_edges = True
    ):
        super().__init__()
        dim_edges = default(dim_edges, dim)
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

        self.rmsnorm = einn.Norm('b... [d]', mean = False, bias = False)

        self.scale = dim_head ** 0.5
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.edges_to_attn_bias = None

        if incorporate_edges:
            self.edges_to_attn_bias = nn.Sequential(
                einn.Norm('b... [d]', mean = False, bias = False),
                nn.Linear(dim_edges, heads),
                Rearrange('b i j h -> b h i j')
            )

        self.pre_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        self.to_edges_out = None

        if incorporate_edges:
            self.to_edges_out = nn.Sequential(
                nn.Conv2d(heads, dim_edges, 1, bias = False),
                Rearrange('b d i j -> b i j d')
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

        if exists(edges) and exists(self.edges_to_attn_bias):
            attn_bias = self.edges_to_attn_bias(edges)
            sim = sim + attn_bias

        sim = self.pre_talking_heads(sim)

        if exists(mask):
            sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = einx.softmax('b h i [j]', sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = out * self.to_gates(x)
        out = self.to_out(out)

        edges_out = None
        if exists(self.to_edges_out):
            edges_out = self.to_edges_out(attn)

        if not exists(edges_out):
            return out

        return out, edges_out

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

# edge embed

class EdgeEmbed(Module):
    def __init__(self, dim, dim_edges = None):
        super().__init__()
        dim_edges = default(dim_edges, dim)
        self.to_rows = nn.Linear(dim, dim_edges, bias = False)
        self.to_cols = nn.Linear(dim, dim_edges, bias = False)

        self.to_edges = nn.Sequential(
            nn.Linear(dim_edges, dim_edges, bias = False),
            nn.LayerNorm(dim_edges)
        )

    def forward(self, x):
        rows = self.to_rows(x)
        cols = self.to_cols(x)
        outer_sum = einx.add('b i d, b j d -> b i j d', rows, cols)
        return self.to_edges(outer_sum)

# axial linear attention

class AxialLinearAttention(Module):
    def __init__(
        self,
        dim,
        diagonal_attn = True,
        **attn_kwargs
    ):
        super().__init__()

        self.row_attn = TaylorSeriesLinearAttn(dim = dim, gate_value_heads = True, prenorm = True, **attn_kwargs)
        self.col_attn = TaylorSeriesLinearAttn(dim = dim, gate_value_heads = True, prenorm = True, **attn_kwargs)

        self.diagonal_attn = Attention(dim = dim, incorporate_edges = False, **attn_kwargs) if diagonal_attn else None

    def forward(
        self,
        x,
        mask = None
    ):
        b, n, device = *x.shape[:2], x.device

        x = rearrange(x, 'b i j d -> (b i) j d')

        x = self.row_attn(x, mask = mask) + x

        x = rearrange(x, '(b i) j d -> (b j) i d', b = b)

        x = self.col_attn(x, mask = mask) + x

        x = rearrange(x, '(b j) i d -> b i j d', b = b)

        if not exists(self.diagonal_attn):
            return x

        diagonal_mask = torch.eye(n, dtype = torch.bool, device = device)
        diagonal_mask = rearrange(diagonal_mask, 'i j -> 1 i j')

        x = rearrange(x[diagonal_mask], '(b n) d -> b n d', b = b)

        x = self.diagonal_attn(x) + x

        diagonal_mask = rearrange(diagonal_mask, '... -> ... 1')
        x = x.masked_scatter(diagonal_mask, x)
        return x

# main class

class QuarticTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_edges = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        linear_dim_head = 16,
        linear_heads = 16,
        ff_mult = 4,
        dropout = 0.,
        max_seq_len = 2048,
        ablate_edges = False,
        edges_diagonal_attn = True
    ):
        super().__init__()
        dim_edges = default(dim_edges, dim)

        self.ablate_edges = ablate_edges
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.dynamic_rel_pos_bias = DynamicPositionBias(dim, depth = 2, heads = dim_edges)

        self.to_edge_emb = EdgeEmbed(dim, dim_edges)

        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                ModuleList([
                    Attention(dim = dim, dim_edges = dim_edges, dim_head = dim_head, heads = heads, dropout = dropout, causal = causal),
                    FeedForward(dim = dim, mult = ff_mult, dropout = dropout)
                ]),
                ModuleList([
                    AxialLinearAttention(dim = dim_edges, dim_head = linear_dim_head, heads = linear_heads, causal = causal, diagonal_attn = edges_diagonal_attn),
                    FeedForward(dim = dim_edges, mult = ff_mult)
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
        seq_len, device = x.shape[-1], x.device
        assert seq_len <= self.max_seq_len

        x = self.token_emb(x)

        x = x + self.pos_emb(torch.arange(seq_len, device = device))
        edges = self.to_edge_emb(x)

        edges_rel_pos = self.dynamic_rel_pos_bias(seq_len, seq_len)
        edges = einx.add('b i j d, d i j -> b i j d', edges, edges_rel_pos)

        edges_mask = None
        if exists(mask):
            edges_mask = einx.logical_and('b i, b j -> b (i j)', mask, mask)

        for (attn, ff), (edges_linear_attn, edges_ff,) in self.layers:

            nodes_out, edges_out = attn(x, mask = mask, edges = edges if not self.ablate_edges else None)

            x = x + nodes_out
            x = ff(x) + x

            if self.ablate_edges:
                continue

            edges = edges + edges_out

            edges = edges_linear_attn(edges, mask = mask) + edges

            edges = edges_ff(edges) + edges

        return self.to_logits(x)
