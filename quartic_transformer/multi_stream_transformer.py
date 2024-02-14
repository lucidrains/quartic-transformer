import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import einx
import einx.nn.torch as einn

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        num_streams,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        pre_talking_heads = False,
        post_talking_heads = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        all_heads = num_streams * heads

        self.num_streams = num_streams

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

        self.pre_talking_heads = nn.Conv2d(all_heads, all_heads, 1, bias = False) if pre_talking_heads else None
        self.post_talking_heads = nn.Conv2d(all_heads, all_heads, 1, bias = False) if post_talking_heads else None

        nn.init.dirac_(self.pre_talking_heads.weight)
        nn.init.dirac_(self.post_talking_heads.weight)

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
        s = self.num_streams
        x = self.rmsnorm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(self.pre_talking_heads):
            sim = rearrange(sim, '(b s) h n d -> b (s h) n d', s = s)
            sim = self.pre_talking_heads(sim)
            sim = rearrange(sim, 'b (s h) n d -> (b s) h n d', s = s)

        if exists(mask):
            sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = einx.softmax('b h i [j]', sim)
        attn = self.dropout(attn)

        if exists(self.post_talking_heads):
            attn = rearrange(attn, '(b s) h n d -> b (s h) n d', s = s)
            attn = self.post_talking_heads(attn)
            attn = rearrange(attn, 'b (s h) n d -> (b s) h n d', s = s)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = out * self.to_gates(x)
        out = self.to_out(out)

        return out

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

# classes

class MultiStreamTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        num_streams = 2,
        dim_head = 64,
        heads = 8,
        max_seq_len = 2048,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4.,
        ablate_cross_stream_talking_heads = False,
        pre_talking_heads = True,
        post_talking_heads = True
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.num_streams = num_streams
        self.stream_emb = nn.Parameter(torch.randn(num_streams, dim))

        self.layers = ModuleList([])

        talking_heads_num_streams = 2 if not ablate_cross_stream_talking_heads else 1

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = attn_dropout,
                    num_streams = talking_heads_num_streams,
                    pre_talking_heads = pre_talking_heads,
                    post_talking_heads = post_talking_heads
                ),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
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
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        x = einx.add('b n d, s d -> (b s) n d', x, self.stream_emb)

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        x = reduce(x, '(b s) n d -> b n d', 'sum', s = self.num_streams)

        return self.to_logits(x)
