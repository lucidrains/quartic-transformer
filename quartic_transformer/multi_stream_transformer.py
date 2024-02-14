"""
einstein notation
b - batch
s - stream
n - seq len
d - feature dimension
"""

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

import einx
from einx import get_at
import einx.nn.torch as einn

# helper

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp(min = eps)

def entropy(prob, dim = -1, keepdim = False):
    return (-prob * log(prob)).sum(dim = dim, keepdim = keepdim)

def calc_stream_loss(attn_matrix, streams):
    attn_matrix = rearrange(attn_matrix, '(b s) h i j -> s b h i j', s = streams)

    # make sure all heads across one stream is dissimilar to all other heads of all other streams

    accum, *rests = attn_matrix
    for rest in rests:
        accum = einx.add('b h1 ..., b h2 ... -> b (h1 h2) ...', accum, rest)

    mean_prob = accum / streams
    return -entropy(mean_prob, dim = -1)

# residual

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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
        post_talking_heads = False,
        non_linear_talking_heads = False
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

        self.pre_talking_heads = None
        self.post_talking_heads = None

        if non_linear_talking_heads:
            self.pre_talking_heads = TalkingHeadsFeedForward(all_heads) if pre_talking_heads else None
            self.post_talking_heads = TalkingHeadsFeedForward(all_heads) if post_talking_heads else None
        else:
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

        post_softmax_attn = attn

        attn = self.dropout(attn)

        if exists(self.post_talking_heads):
            attn = rearrange(attn, '(b s) h n d -> b (s h) n d', s = s)
            attn = self.post_talking_heads(attn)
            attn = rearrange(attn, 'b (s h) n d -> (b s) h n d', s = s)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = out * self.to_gates(x)
        out = self.to_out(out)

        return out, post_softmax_attn

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

def TalkingHeadsFeedForward(dim, mult = 2, dropout = 0.):
    dim_inner = int(dim * mult)
    net = nn.Sequential(
        einn.Norm('b [c] ...', mean = False, bias = False),
        nn.Conv2d(dim, dim_inner, 1, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim_inner, dim, 1, bias = False)
    )

    nn.init.zeros_(net[-1].weight)
    return Residual(net)

# embedding types
# streams can either have (1) shared embeddings with a stream specific embedding added
# or (2) separate token / abs pos emb

class TokenAndPosEmb(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        num_streams
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.stream_emb = nn.Parameter(torch.zeros(num_streams, dim))
        nn.init.normal_(self.stream_emb, std = 0.02)

    def forward(self, x):
        seq_len = torch.arange(x.shape[-1], device = x.device)

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(seq_len)

        return einx.add('b n d, n d, s d -> (b s) n d', token_emb, pos_emb, self.stream_emb)

class SeparateTokenAndPosEmb(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        num_streams
    ):
        super().__init__()
        self.token_emb = nn.Parameter(torch.zeros(num_streams, num_tokens, dim))
        self.pos_emb = nn.Parameter(torch.zeros(num_streams, max_seq_len, dim))

        nn.init.normal_(self.token_emb, std = 0.02)
        nn.init.normal_(self.pos_emb, std = 0.02)

    def forward(self, x):
        seq_len = torch.arange(x.shape[-1], device = x.device)

        token_emb = get_at('s [e] d, b n -> b s n d', self.token_emb, x)
        pos_emb = get_at('s [e] d, n -> s n d', self.pos_emb, x)

        return einx.add('b s n d, s n d -> (b s) n d', token_emb, pos_emb)

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
        post_talking_heads = True,
        separate_stream_emb = True,
        non_linear_talking_heads = False
    ):
        super().__init__()
        embed_klass = SeparateTokenAndPosEmb if separate_stream_emb else TokenAndPosEmb

        self.emb = embed_klass(
            dim = dim,
            num_tokens = num_tokens,
            num_streams = num_streams,
            max_seq_len = max_seq_len
        )

        self.num_streams = num_streams
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
                    post_talking_heads = post_talking_heads,
                    non_linear_talking_heads = non_linear_talking_heads
                ),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.to_logits = nn.Sequential(
            Reduce('(b s) n d -> b n d', 'sum', s = num_streams),
            einn.Norm('b... [d]', mean = False, bias = False),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        mask = None,
        stream_attn_diversity_loss = False
    ):
        b, n, s, device = *x.shape, self.num_streams, x.device

        stream_attn_diversity_loss &= s > 1

        x = self.emb(x)

        attn_matrices = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x, mask = mask)

            attn_matrices.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if stream_attn_diversity_loss:
            aux_loss = sum([calc_stream_loss(attn_matrix, s).mean() for attn_matrix in attn_matrices])

        logits = self.to_logits(x)

        if not stream_attn_diversity_loss:
            return logits

        return logits, aux_loss
