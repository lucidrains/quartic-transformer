import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

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

class RelationalAttention(Module):
    def __init__(
        self,
        dim,
        dim_edges,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
    ):
        super().__init__()
        self.scale = dim_head ** 0.5
        self.causal = causal

        self.norm = nn.RMSNorm(dim)
        self.edges_norm = nn.RMSNorm(dim_edges)

        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.to_edges_qkv = nn.Sequential(
            nn.Linear(dim_edges, dim_inner * 3, bias = False),
            Rearrange('b ... (qkv h d) -> qkv b h ... d', h = heads, qkv = 3)
        )

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        *,
        edges,
        mask = None,
    ):
        x = self.norm(x)
        edges = self.edges_norm(edges)

        # projection to queries, keys, values for both nodes and edges

        nodes_qkv = self.to_qkv(x)

        edges_qkv = self.to_edges_qkv(edges)

        # add node and edges contributions

        q, k, v = tuple(einx.add('b h j d, b h i j d -> b h i j d', nt, et) for nt, et in zip(nodes_qkv, edges_qkv))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum('b h i j d, b h i j d -> b h i j', q, k)

        # masking

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention

        attn = einx.softmax('b h i [j]', sim)

        # dropout

        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)

        out = self.to_out(out)
        return out

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult)

    return nn.Sequential(
        nn.RMSNorm(dim),
        nn.Linear(dim, dim_inner, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim, bias = False)
    )

# edge embed

class EdgeEmbed(Module):
    def __init__(
        self,
        dim,
        dim_edges
    ):
        super().__init__()
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

        out = self.to_edges(outer_sum)
        return out

class NodesToEdges(Module):
    def __init__(
        self,
        *,
        dim_nodes,
        dim_edges
    ):
        super().__init__()
        self.node_norm = nn.RMSNorm(dim_nodes)
        self.edges_norm = nn.RMSNorm(dim_edges)

        self.proj = nn.Linear(dim_nodes * 2 + dim_edges, dim_edges)

    def forward(
        self,
        edges,
        nodes
    ):
        nodes = self.node_norm(nodes)
        edges = self.edges_norm(edges)

        rows = rearrange(nodes, 'b i d -> b i 1 d')
        cols = rearrange(nodes, 'b j d -> b 1 j d')

        rows, cols = torch.broadcast_tensors(rows, cols)

        concat_feats = torch.cat((rows, cols, edges), dim = -1)
        out = self.proj(concat_feats)
        return out

# main class

class RelationalTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_edges,
        dim_head = 64,
        heads = 8,
        causal = False,
        linear_dim_head = 16,
        linear_heads = 16,
        ff_mult = 4,
        dropout = 0.,
        max_seq_len = 2048,
        ablate_edges = False,
        edges_diagonal_attn = True,
        num_residual_streams = 4
    ):
        super().__init__()
        dim_edges = default(dim_edges, dim)
        self.causal = causal
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.to_edge_emb = EdgeEmbed(dim, dim_edges)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                ModuleList([
                    RelationalAttention(dim = dim, dim_edges = dim_edges, dim_head = dim_head, heads = heads, dropout = dropout, causal = causal),
                    FeedForward(dim = dim, mult = ff_mult, dropout = dropout)
                ]),
                ModuleList([
                    NodesToEdges(dim_nodes = dim, dim_edges = dim_edges),
                    FeedForward(dim = dim_edges, mult = ff_mult)
                ])
            ]))

        self.final_norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        mask = None
    ):
        batch, seq_len, device = *x.shape, x.device
        assert seq_len <= self.max_seq_len

        x = self.token_emb(x)

        x = x + self.pos_emb(torch.arange(seq_len, device = device))

        edges = self.to_edge_emb(x)

        for (attn, ff), (nodes_to_edges, edges_ff) in self.layers:

            x = attn(x, mask = mask, edges = edges) + x

            x = ff(x) + x

            edges = nodes_to_edges(edges, nodes = x) + edges

            edges = edges_ff(edges) + edges

        return self.to_logits(x)

# quick test

if __name__ == '__main__':

    transformer = RelationalTransformer(
        num_tokens = 256,
        dim = 128,
        dim_edges = 16,
        depth = 2
    )

    x = torch.randint(0, 256, (2, 128))
    logits = transformer(x)

    assert logits.shape == (2, 128, 256)
