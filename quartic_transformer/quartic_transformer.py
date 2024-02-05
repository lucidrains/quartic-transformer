import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange

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
