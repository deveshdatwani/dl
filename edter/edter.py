from vit.vit import transformer
from torch import nn


class edter(nn.Module):
    def __init__(self, depth=24, in_dim=128):
        self.depth = depth
        self.transformer_block = transformer(in_dim=in_dim)
        self.positional_embedding = nn.Parameter()
    
    def forward(self,):
        return None