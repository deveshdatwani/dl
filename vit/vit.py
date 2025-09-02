import torch 
from torch import nn 


class FeedForward(nn.Module):
    def __init__(self, in_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)
    
    def forward(self, x):
        return self.norm(self.net(x) + x)
    

class Attention(nn.Module):
    def __init__(self, in_dim, heads=8, head_dim=64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(in_dim, heads * head_dim * 3)
        self.proj = nn.Linear(heads * head_dim, in_dim)
    
    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, self.heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.heads * self.head_dim)
        return self.proj(out)
        

class Transformer(nn.Module):
    def __init__(self, in_dim=128, inner_dim=64, heads=8, head_dim=64):
        super().__init__()
        self.attn = Attention(in_dim, heads, head_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ff = FeedForward(in_dim, inner_dim)
    
    def forward(self, x):
        x = self.norm1(self.attn(x) + x)
        x = self.ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, depth=2, in_dim=128, inner_dim=64, heads=8, head_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([Transformer(in_dim, inner_dim, heads, head_dim) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

model = ViT(20)
x = torch.rand(4, 64, 128)
print(model(x))