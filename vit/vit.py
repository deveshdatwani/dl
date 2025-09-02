import torch 
from torch import nn 
import einops


class feed_forward(nn.Module):
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
    

class attention(nn.Module):
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
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, self.heads * self.head_dim)
        return self.proj(out)
        

class transformer(nn.Module):
    def __init__(self, in_dim=128, inner_dim=64, heads=8, head_dim=64):
        super().__init__()
        self.attend = attention(in_dim, heads, head_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.ff = feed_forward(in_dim, inner_dim)
    
    def forward(self, x):
        x = self.norm1(self.attend(x) + x)
        x = self.ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, depth=2, in_dim=784, inner_dim=64, heads=8, head_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([transformer(in_dim, inner_dim, heads, head_dim) for _ in range(depth)])
        self.final_layer = nn.Linear(48, 10)
    
    def forward(self, x):
        patch_size = 8
        P1 = P2 = x.shape[-1] // patch_size
        x = einops.rearrange(x, 'B C (H P1) (W P2) -> B (H W) (C P1 P2)', P1=P1, P2=P2)
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x).squeeze(1)