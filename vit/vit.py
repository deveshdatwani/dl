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
    def __init__(self, heads=8, head_dim=64, inner_dim=256):
        super().__init__()
        self.heads = heads
        self.inner_dim = inner_dim
        self.head_dim = head_dim
        self.linear = nn.Linear(inner_dim, heads * head_dim)
        self.to_qkv = nn.Linear(head_dim, head_dim * 3)
        self.sm = nn.Softmax(-1)
        self.linear_back = nn.Linear(self.heads * self.head_dim, self.inner_dim)
        self.ln = nn.LayerNorm(inner_dim)
        
    def forward(self, x):
        residual = x
        x = self.linear(x).chunk(self.heads, -1)
        xx = []
        for head in x:
            q, k, v = self.to_qkv(head).chunk(3, -1)
            attention = self.sm(torch.matmul(q, k.transpose(-2, -1)) * 1 / sqrt(q.shape[-1]))
            kv = torch.matmul(attention, v)
            xx.append(kv)
        x = torch.cat(xx, dim=-1)
        return self.ln(self.linear_back(x) + residual)
        

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


class vit(nn.Module):
    def __init__(self, depth=2, in_dim=192, inner_dim=64, heads=8, head_dim=64, num_classes=10, img_size=32, patch_size=8, in_channels=3):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=False) 
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, in_dim))  
        self.layers = nn.ModuleList([transformer(in_dim, inner_dim, heads, head_dim) for _ in range(depth)])
        self.final_layer = nn.Linear(in_dim, num_classes)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.in_dim = in_dim

    def forward(self, x):
        P = self.patch_size
        x = einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=P, p2=P)
        B = x.shape[0]
        self.cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat([self.cls_tokens, x], dim=1)           
        # x = x + self.pos_embed[:, :x.size(1), :] 
        for layer in self.layers:
            x = layer(x)
        cls_out = x[:, 0, :]  
        return self.final_layer(cls_out)
