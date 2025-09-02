import torch 
from torch import nn 


class feed_forward(nn.Module):
    def __init__(self, in_dim, inner_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, inner_dim)
        self.l2 = nn.Linear(inner_dim, in_dim)
        self.layer_norm_1 = nn.LayerNorm(in_dim)
        self.layer_norm_2 = nn.LayerNorm(inner_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.layer_norm_2(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.layer_norm_1(x)
        return x
    

class attention(nn.Module):
    def __init__(self, in_dim, inner_dim, heads=8, head_dim=64,):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_qkv = nn.Linear(in_dim, in_dim*3)
        self.attend = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        qk = torch.matmul(q, k.transpose(-2, -1)) * q.shape[-1] ** -0.5
        qk = self.attend(qk)
        x = torch.matmul(qk, v)
        return x
        

class transformer(nn.Module):
    def __init__(self, in_dim=128, inner_dim=64, heads=8, head_dim=64):
        super().__init__()
        self.attention_block = attention(in_dim=in_dim, inner_dim=inner_dim, heads=heads, head_dim=head_dim)
        self.ff = feed_forward(in_dim=in_dim, inner_dim=inner_dim)
    
    def forward(self, x):
        x1 = x
        x = self.attention_block(x)
        x += x1
        x2 = x
        x = self.ff(x)
        x += x2
        return x


class ViT(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth
        self.transformer_block = transformer(in_dim=128, inner_dim=64, heads=8, head_dim=64)
        self._model = nn.Sequential(
                    *[self.transformer_block for _ in range(depth)]
        )

    def forward(self, x):
        return self._model(x)


model = ViT(20)
x = torch.rand(4, 64, 128)
print(model(x))