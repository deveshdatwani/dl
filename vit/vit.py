import torch 
from torch import nn 


class feed_forward(nn.Module):
    def __init__(self, in_dim, inner_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, inner_dim)
        self.l2 = nn.Linear(inner_dim, out_dim)
        self.batch_norm_1 = nn.BatchNorm1d(in_dim)
        self.batch_norm_2 = nn.BatchNorm1d(inner_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.batch_norm_2(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.batch_norm_1(x)
        return x
    

class attention(nn.Module):
    def __init__(self, input_dim, inner_dim, heads=8, head_dim=64,):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_qkv = nn.Linear(input_dim, input_dim*3)
        self.attend = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        qk = torch.matmul(q, k.transpose(-2, -1)) * q.shape[-1] ** -0.5
        qk = self.attend(qk)
        x = torch.matmul(qk, v)
        return x
        

class transformer(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.tf = nn.Sequential(
                attention(128, 64, 128), 
                feed_forward(128, 256, 128)
                )

    def forward(self, x):
        xf = self.tf(x)
        xf += x
        return xf
    

model = transformer(8)
x = torch.rand(4, 128)
print(model(x).shape)