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
    

ff = nn.Sequential(*[feed_forward(128, 256, 128) for _ in range(20)])
x = torch.rand((2, 128))
x = ff(x)

print(ff.eval)
