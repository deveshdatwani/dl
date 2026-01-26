import torch
from torch import nn

class TemplateModel(nn.Module):
    """Example template for new models."""
    def __init__(self, input_dim=128, output_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)
