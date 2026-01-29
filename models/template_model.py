import torch.nn as nn

class TemplateModel(nn.Module):
    """
    Example model template for use with the modular trainer.
    Replace this with your own architecture.
    """
    def __init__(self, input_dim=3*32*32, num_classes=10, **kwargs):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
import torch
from torch import nn

class TemplateModel(nn.Module):
    """Example template for new models."""
    def __init__(self, input_dim=128, output_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)
