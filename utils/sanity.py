import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_tensor(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError(f"{name} contains NaN/Inf")

def check_dataloader(dl, name):
    if len(dl) == 0:
        raise ValueError(f"{name} dataloader is empty")
