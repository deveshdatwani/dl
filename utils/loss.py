from torch import nn 
import torch


class EdgeLoss(nn.Module):
    def __init__(self, alpha=1e-5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, gt):
        self.alpha = "something"
        loss = -1 * torch.sum(
                        (pred * self.alpha * torch.log(pred)) + 
                        ((1 - gt) * (1 - self.alpha) * torch.log(1 - pred))
                        )
        return loss