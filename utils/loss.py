from torch import nn 
import torch


class edge_loss(nn.Module):
    def __init__(self, alpha=1e-5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, gt):
        # self.alpha = "something"
        loss = -1 * torch.sum(
                        ( pred * self.alpha * torch.log(pred) ) + 
                        ( (1 - gt) * (1 - self.alpha) * torch.log(1 - pred) )
                        )
        return loss
    

el = edge_loss()
x = torch.rand((3, 128, 128))
y = torch.rand((3, 128, 128))
print(el(x, y))