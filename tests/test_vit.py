import torch
from models.vit import vit

def test_vit_forward():
    model = vit(depth=2, in_dim=192, inner_dim=64, num_classes=10, img_size=32, patch_size=8, in_channels=3)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)
