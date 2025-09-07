from vit.vit import vit as vit_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.trainer import train_model
from torch import nn as nn
import torch.optim as optim
from torchvision import datasets, transforms    


model = vit_model(depth=10, in_dim=192, inner_dim=128)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)
train_model(model=model, dataloader=train_dataloader, criterion=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-4), 
            epochs=10, save_path="./checkpoint/cifar_transformer.pth", device='cpu')