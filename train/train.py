"""
Modular training entry point for DL models
Usage:
    python train.py --model vit --dataset cifar10 --config config.yaml
"""
import argparse
import yaml
import torch
from models import vit
from utils.trainer import train_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import torch.optim as optim
from optim.post_training import quantize_dynamic

# Weights & Biases
import wandb

MODEL_REGISTRY = {
    "vit": vit,
    # Add more models here
}

def main():
    parser = argparse.ArgumentParser(description="Train DL models modularly")
    parser.add_argument('--model', type=str, default='vit', help='Model name')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(project=config.get('wandb_project', 'dl-training'), config=config)

    # Model selection
    model_cls = MODEL_REGISTRY.get(args.model)
    if not model_cls:
        raise ValueError(f"Model {args.model} not found")
    model = model_cls(**config.get('model', {}))

    # Dataset selection
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    train_dataloader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32))
    test_dataloader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32))

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    criterion = nn.CrossEntropyLoss()
    device = config.get('device', 'cpu')

    # Pass wandb to train_model if enabled
    train_model(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config.get('epochs', 10),
        save_path=config.get('save_path', "./checkpoint/model.pth"),
        device=device,
        wandb_run=wandb if args.wandb else None
    )

    # Post-training quantization (optional)
    if config.get('quantize', False):
        print("Applying dynamic quantization...")
        quantized_model = quantize_dynamic(model)
        torch.save(quantized_model.state_dict(), config.get('quantized_save_path', './checkpoint/model_quantized.pth'))
        print(f"Quantized model saved to {config.get('quantized_save_path', './checkpoint/model_quantized.pth')}")

if __name__ == "__main__":
    main()
