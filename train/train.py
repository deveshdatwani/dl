"""
Modular training entry point for DL models
Usage:
    python train.py --model vit --dataset cifar10 --config config.yaml
"""

import argparse
import yaml
import torch
import importlib
from utils.trainer import train_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from optim.post_training import quantize_dynamic
import wandb

def get_model(model_cfg):
    src = model_cfg.get('source', 'custom')
    name = model_cfg.get('name')
    if src == 'custom':
        mod = importlib.import_module(f'dl.models.custom.{name}')
        return getattr(mod, name)()
    elif src == 'cloned':
        mod = importlib.import_module(f'dl.models.cloned.{name}')
        return getattr(mod, name)()
    elif src == 'huggingface':
        from transformers import AutoModel
        return AutoModel.from_pretrained(name)
    else:
        raise ValueError(f"Unknown model source: {src}")

def get_loss_fn(loss_cfg):
    typ = loss_cfg.get('type', 'cross_entropy')
    if typ == 'cross_entropy':
        from torch import nn
        return nn.CrossEntropyLoss(**loss_cfg.get('params', {}))
    elif typ == 'focal':
        from utils.loss import FocalLoss
        return FocalLoss(**loss_cfg.get('params', {}))
    else:
        raise ValueError(f"Unknown loss type: {typ}")

def main():
    parser = argparse.ArgumentParser(description="Train DL models modularly")
    parser.add_argument('--config', type=str, default='dl/train/config.yaml', help='Path to config YAML')
    parser.add_argument('--model__source', type=str, help='Model source: custom, cloned, huggingface')
    parser.add_argument('--model__name', type=str, help='Model name or huggingface id')
    parser.add_argument('--loss__type', type=str, help='Loss type')
    parser.add_argument('--wandb__project', type=str, help='wandb project name')
    parser.add_argument('--wandb__plot_name', type=str, help='wandb plot name')
    parser.add_argument('--dataset__name', type=str, help='Dataset name')
    parser.add_argument('--dataset__path', type=str, help='Dataset path')
    parser.add_argument('--run__batch_size', type=int, help='Batch size')
    parser.add_argument('--run__lr', type=float, help='Learning rate')
    parser.add_argument('--run__epochs', type=int, help='Epochs')
    parser.add_argument('--run__save_path', type=str, help='Save path')
    parser.add_argument('--run__device', type=str, help='Device')
    parser.add_argument('--run__quantize', type=bool, help='Quantize model')
    parser.add_argument('--run__quantized_save_path', type=str, help='Quantized save path')
    args = parser.parse_args()

    # Load config and override with CLI args
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            keys = k.split('__')
            d = config
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v

    # Print config and confirm
    print("\n===== Training Configuration =====")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    resp = input("Proceed with these settings? (y/n): ").strip().lower()
    if resp != 'y':
        print("Aborted by user.")
        exit(0)

    # Setup wandb
    wandb_run = wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['plot_name'],
        config=config
    ) if 'wandb' in config and config['wandb'].get('project') else None

    # Model and loss
    model = get_model(config['model'])
    loss_fn = get_loss_fn(config['loss'])

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    ds_cfg = config.get('dataset', {})
    if ds_cfg.get('name', 'cifar10') == 'cifar10':
        train_dataset = datasets.CIFAR10(root=ds_cfg.get('path', "./data"), train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=ds_cfg.get('path', "./data"), train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {ds_cfg.get('name')} not supported")
    run_cfg = config.get('run', {})
    train_dataloader = DataLoader(train_dataset, batch_size=run_cfg.get('batch_size', 32))
    test_dataloader = DataLoader(test_dataset, batch_size=run_cfg.get('batch_size', 32))

    optimizer = optim.Adam(model.parameters(), lr=run_cfg.get('lr', 1e-4))
    device = run_cfg.get('device', 'cpu')

    # Train
    train_model(
        model=model,
        dataloader=train_dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        epochs=run_cfg.get('epochs', 10),
        save_path=run_cfg.get('save_path', "./checkpoint/model.pth"),
        device=device,
        wandb_run=wandb_run
    )

    # Post-training quantization (optional)
    if run_cfg.get('quantize', False):
        print("Applying dynamic quantization...")
        quantized_model = quantize_dynamic(model)
        torch.save(quantized_model.state_dict(), run_cfg.get('quantized_save_path', './checkpoint/model_quantized.pth'))
        print(f"Quantized model saved to {run_cfg.get('quantized_save_path', './checkpoint/model_quantized.pth')}")

if __name__ == "__main__":
    main()

    # Load config
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(project=config.get('wandb_project', 'dl-training'), config=config)


    # Model import and instantiation
    try:
        model_cls = get_model_class(args.model)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model class from '{args.model}': {e}")
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
