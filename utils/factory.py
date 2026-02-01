"""
Factory functions for model, loss, optimizer, and dataset creation
"""
import importlib
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.loss import edge_loss
from optim.post_training import quantize_dynamic


def get_model(model_cfg):
    from models.cloned.cifar.models.vgg import VGG
    return VGG()
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
    elif src == 'vgg':
        from models.cloned.cifar.models.vgg import VGG
        return VGG
    else:
        raise ValueError(f"Unknown model source: {src}")

def get_loss_fn(loss_cfg):
    typ = loss_cfg.get('type', 'cross_entropy')
    if typ == 'cross_entropy':
        return nn.CrossEntropyLoss(**loss_cfg.get('params', {}))
    elif typ == 'focal':
        from utils.loss import FocalLoss
        return FocalLoss(**loss_cfg.get('params', {}))
    elif typ == 'edge':
        return edge_loss(**loss_cfg.get('params', {}))
    else:
        raise ValueError(f"Unknown loss type: {typ}")

def get_optimizer(opt_cfg, model_params, lr):
    opt_type = opt_cfg.get('type', 'adam').lower()
    opt_params = opt_cfg.get('params', {})
    if opt_type == 'adam':
        return torch.optim.Adam(model_params, lr=lr, **opt_params)
    elif opt_type == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, **opt_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_dataloaders(ds_cfg, run_cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    if ds_cfg.get('name', 'cifar10') == 'cifar10':
        train_dataset = datasets.CIFAR10(root=ds_cfg.get('path', "./data"), train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=ds_cfg.get('path', "./data"), train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {ds_cfg.get('name')} not supported")
    train_dataloader = DataLoader(train_dataset, batch_size=run_cfg.get('batch_size', 32))
    test_dataloader = DataLoader(test_dataset, batch_size=run_cfg.get('batch_size', 32))
    return train_dataloader, test_dataloader

def load_and_override_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    def get_nested(cfg, keys):
        for k in keys:
            if not isinstance(cfg, dict) or k not in cfg:
                return None
            cfg = cfg[k]
        return cfg
    def set_nested(cfg, keys, value):
        for k in keys[:-1]:
            cfg = cfg.setdefault(k, {})
        cfg[keys[-1]] = value
    for arg, value in vars(args).items():
        if value is None or arg == 'config':
            continue
        keys = arg.split('__')
        if get_nested(config, keys) is not None:
            set_nested(config, keys, value)
    return config
