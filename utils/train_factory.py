"""
Factory functions for model, loss, optimizer, and dataset creation
"""
import importlib
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from optim.post_training import quantize_dynamic
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def get_loss_fn(typ='cross_entropy', optimizer_params={}):
    if typ == 'cross_entropy':
        return nn.CrossEntropyLoss(**optimizer_params)

def get_optimizer(opt, model_params, lr, opt_params={}):
    if opt == 'adam':
        return torch.optim.Adam(model_params, lr=lr, **opt_params)
    elif opt == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, **opt_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_dataloaders(data_set_path, dataset_name, train_split, batch_size):
    transform = get_transform()
    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_set_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_set_path, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader

def load_and_override_config(args):
    with open(args.config_yaml) as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if args.__dict__[key]: 
            config[key] = args.__dict__[key]
    return config

def get_transform():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    return tf

def quantize_model(model, config):
    logger.info("Applying dynamic quantization...")
    quantized_model = quantize_dynamic(model)
    torch.save(quantized_model.state_dict(), config.get('quantized_save_path', './checkpoint/model_quantized.pth'))
    logger.info("Quantized model saved to %s", config.get('quantized_save_path', './checkpoint/model_quantized.pth'))