"""
Modular training entry point for DL models
Usage:
    python train.py --model vit --dataset cifar10 --config config.yaml
"""


import argparse
import yaml
import torch
import wandb
from utils.trainer import train_model
from utils.factory import get_model, get_loss_fn, get_optimizer, get_dataloaders
from optim.post_training import quantize_dynamic


def main():
    parser = argparse.ArgumentParser(description="Train DL models modularly")
    parser.add_argument('--config', type=str, default='train/config.yaml', help='Path to config YAML')
    parser.add_argument('--model-source', type=str, help='Model source: custom, cloned, huggingface')
    parser.add_argument('--model-name', type=str, help='Model name or huggingface id')
    parser.add_argument('--loss-type', type=str, help='Loss type')
    parser.add_argument('--wandb-project', type=str, help='wandb project name')
    parser.add_argument('--wandb-plot-name', type=str, help='wandb plot name')
    parser.add_argument('--dataset-name', type=str, help='Dataset name')
    parser.add_argument('--dataset-path', type=str, help='Dataset path')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--save-path', type=str, help='Save path')
    parser.add_argument('--device', type=str, help='Device')
    parser.add_argument('--quantize', type=bool, help='Quantize model')
    parser.add_argument('--quantized-save-path', type=str, help='Quantized save path')
    parser.add_argument('--optimizer-type', type=str, help='Optimizer type (adam, sgd, etc.)')
    parser.add_argument('--optimizer-params', type=str, help='Optimizer params as YAML/JSON string (optional)')
    args = parser.parse_args()

    # Load config and override with CLI args
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            run_keys = ['batch_size', 'lr', 'epochs', 'save_path', 'device', 'quantize', 'quantized_save_path']
            if k in run_keys:
                config.setdefault('run', {})[k] = v
            else:
                keys = k.split('__')
                d = config
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = v

    print("\n===== Training Configuration =====")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    resp = input("Proceed with these settings? (y/n): ").strip().lower()
    if resp != 'y':
        print("Aborted by user.")
        exit(0)

    wandb_run = wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['plot_name'],
        config=config
    ) if 'wandb' in config and config['wandb'].get('project') else None

    model = get_model(config['model'])
    loss_fn = get_loss_fn(config['loss'])
    run_cfg = config.get('run', {})
    ds_cfg = config.get('dataset', {})
    train_dataloader, test_dataloader = get_dataloaders(ds_cfg, run_cfg)
    optimizer = get_optimizer(config.get('optimizer', {}), model.parameters(), run_cfg.get('lr', 1e-4))
    device = run_cfg.get('device', 'cpu')

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

    if run_cfg.get('quantize', False):
        print("Applying dynamic quantization...")
        quantized_model = quantize_dynamic(model)
        torch.save(quantized_model.state_dict(), run_cfg.get('quantized_save_path', './checkpoint/model_quantized.pth'))
        print(f"Quantized model saved to {run_cfg.get('quantized_save_path', './checkpoint/model_quantized.pth')}")


if __name__ == "__main__":
    main()
