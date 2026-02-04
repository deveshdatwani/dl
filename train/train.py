import argparse
import yaml
import torch
import wandb
import logging
from utils.trainer import train_model
from utils.train_factory import get_model, get_loss_fn, get_optimizer, get_dataloaders, load_and_override_config, quantize_model
from optim.post_training import quantize_dynamic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Train DL models modularly")
    parser.add_argument('--config_yaml', type=str, default='train/config.yaml', help='Path to config YAML')
    parser.add_argument('--model-source', type=str, help='Model source: custom, cloned, huggingface')
    parser.add_argument('--model-name', type=str, help='Model name or huggingface id')
    parser.add_argument('--loss-fn', type=str, help='Loss type')
    parser.add_argument('--wandb-project', type=str, help='wandb project name')
    parser.add_argument('--wandb-plot-name', type=str, help='wandb plot name')
    parser.add_argument('--dataset-name', type=str, help='Dataset name')
    parser.add_argument('--dataset-path', type=str, help='Dataset path')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--train-split', type=float, help='Train/test split')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--save-path', type=str, help='Save path')
    parser.add_argument('--device', type=str, help='Device')
    parser.add_argument('--quantize', type=bool, help='Quantize model')
    parser.add_argument('--quantized-save-path', type=str, help='Quantized save path')
    parser.add_argument('--optimizer', type=str, help='Optimizer type (adam, sgd, etc.)')
    parser.add_argument('--optimizer-params', type=str, help='Optimizer params as YAML/JSON string (optional)')
    args = parser.parse_args()
    config = load_and_override_config(args)
    logger.info("===== Training Configuration =====\n%s", yaml.dump(config, sort_keys=False, default_flow_style=False))
    resp = input("Proceed with these settings? (y/n): ").strip().lower()
    if resp != 'y':
        logger.info("Aborted by user.")
        exit(0)
    wandb_run = wandb.init(project=config['wandb_project'], name=config['wandb_plot_name'], config=config) if 'wandb_project' in config and config['wandb_plot_name'] else None
    model = get_model(config['model_source'])
    loss_fn = get_loss_fn(typ=config['loss_fn'], optimizer_params=config.get('loss_params', {}))
    train_dataloader, test_dataloader = get_dataloaders(config['dataset_path'], config['dataset_name'], config['train_split'], config['batch_size'])
    optimizer = get_optimizer(config['optimizer'], model.parameters(), config['lr'], config['optimizer_params'])
    device = config.get('device', 'cpu')
    train_model(model=model, dataloader=train_dataloader, criterion=loss_fn, optimizer=optimizer, epochs=config.get('epochs', 10), save_path=config.get('save_path', "./checkpoint/model.pth"), device=device, wandb_run=wandb_run)
    if config.get('quantize', False): quantize_model(model)

if __name__ == "__main__":
    main()