import torch, os, logging, math, tempfile, argparse, yaml, importlib
import wandb


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_config(yaml_path, cli_args=None):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    if cli_args:
        # Override YAML with CLI args if provided
        for k, v in vars(cli_args).items():
            if v is not None:
                # Support nested keys (e.g., model.name)
                keys = k.split('__')
                d = config
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = v
    return config

def prompt_user_config(config):
    print("\n===== Training Configuration =====")
    print(yaml.dump(config, sort_keys=False, default_flow_style=False))
    resp = input("Proceed with these settings? (y/n): ").strip().lower()
    if resp != 'y':
        print("Aborted by user.")
        exit(0)

def _check_tensor(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError(f"{name} contains NaN/Inf")

def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)

def train_model(
    model, dataloader, criterion, optimizer, device,
    epochs=10, val_dataloader=None, scheduler=None,
    grad_clip=None, save_path="best_model.pth", patience=5,
    wandb_run=None
):
    if len(dataloader) == 0:
        raise ValueError("Train dataloader is empty")
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Model has no trainable parameters")
    model.to(device)
    best_val_loss = math.inf
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (inputs, labels) in enumerate(dataloader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                assert labels.dtype == torch.long
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                if isinstance(outputs, (tuple, dict)):
                    raise TypeError("Model output must be a tensor")
                loss = criterion(outputs, labels)
                _check_tensor(loss, "loss")
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            except Exception:
                logger.exception(f"Train failure at epoch {epoch}, batch {i}")
                _atomic_save(
                    {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()},
                    save_path + ".crash"
                )
                raise
        if total == 0:
            raise ValueError("No samples processed in training epoch")
        train_loss = running_loss / total
        train_acc = correct / total
        if wandb_run:
            wandb_run.log({"epoch": epoch+1, "train/loss": train_loss, "train/acc": train_acc})
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss {train_loss:.4f} Acc {train_acc:.4f}")
        val_loss = None
        if val_dataloader:
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            logger.info(f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
            if wandb_run:
                wandb_run.log({"epoch": epoch+1, "val/loss": val_loss, "val/acc": val_acc})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                _atomic_save(
                    {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict() if scheduler else None},
                    save_path
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping")
                    return
        def get_loss_fn(loss_cfg):
            typ = loss_cfg.get('type', 'cross_entropy')
            if typ == 'cross_entropy':
                return torch.nn.CrossEntropyLoss(**loss_cfg.get('params', {}))
            elif typ == 'focal':
                # Example: expects FocalLoss in utils.loss
                from dl.utils.loss import FocalLoss
                return FocalLoss(**loss_cfg.get('params', {}))
            else:
                raise ValueError(f"Unknown loss type: {typ}")

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

        def main():
            parser = argparse.ArgumentParser(description="Trainer CLI")
            parser.add_argument('--config', type=str, default='dl/utils/trainer_config.yaml', help='Path to config YAML')
            parser.add_argument('--model__source', type=str, help='Model source: custom, cloned, huggingface')
            parser.add_argument('--model__name', type=str, help='Model name')
            parser.add_argument('--model__path', type=str, help='Model path')
            parser.add_argument('--dataset__name', type=str, help='Dataset name')
            parser.add_argument('--dataset__path', type=str, help='Dataset path')
            parser.add_argument('--loss__type', type=str, help='Loss type')
            parser.add_argument('--wandb__project', type=str, help='wandb project name')
            parser.add_argument('--wandb__plot_name', type=str, help='wandb plot name')
            parser.add_argument('--run__mode', type=str, help='Run mode: train, validate, deploy')
            parser.add_argument('--run__epochs', type=int, help='Epochs')
            parser.add_argument('--run__batch_size', type=int, help='Batch size')
            parser.add_argument('--run__learning_rate', type=float, help='Learning rate')
            parser.add_argument('--run__seed', type=int, help='Random seed')
            args = parser.parse_args()

            config = load_config(args.config, args)
            prompt_user_config(config)

            # Set random seed
            if 'seed' in config.get('run', {}):
                torch.manual_seed(config['run']['seed'])

            # Setup wandb
            wandb_run = wandb.init(
                project=config['wandb']['project'],
                name=config['wandb']['plot_name'],
                config=config
            )

            # Load model and loss
            model = get_model(config['model'])
            loss_fn = get_loss_fn(config['loss'])
            # ...existing code to load data, optimizer, etc...
            # Example: train_model(model, train_loader, loss_fn, optimizer, device, ...)

        if __name__ == "__main__":
            main()
        if scheduler:
            if "ReduceLROnPlateau" in scheduler.__class__.__name__:
                if val_loss is None:
                    raise ValueError("ReduceLROnPlateau requires val_loss")
                scheduler.step(val_loss)
            else:
                scheduler.step()

def validate_model(model, dataloader, criterion, device):
    if len(dataloader) == 0:
        raise ValueError("Val dataloader is empty")
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _check_tensor(loss, "val_loss")

            loss_sum += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("No samples processed in validation")
    return loss_sum / total, correct / total