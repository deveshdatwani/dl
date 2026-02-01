
import torch, os, logging, math, tempfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for i, (inputs, labels) in enumerate(dataloader):
            try:
                logger.debug(f"Epoch {epoch+1} Batch {i+1}: Loading data and moving to device {device}")
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
                    logger.debug(f"Applying gradient clipping: {grad_clip}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                if (i+1) % 10 == 0 or (i+1) == len(dataloader):
                    logger.info(f"Epoch {epoch+1} Batch {i+1}/{len(dataloader)}: Loss {loss.item():.4f}")
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
        logger.info(f"Epoch {epoch+1} summary: Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f}")
        if wandb_run:
            wandb_run.log({"epoch": epoch+1, "train/loss": train_loss, "train/acc": train_acc})
        val_loss = None
        if val_dataloader:
            logger.info(f"Running validation for epoch {epoch+1}")
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            logger.info(f"Epoch {epoch+1} summary: Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")
            if wandb_run:
                wandb_run.log({"epoch": epoch+1, "val/loss": val_loss, "val/acc": val_acc})
            if val_loss < best_val_loss:
                logger.info(f"New best validation loss: {val_loss:.4f} (prev: {best_val_loss:.4f}) - saving model.")
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
                logger.info(f"No improvement in val loss. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info("Early stopping")
                    return
        if scheduler:
            logger.info(f"Stepping scheduler at epoch {epoch+1}")
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
    logger.info("Starting validation loop")
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            logger.debug(f"Validation batch {i+1}/{len(dataloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _check_tensor(loss, "val_loss")

            loss_sum += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ValueError("No samples processed in validation")
    avg_loss = loss_sum / total
    avg_acc = correct / total
    logger.info(f"Validation summary: Loss {avg_loss:.4f} | Acc {avg_acc:.4f}")
    return avg_loss, avg_acc