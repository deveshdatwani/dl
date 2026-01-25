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
    grad_clip=None, save_path="best_model.pth", patience=5
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
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss {train_loss:.4f} Acc {train_acc:.4f}")
        val_loss = None
        if val_dataloader:
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            logger.info(f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
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