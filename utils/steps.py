import torch
from utils.sanity import check_tensor
import logging
from utils.checkpoint import atomic_save
import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None, logger=None, wandb_run=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        check_tensor(loss, "loss")
        logger.debug(f"Batch {i+1}: Loss value {loss.item():.4f}")
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        logger.debug(f"Batch {i+1}: Optimizer step")
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        if wandb_run: wandb_run.log({"epoch":epoch+1,"train/loss":train_loss,"train/acc":train_acc})
        logger.info(f"Batch {i+1}/{len(dataloader)} summary: loss={loss.item():.4f}, acc={(outputs.argmax(1)==labels).float().mean():.4f}")
    avg_loss = loss_sum / total
    avg_acc = correct / total
    logger.info(f"Epoch complete: Train Loss {avg_loss:.4f} | Train Acc {avg_acc:.4f}")
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, val_loss, wandb_run, epoch, scheduler, val_acc, optimizer, save_path, best_val_loss, patience, patience_counter):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        check_tensor(loss, "val_loss")
        loss_sum += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    logger.info(f"Epoch {epoch+1} val   | loss={val_loss:.4f} acc={val_acc:.4f}")
    if wandb_run:
        wandb_run.log({"epoch":epoch+1,"val/loss":val_loss,"val/acc":val_acc})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        atomic_save({"model":model.state_dict(), "optimizer":optimizer.state_dict(),
                        "scheduler":scheduler.state_dict() if scheduler else None}, save_path)
    else:
        patience_counter+=1
        if patience_counter>=patience:
            logger.info("Early stopping triggered")
            return
    return loss_sum / total, correct / total