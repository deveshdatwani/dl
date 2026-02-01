import torch
from utils.sanity import check_tensor
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=None, logger=None):
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
        logger.info(f"Batch {i+1}/{len(dataloader)} summary: loss={loss.item():.4f}, acc={(outputs.argmax(1)==labels).float().mean():.4f}")
    avg_loss = loss_sum / total
    avg_acc = correct / total
    logger.info(f"Epoch complete: Train Loss {avg_loss:.4f} | Train Acc {avg_acc:.4f}")
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
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
    return loss_sum / total, correct / total