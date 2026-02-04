# trainer.py
import math
import logging
import torch
from utils.steps import train_one_epoch, validate
from utils.checkpoint import atomic_save
from utils.sanity import check_dataloader
from utils.train_factory import schedule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(model, dataloader, criterion, optimizer, device, epochs=10,
                val_dataloader=None, scheduler=None, grad_clip=None,
                save_path="best_model.pth", patience=5, wandb_run=None):
    check_dataloader(dataloader, "train")
    if val_dataloader: check_dataloader(val_dataloader, "val")
    if not any(p.requires_grad for p in model.parameters()): raise ValueError("Model has no trainable parameters")
    model.to(device)
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} start")
        try:
            train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=grad_clip, wandb_run=wandb_run, epoch=epoch)
            atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                             "scheduler":scheduler.state_dict() if scheduler else None},save_path)
        except Exception:
            logger.exception("Crash during training epoch")
            atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},save_path+".crash")
            raise 
        logger.info(f"Epoch {epoch+1} train | loss={train_loss:.4f} acc={train_acc:.4f}")
        val_loss = None
        if val_dataloader: val_loss, val_acc = validate(model, val_dataloader, criterion, device, val_loss, wandb_run, epoch, scheduler, val_acc, optimizer, save_path, best_val_loss=float('inf'), patience=patience, patience_counter=0)
        if scheduler: schedule(scheduler, val_dataloader, val_loss)
    return model