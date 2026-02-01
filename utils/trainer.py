# trainer.py
import math
import logging
import torch
from utils.steps import train_one_epoch, validate
from utils.checkpoint import atomic_save
from utils.sanity import check_dataloader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_model(model,dataloader,criterion,optimizer,device,epochs=10,val_dataloader=None,scheduler=None,grad_clip=None,save_path="best_model.pth",patience=5,wandb_run=None):
    check_dataloader(dataloader,"train")
    if val_dataloader:
        check_dataloader(val_dataloader,"val")
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Model has no trainable parameters")
    model.to(device)
    best_val_loss=math.inf
    patience_counter=0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} start")
        try:
            train_loss,train_acc=train_one_epoch(model,dataloader,criterion,optimizer,device,grad_clip=grad_clip)
        except Exception:
            logger.exception("Crash during training epoch")
            atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict()},save_path+".crash")
            raise
        logger.info(f"Epoch {epoch+1} train | loss={train_loss:.4f} acc={train_acc:.4f}")
        if wandb_run:
            wandb_run.log({"epoch":epoch+1,"train/loss":train_loss,"train/acc":train_acc})
        val_loss=None
        if val_dataloader:
            val_loss,val_acc=validate(model,val_dataloader,criterion,device)
            logger.info(f"Epoch {epoch+1} val   | loss={val_loss:.4f} acc={val_acc:.4f}")
            if wandb_run:
                wandb_run.log({"epoch":epoch+1,"val/loss":val_loss,"val/acc":val_acc})
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
                atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None},save_path)
            else:
                patience_counter+=1
                if patience_counter>=patience:
                    logger.info("Early stopping triggered")
                    return
        if scheduler:
            if val_dataloader and "ReduceLROnPlateau" in scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()