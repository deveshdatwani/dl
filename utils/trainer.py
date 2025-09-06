import torch
import os
import logging 


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def train_model(model, dataloader, criterion, optimizer, device, epochs=10, 
                val_dataloader=None, scheduler=None, grad_clip=None, save_path="best_model.pth", 
                patience=5):
    best_val_loss = float("inf")
    patience_counter = 0
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        torch.save(model.state_dict(), save_path)
        logger.info("Saved model")
        if scheduler:
            scheduler.step()
        if val_dataloader:
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                logger.info("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    return


def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_corrects, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, val_corrects / total