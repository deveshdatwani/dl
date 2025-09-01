import torch


def train_model(model, dataloader, criterion, optimizer, device, epochs=10, val_dataloader=None):
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
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        if val_dataloader:
            validate_model(model, val_dataloader, criterion, device)


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
    val_loss /= total
    val_acc = val_corrects / total
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")