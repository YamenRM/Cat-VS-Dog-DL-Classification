from torchvision import transforms, datasets 
from torch.utils.data import DataLoader , random_split
import torch


# data loading utilities 
def get_dataloaders(train_dataset, test_dataset, batch_size=32):
    train_Transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    test_Transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    full_train_data= datasets.ImageFolder(train_dataset, transform=train_Transforms)
    test_data = datasets.ImageFolder(test_dataset, transform=test_Transforms)

# split the training data into training and validation sets
    train_size = int(0.8 * len(full_train_data))
    val_size = len(full_train_data) - train_size
    train_data, val_data =random_split(full_train_data, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader , val_loader


# validation loss utility
def validation_loss(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader) # Average validation loss



# Eaerly stopping utility
def early_stopping(patience_counter, val_loss, best_loss):
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    return patience_counter , best_loss 