from torchvision import transforms, datasets 
from torch.utils.data.dataloader import DataLoader

# data loading utilities
def get_dataloaders(train_dataset, test_dataset, batch_size=32):
    Transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_data= datasets.ImageFolder(train_dataset, transform=Transforms)
    test_data = datasets.ImageFolder(test_dataset, transform=Transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


