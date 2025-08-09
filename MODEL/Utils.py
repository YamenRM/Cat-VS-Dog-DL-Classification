import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 

# data loading utilities
def get_dataloaders(train_dataset, test_dataset, batch_size=32):
    transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_data= Dataset.image_folder(root=train_dataset, transform=transforms)
    test_data = Dataset.image_folder(root=test_dataset, transform=transforms)

    train_loader = DataLoader(root=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(root=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
# end of data loading

