import torch
from torch import nn
from torch.nn import functional as F


# bulding the architecture of the CNN model

class CNNModel(nn.Module):
    def __init__(self):
     super().__init__()
     self.conv1 = nn.Conv2d(3 , 16 , 3)
     self.conv2 = nn.Conv2d(16, 32, 3)
     self.bn1 = nn.BatchNorm2d(16)
     self.bn2 = nn.BatchNorm2d(32)
     self.pool = nn.MaxPool2d(2, 2)
     self.dropout = nn.Dropout(0.5)

# Pass a dummy input to get the flattened size
     dummy = torch.randn(1, 3, 128, 128)  
     x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
     x = self.pool(F.relu(self.bn2(self.conv2(x))))
     flattened_size = x.numel()  # total number of elements
        
     self.fc1 = nn.Linear(flattened_size, 128)
     self.fc2 = nn.Linear(128, 2)


# forward pass of the model
    def forward(self, x):
      x= self.pool(F.relu(self.bn1(self.conv1(x))))
      x= self.pool(F.relu(self.bn2(self.conv2(x))))
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      return x


# finshed model architecture