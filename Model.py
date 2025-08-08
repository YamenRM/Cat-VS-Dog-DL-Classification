import torch
from torch import nn
from torch.nn import functional as F


# bulding the architecture of the CNN model

class CNNModel(nn.Module):
    def __init__(self):
     super().__init__()
     self.conv1 = nn.Conv2d(3 , 16 , 3)
     self.conv2 = nn.Conv2d(16, 32, 3)
     self.pool = nn.MaxPool2d(2, 2)
     self.fc1 = nn.linear(32*62*62, 128)
     self.fc2 = nn.Linear(128, 2) 

    # forward pass of the model
def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 32 * 62 * 62)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# finshed model architecture