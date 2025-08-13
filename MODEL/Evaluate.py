import torch
from Model_architecture import CNNModel
from Utils import get_dataloaders
from sklearn.metrics import  classification_report
import numpy as np

# load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load('APP/cnn_model.pth', map_location=device))
model.eval()

# Load test data
train_loader, test_loader, val_loader = get_dataloaders('DATA/archive/training_set/training_set', 'DATA/archive/test_set/test_set', batch_size=32)

# Evaluate the model
y_pred = []
y_true = []
with torch.no_grad():
    for imges, labels in test_loader:
        imges = imges.to(device)
        outputs = model(imges)
        predicted = torch.argmax(outputs, 1).cpu().numpy()
        y_pred.extend(predicted)
        y_true.extend(labels.numpy())


# classification report
print(classification_report(y_true, y_pred, target_names=test_loader.dataset.classes))
