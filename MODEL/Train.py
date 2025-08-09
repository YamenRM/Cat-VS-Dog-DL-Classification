import torch
import torch.nn as nn
import torch.optim as optim
from Utils import get_dataloaders
from Model_architecture import CNNModel

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size=32
num_epochs = 10
learning_rate = 0.001

# Load data
train_loader, test_loader = get_dataloaders('DATA/archive/training_set/training_set' , 'DATA/archive/test_set/test_set' , batch_size=batch_size)


# Initialize model, loss function, and optimizer
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved successfully.")

