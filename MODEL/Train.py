import torch
import torch.nn as nn
import torch.optim as optim
from Utils import get_dataloaders , early_stopping , validation_loss
from Model_architecture import CNNModel


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size=32
num_epochs = 50
learning_rate = 0.001

# Load data
train_loader, test_loader , val_loader = get_dataloaders('DATA/archive/training_set/training_set' , 'DATA/archive/test_set/test_set' , batch_size=batch_size)


# Initialize model, loss function, optimizer and learning rate scheduler
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
shedule = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize variables for early stopping
patience_counter = 0
best_loss = float('inf')
early_stopping_patience = 5  # Number of epochs to wait before stopping if no improvement


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


 # Validation loss
    val_loss = validation_loss(model, criterion, val_loader, device)
    print(f'Validation Loss: {val_loss:.4f}')



    # Early stopping
    patience_counter, best_loss = early_stopping(patience_counter, val_loss, best_loss)
    if patience_counter == 0:
        torch.save(model.state_dict(), 'APP/cnn_model.pth')  # Save the best model
        print('Best model saved.')
    if patience_counter >= early_stopping_patience:
        print(f'Early stopping triggered after {epoch+1} epochs.')
        break

    
    shedule.step()  # Update learning rate       
    





