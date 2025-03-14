import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import models
import numpy as np

X_train_np = np.array(np.load("data/tensors/data_train.npy", allow_pickle=True), dtype = np.float32)
X_test_np = np.array(np.load("data/tensors/data_test.npy", allow_pickle=True), dtype = np.float32)

X_train = torch.from_numpy(X_train_np).float()
X_test = torch.from_numpy(X_test_np).float()

Y_train_np = np.array(np.load("labels/Y_train.npy", allow_pickle=True), dtype = np.float32)
Y_test_np = np.array(np.load("labels/Y_test.npy", allow_pickle=True), dtype = np.float32)

Y_train = torch.from_numpy(Y_train_np).float()
Y_test = torch.from_numpy(Y_test_np).float()

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = models.EmpathyTransformer(d_model=1000, n_heads=4, num_transformer_layers=1)

criterion = nn.BCEWithLogitsLoss()  # binary classification with raw logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 15

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()

        logits = model(batch_inputs)  # Expected shape: (batch_size, 1)

        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)

        # Convert logits to probabilities using sigmoid, then threshold at 0.5
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

torch.save(model.state_dict(), "empathy_transformer.pth")


test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

criterion = torch.nn.BCEWithLogitsLoss()

# Switch the model to evaluation mode
model.eval()

# Initialize accumulators for loss and accuracy
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        # Forward pass: get predictions (logits)
        logits = model(batch_inputs)  # expected shape: (batch_size, 1)

        # Compute loss for this batch
        loss = criterion(logits, batch_labels)
        test_loss += loss.item() * batch_inputs.size(0)

        # Compute predictions and count correct ones:
        preds = (torch.sigmoid(logits) > 0.5).float()  # Convert logits to binary predictions (0 or 1)
        correct += (preds == batch_labels).sum().item()
        total += batch_inputs.size(0)

# Average validation loss and accuracy
avg_loss = test_loss / total
test_accuracy = correct / total

print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}")