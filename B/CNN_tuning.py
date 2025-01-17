import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import optuna


# Load the data
data = np.load("Datasets/BloodMNIST/bloodmnist.npz")

train_images = data['train_images']
train_labels = data['train_labels'].ravel()
val_images = data['val_images']
val_labels = data['val_labels'].ravel()
test_images = data['test_images']
test_labels = data['test_labels'].ravel()


# Create Torch Tensors
train_images_torch = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
val_images_torch = torch.tensor(val_images, dtype=torch.float32).permute(0, 3, 1, 2)
test_images_torch = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)

train_labels_torch = torch.tensor(train_labels, dtype=torch.long)
val_labels_torch = torch.tensor(val_labels, dtype=torch.long)
test_labels_torch = torch.tensor(test_labels, dtype=torch.long)

# transforms
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_images_torch = normalize(train_images_torch / 255.0)
val_images_torch = normalize(val_images_torch / 255.0)
test_images_torch = normalize(test_images_torch / 255.0)


# Create Dataset and Loader
def create_loaders(batch_size):
    train_dataset = TensorDataset(train_images_torch, train_labels_torch)
    val_dataset = TensorDataset(val_images_torch, val_labels_torch)
    test_dataset = TensorDataset(test_images_torch, test_labels_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Define a CNN Architecture
class TunedCNN(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, num_classes=8):
        super(TunedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(conv2_filters * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 5. Define the Objective Function for Optuna
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    conv1_filters = trial.suggest_int('conv1_filters', 8, 64, step=8)
    conv2_filters = trial.suggest_int('conv2_filters', 16, 128, step=16)

    # Create model and optimizer
    model = TunedCNN(conv1_filters, conv2_filters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders
    train_loader, val_loader, _ = create_loaders(batch_size)

    # Train for 1 epoch (fast tuning)
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    return val_accuracy

# 6. Run Optuna Study
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print the best parameters
print("Best Hyperparameters:", study.best_params)
print("Best Validation Accuracy:", study.best_value)


# Final Training and Evaluation with Best Hyperparameters
best_params = study.best_params
final_model = TunedCNN(best_params['conv1_filters'], best_params['conv2_filters']).to(device)
final_criterion = nn.CrossEntropyLoss()
final_optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])

train_loader, val_loader, test_loader = create_loaders(best_params['batch_size'])

num_epochs = 20
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    final_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        final_optimizer.zero_grad()
        outputs = final_model(images)
        loss = final_criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    final_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = final_model(images)
            loss = final_criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_correct / val_total

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, "
          f"Val Acc: {epoch_val_acc:.4f}")

# Test the final model
final_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = final_model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print("\nClassification Report (Test Data):")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix (Test Data):")
print(confusion_matrix(all_labels, all_preds))
