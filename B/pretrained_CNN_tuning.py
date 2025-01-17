import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna

# Load dataset
data = np.load("Datasets/BloodMNIST/bloodmnist.npz")
train_images = data['train_images']   
train_labels = data['train_labels'].ravel()
val_images   = data['val_images']
val_labels   = data['val_labels'].ravel()
test_images  = data['test_images']
test_labels  = data['test_labels'].ravel()

# Preprocess the data (resize to 224x224 for ResNet)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Convert images to torch tensors
train_images_torch = torch.stack([transform(img) for img in train_images])
val_images_torch = torch.stack([transform(img) for img in val_images])
test_images_torch = torch.stack([transform(img) for img in test_images])

train_labels_torch = torch.tensor(train_labels, dtype=torch.long)
val_labels_torch = torch.tensor(val_labels, dtype=torch.long)
test_labels_torch = torch.tensor(test_labels, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(train_images_torch, train_labels_torch)
val_dataset = TensorDataset(val_images_torch, val_labels_torch)
test_dataset = TensorDataset(test_images_torch, test_labels_torch)

def create_dataloaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Objective function for Optuna
def objective(trial):
    # give suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Create DataLoaders with the current batch size
    train_loader, val_loader, _ = create_dataloaders(batch_size)

    # Load a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    num_classes = 8
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move model to GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop for 1 epoch (speed up tuning)
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
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

# Run Optuna study
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print best parameters
print("Best Hyperparameters:", study.best_params)
print("Best Validation Accuracy:", study.best_value)

