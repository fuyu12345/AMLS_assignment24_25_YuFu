import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

# 1. Load the dataset 
data = np.load("Datasets/BloodMNIST/bloodmnist.npz")

train_images = data['train_images']    
train_labels = data['train_labels'].ravel() 
val_images   = data['val_images']     
val_labels   = data['val_labels'].ravel()   
test_images  = data['test_images']     
test_labels  = data['test_labels'].ravel()  
print(train_images.shape)
print(train_labels.shape)



# Dataset information display

# def dataset_summary(images, labels, dataset_name):
#     print(f"=== {dataset_name} Dataset ===")
#     print(f"Number of samples: {images.shape[0]}")
#     print(f"Image shape: {images.shape[1:]} (Height x Width x Channels)")
#     print(f"Data type of images: {images.dtype}")
#     print(f"Data type of labels: {labels.dtype}")
#     print(f"Labels: {np.unique(labels)}")
#     print(f"Class distribution:")
#     for label in np.unique(labels):
#         count = np.sum(labels == label)
#         print(f"  Label {label}: {count} samples ({(count / len(labels)) * 100:.2f}%)")
#     print("\n")

# Display dataset summaries

# dataset_summary(train_images, train_labels, "Training")
# dataset_summary(val_images, val_labels, "Validation")
# dataset_summary(test_images, test_labels, "Testing")

# Create Torch Tensors
# Convert images to float32, labels to long 
train_images_torch = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
val_images_torch   = torch.tensor(val_images,   dtype=torch.float32).permute(0, 3, 1, 2)
test_images_torch  = torch.tensor(test_images,  dtype=torch.float32).permute(0, 3, 1, 2)


train_labels_torch = torch.tensor(train_labels, dtype=torch.long)
val_labels_torch   = torch.tensor(val_labels,   dtype=torch.long)
test_labels_torch  = torch.tensor(test_labels,  dtype=torch.long)
print(train_images_torch.shape)
print(train_labels_torch.shape)

# transforms 
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_images_torch = normalize(train_images_torch / 255.0)
val_images_torch   = normalize(val_images_torch / 255.0)
test_images_torch  = normalize(test_images_torch / 255.0)


#Create Dataset and Loader
batch_size = 64
train_dataset = TensorDataset(train_images_torch, train_labels_torch)
val_dataset   = TensorDataset(val_images_torch,   val_labels_torch)
test_dataset  = TensorDataset(test_images_torch,  test_labels_torch)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


#Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        # Update in_channels to 3 for RGB input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 3, 28, 28)
        x = F.relu(self.conv1(x))  
        x = F.max_pool2d(x, 2)     

        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, 2)     
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))   
        x = self.fc2(x)           
        
        return x


# give the model
model = SimpleCNN(num_classes=8)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Train + Eval
num_epochs = 20

# Initialize lists to store metrics
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    epoch_train_accuracy = train_correct / train_total
    train_accuracies.append(epoch_train_accuracy)

    # Validation Loop
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

    epoch_val_accuracy = val_correct / val_total
    val_accuracies.append(epoch_val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Acc: {epoch_train_accuracy:.4f}, "
          f"Val Acc: {epoch_val_accuracy:.4f}")



# Evaluate on the test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

print("\nClassification Report (Test Data):")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix (Test Data):")
print(confusion_matrix(all_labels, all_preds))

# --------------------
# Plot the Learning Curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid()
plt.show()
