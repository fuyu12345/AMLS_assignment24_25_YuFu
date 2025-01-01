import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = np.load("Datasets/BreastMNIST/breastmnist.npz")
train_images = data['train_images']  # Training images
train_labels = data['train_labels'].ravel()  # Flatten training labels
test_images = data['test_images']  # Test images
test_labels = data['test_labels'].ravel()  # Flatten test labels

# Flatten images into vectors (28x28 → 784)
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

# Normalize the pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, train_labels)

# Predict 
y_test_pred = rf_model.predict(X_test)


test_accuracy = accuracy_score(test_labels, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


print("\nClassification Report (Test Data):")
print(classification_report(test_labels, y_test_pred, target_names=["Benign", "Malignant"]))


print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(test_labels, y_test_pred))
