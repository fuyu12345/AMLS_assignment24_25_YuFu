import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = np.load("Datasets/BreastMNIST/breastmnist.npz")
train_images = data['train_images']  # Training images
train_labels = data['train_labels'].ravel()  # Training labels (flatten to 1D array)
test_images = data['test_images']  # Test images
test_labels = data['test_labels'].ravel()  # Test labels (flatten to 1D array)

# Flatten images into vectors (28x28 → 784)
X_train = train_images.reshape(train_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

# Normalize pixel values to [0, 1]
X_train = normalize(X_train)
X_test = normalize(X_test)

# Train an SVM with RBF kernel
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')  # Experiment with kernel='linear' or 'poly' if needed
svm_classifier.fit(X_train, train_labels)

# Make predictions 
y_pred = svm_classifier.predict(X_test)


accuracy = accuracy_score(test_labels, y_pred)
print(f"SVM Classification Accuracy: {accuracy * 100:.2f}%")


from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=["Benign", "Malignant"]))

print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, y_pred))
from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf']
# }
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# grid.fit(X_train, train_labels)

# print("Best Parameters:", grid.best_params_)
