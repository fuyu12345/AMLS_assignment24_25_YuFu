import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 1. Load the dataset

data = np.load("Datasets/BreastMNIST/breastmnist.npz")

train_images = data['train_images']
train_labels = data['train_labels'].ravel()

val_images = data['val_images']
val_labels = data['val_labels'].ravel()

test_images = data['test_images']
test_labels = data['test_labels'].ravel()



best_params = {
    'C': 10,
    'penalty': 'l2',
    'solver': 'lbfgs'
}



# 2. Combine the training and validation sets

X_train_val_images = np.concatenate([train_images, val_images], axis=0)
y_train_val_labels = np.concatenate([train_labels, val_labels], axis=0)

# Flatten (28×28 → 784)
X_train_val = X_train_val_images.reshape(X_train_val_images.shape[0], -1)
X_test      = test_images.reshape(test_images.shape[0], -1)


# Scale the data
# Fit the scaler on (train+val), then transform (train+val) and test

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test      = scaler.transform(X_test)

# 4. Retrain Logistic Regression with the best hyperparameters

final_model = LogisticRegression(
    **best_params,           
    class_weight='balanced',
    max_iter=5000,          
    random_state=42
)
final_model.fit(X_train_val, y_train_val_labels)


# 5. Evaluate 

test_preds = final_model.predict(X_test)
test_accuracy = accuracy_score(test_labels, test_preds)

print("=== Final Model (Trained on Train+Val) ===")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report (Test Data):")
print(classification_report(test_labels, test_preds, target_names=["Benign", "Malignant"]))

print("Confusion Matrix (Test Data):")
print(confusion_matrix(test_labels, test_preds))
