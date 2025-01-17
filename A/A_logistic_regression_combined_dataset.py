import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the dataset
data = np.load("Datasets/BreastMNIST/breastmnist.npz")
train_images, train_labels = data['train_images'], data['train_labels'].ravel()
val_images, val_labels = data['val_images'], data['val_labels'].ravel()
test_images, test_labels = data['test_images'], data['test_labels'].ravel()

#  Combine the training and validation sets
X_train_val = np.concatenate([train_images, val_images], axis=0)
y_train_val = np.concatenate([train_labels, val_labels], axis=0)

#  Flatten images and scale the data
X_train_val = X_train_val.reshape(X_train_val.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)

#  Hyperparameter tuning with GridSearchCV
param_grid = [
    # Dictionary 1: solvers that support l1 or l2
    {
        'solver': ['liblinear', 'saga'],  
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
    },
    # Dictionary 2: solvers that only support l2
    {
        'solver': ['lbfgs', 'sag'],       
        'penalty': ['l2'],               # only 'l2' is valid here
        'C': [0.01, 0.1, 1, 10, 100],
    }
]

lr_model = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)

grid_search = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_val, y_train_val)

# Best hyperparameters
best_params = grid_search.best_params_
print("\n=== Hyperparameter Tuning Results ===")
print(f"Best Hyperparameters: {best_params}")

#  Retrain Logistic Regression with best hyperparameters
final_model = LogisticRegression(
    **best_params,
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)
final_model.fit(X_train_val, y_train_val)

# Evaluate on the test set
test_preds = final_model.predict(X_test)
test_accuracy = accuracy_score(test_labels, test_preds)

print("\n=== Final Model (Trained on Train+Val) ===")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\nClassification Report (Test Data):")
print(classification_report(test_labels, test_preds, target_names=["Benign", "Malignant"]))

print("Confusion Matrix (Test Data):")
print(confusion_matrix(test_labels, test_preds))
