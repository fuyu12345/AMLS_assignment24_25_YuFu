import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

# 1. Load the dataset

data = np.load("Datasets/BreastMNIST/breastmnist.npz")

train_images = data['train_images']
train_labels = data['train_labels'].ravel()

val_images = data['val_images']
val_labels = data['val_labels'].ravel()

test_images = data['test_images']
test_labels = data['test_labels'].ravel()



best_params = {
    'C': 1,
    'penalty': 'l2',
    'solver': 'sag'
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


lr_for_gridsearch = LogisticRegression(
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)

grid_search = GridSearchCV(
    lr_for_gridsearch,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_val, y_train_val_labels)

# Best hyperparameters found
print("\n=== Hyperparameter Tuning Results ===")
print(f"Best Hyperparameters: {grid_search.best_params_}")