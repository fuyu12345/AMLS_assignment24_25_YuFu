import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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


# 2. Flatten images (28x28 → 784) and scale (StandardScaler)

X_train = train_images.reshape(train_images.shape[0], -1)
X_val   = val_images.reshape(val_images.shape[0], -1)
X_test  = test_images.reshape(test_images.shape[0], -1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# 3. Train a "vanilla" Logistic Regression (before optimization)

vanilla_lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
vanilla_lr.fit(X_train, train_labels)

# Evaluate on test set (before hyperparameter tuning)
vanilla_test_preds = vanilla_lr.predict(X_val)
vanilla_test_accuracy = accuracy_score(val_labels, vanilla_test_preds)

print("=== Before Hyperparameter Tuning ===")
print(f"Validation Accuracy: {vanilla_test_accuracy * 100:.2f}%")
print("\nClassification Report (Val Data):")
print(classification_report(val_labels, vanilla_test_preds, target_names=["Benign", "Malignant"]))
print("Confusion Matrix (Val Data):")
print(confusion_matrix(val_labels, vanilla_test_preds))



# 4. Hyperparameter tuning with GridSearchCV on the training data
#    We use 5-fold CV and 'accuracy' as the scoring metriX

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

grid_search.fit(X_train, train_labels)

# Best hyperparameters found
print("\n=== Hyperparameter Tuning Results ===")
print(f"Best Hyperparameters: {grid_search.best_params_}")


# 5. Evaluate the best model on the validation set

best_lr = grid_search.best_estimator_
val_preds = best_lr.predict(X_val)
val_accuracy = accuracy_score(val_labels, val_preds)

print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")
print("Classification Report (Validation Data):")
print(classification_report(val_labels, val_preds, target_names=["Benign", "Malignant"]))
print("Confusion Matrix (Validation Data):")
print(confusion_matrix(val_labels, val_preds))


# 6. Evaluate the best model on the test set (after tuning)

best_test_preds = best_lr.predict(X_test)
best_test_accuracy = accuracy_score(test_labels, best_test_preds)

print("\n=== After Hyperparameter Tuning ===")
print(f"Test Accuracy: {best_test_accuracy * 100:.2f}%")
print("\nClassification Report (Test Data):")
print(classification_report(test_labels, best_test_preds, target_names=["Benign", "Malignant"]))
print("Confusion Matrix (Test Data):")
print(confusion_matrix(test_labels, best_test_preds))
