import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# 1. Load the dataset
data = np.load("Datasets/BreastMNIST/breastmnist.npz")

train_images = data['train_images']
train_labels = data['train_labels'].ravel()

val_images = data['val_images']
val_labels = data['val_labels'].ravel()

test_images = data['test_images']
test_labels = data['test_labels'].ravel()

# 2. Combine the training and validation sets
X_train_val_images = np.concatenate([train_images, val_images], axis=0)
y_train_val_labels = np.concatenate([train_labels, val_labels], axis=0)

# Flatten (28×28 → 784)
X_train_val = X_train_val_images.reshape(X_train_val_images.shape[0], -1)
X_test = test_images.reshape(test_images.shape[0], -1)

# 3. Scale the data
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)
X_test = scaler.transform(X_test)



# 4. Hyperparameter tuning for SVM
param_grid_svc = {
    'C':      [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma':  ['scale', 'auto']  # gamma options for kernel='rbf', 'poly', or 'sigmoid'
}

svc_for_gridsearch = SVC(
    class_weight='balanced',  # handle class imbalance
    max_iter=5000,            # increase iteration limit if needed
    random_state=42
)

grid_search_svc = GridSearchCV(
    estimator=svc_for_gridsearch,
    param_grid=param_grid_svc,
    scoring='accuracy',
    cv=5,    # 5-fold cross-validation
    n_jobs=-1,
    verbose=1
)

grid_search_svc.fit(X_train_val, y_train_val_labels)

print("\n=== Hyperparameter Tuning Results (SVM) ===")
print(f"Best Hyperparameters: {grid_search_svc.best_params_}")
print(f"Best CV Accuracy: {grid_search_svc.best_score_:.4f}")

# 5. Retrain the final model (SVC) on (train+val) with the best hyperparameters
best_params_svc = grid_search_svc.best_params_

final_svc_model = SVC(
    **best_params_svc,
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)

# final_svc_model = SVC(
#     C=10,
#     kernel='linear',
#     gamma='scale',
#     class_weight='balanced',
#     max_iter=5000,
    
# )


final_svc_model.fit(X_train_val, y_train_val_labels)

# 6. Evaluate on the test set
test_preds_svc = final_svc_model.predict(X_test)
test_accuracy_svc = accuracy_score(test_labels, test_preds_svc)

print("\n=== Final SVM Model (Trained on Train+Val) ===")
print(f"Test Accuracy: {test_accuracy_svc * 100:.2f}%")

print("\nClassification Report (Test Data):")
print(classification_report(test_labels, test_preds_svc, target_names=["Benign", "Malignant"]))

print("Confusion Matrix (Test Data):")
print(confusion_matrix(test_labels, test_preds_svc))



train_sizes, train_scores, valid_scores = learning_curve(
    estimator=final_svc_model,
    X=X_train_val,
    y=y_train_val_labels,
    train_sizes=np.linspace(0.1, 1, 5),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Compute mean and standard deviation across folds
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std  = np.std(valid_scores, axis=1)


plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.2, color='blue')

plt.plot(train_sizes, valid_mean, 'o-', color='green', label='Validation score')
plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std,
                 alpha=0.2, color='green')

plt.title("Learning Curve for SVC (BreastMNIST)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()