import os
import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Path where features are saved
FEATURES_PATH = "features"

# -----------------------------
# Load extracted features
# -----------------------------
X_train = np.load(os.path.join(FEATURES_PATH, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_PATH, "y_train.npy"))
X_test = np.load(os.path.join(FEATURES_PATH, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_PATH, "y_test.npy"))

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

# -----------------------------
# Create Linear SVM model
# -----------------------------
svm_model = LinearSVC(max_iter=10000)

# -----------------------------
# Train model
# -----------------------------
print("\nTraining Linear SVM...")
svm_model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = svm_model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(svm_model, os.path.join(FEATURES_PATH, "svm_model.pkl"))
print("\nLinear SVM model saved successfully!")
