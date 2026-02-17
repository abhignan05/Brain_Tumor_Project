import os
import cv2
import numpy as np
import joblib

from feature_extraction import extract_features

# =========================
# Load Model + Scaler ONCE
# =========================
MODEL_PATH = "models"
FEATURES_PATH = "features"

model = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

# Load class names (VERY IMPORTANT for multi-class)
class_names = np.load(os.path.join(FEATURES_PATH, "class_names.npy"), allow_pickle=True)

print("✅ Model loaded successfully!")
print("Classes:", class_names)


# =========================
# Tumor Segmentation
# =========================
def calculate_tumor_percentage(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        raise ValueError(f"❌ Image not found at path: {image_path}")

    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    tumor_pixels = np.sum(thresh == 255)
    total_pixels = img.shape[0] * img.shape[1]

    percentage = (tumor_pixels / total_pixels) * 100
    return round(float(percentage), 2)


# =========================
# Prediction Function
# =========================
def analyze_tumor_region(image_path):

    if not os.path.exists(image_path):
        raise ValueError(f"❌ Invalid image path: {image_path}")

    # Extract features
    features = extract_features(image_path)

    # Scale features
    features = scaler.transform([features])

    # Predict class index
    prediction = model.predict(features)[0]

    # Get actual class name
    predicted_class = class_names[prediction]

    # =========================
    # Confidence Handling (Safe for LinearSVC)
    # =========================
    decision_score = model.decision_function(features)

    if len(decision_score.shape) > 1:
        score = np.max(decision_score)
    else:
        score = decision_score[0]

    confidence = abs(float(score))
    confidence_percent = min(100.0, round(confidence * 100.0, 2))

    # =========================
    # Tumor / No Tumor Logic
    # =========================
    if predicted_class == "notumor":
        return {
            "result": "No Tumor",
            "predicted_class": predicted_class,
            "confidence": confidence_percent,
            "tumor_percentage": 0.0
        }

    else:
        tumor_percentage = calculate_tumor_percentage(image_path)

        return {
            "result": f"Tumor Detected",
            "predicted_class": predicted_class,
            "confidence": confidence_percent,
            "tumor_percentage": tumor_percentage
        }
