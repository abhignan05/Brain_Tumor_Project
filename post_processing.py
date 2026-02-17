import os
import cv2
import numpy as np
import joblib

from feature_extraction import extract_features

# =========================
# Load Model + Scaler
# =========================
MODEL_PATH = "models"
FEATURES_PATH = "features"

model = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
class_names = np.load(os.path.join(FEATURES_PATH, "class_names.npy"), allow_pickle=True)

print("✅ Model loaded successfully!")
print("Classes:", class_names)


# =========================
# Tumor Percentage Estimation
# =========================
def calculate_tumor_percentage(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        raise ValueError(f"Invalid image path: {image_path}")

    # Simple threshold-based estimation
    _, thresh = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)

    tumor_pixels = np.sum(thresh == 255)
    total_pixels = img.shape[0] * img.shape[1]

    percentage = (tumor_pixels / total_pixels) * 100
    return round(float(percentage), 2)


# =========================
# Main Prediction Function
# =========================
def analyze_tumor_region(image_path):

    if not os.path.exists(image_path):
        raise ValueError(f"Invalid image path: {image_path}")

    # Feature Extraction
    features = extract_features(image_path)
    features = scaler.transform([features])

    # Prediction
    prediction = model.predict(features)[0]
    predicted_class = class_names[prediction]

    # Confidence Score (LinearSVC safe handling)
    decision_score = model.decision_function(features)

    if len(decision_score.shape) > 1:
        score = np.max(decision_score)
    else:
        score = decision_score[0]

    confidence = abs(float(score))
    confidence_percent = min(100.0, round(confidence * 100.0, 2))

    # =========================
    # No Tumor Case
    # =========================
    if predicted_class == "notumor":

        return {
            "result": "No Tumor",
            "predicted_class": predicted_class,
            "confidence": confidence_percent,
            "tumor_percentage": 0.0,
            "risk_level": "No Risk",
            "precaution": "Maintain regular health checkups and consult a doctor if neurological symptoms persist.",
            "recommendation": "No immediate action required. Continue routine medical observation."
        }

    # =========================
    # Tumor Detected Case
    # =========================
    else:

        tumor_percentage = calculate_tumor_percentage(image_path)

        # Risk Level Estimation
        if tumor_percentage < 2:
            risk_level = "Low Risk"
        elif tumor_percentage < 5:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "result": "Tumor Detected",
            "predicted_class": predicted_class,
            "confidence": confidence_percent,
            "tumor_percentage": tumor_percentage,
            "risk_level": risk_level,
            "precaution": "Consult a certified neurologist immediately for further clinical evaluation.",
            "recommendation": "Further MRI scans and medical diagnosis are strongly recommended. Do not rely solely on automated results."
        }
