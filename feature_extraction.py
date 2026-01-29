
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
DATASET_PATH = "Dataset"
FEATURES_PATH = "features"

os.makedirs(FEATURES_PATH, exist_ok=True)

# ------------------------------
# Load Pretrained Model
# ------------------------------
model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model.trainable = False
print("✅ MobileNetV2 loaded")

# ------------------------------
# Image Preprocessing
# ------------------------------
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img

# ------------------------------
# Feature Extraction Function
# ------------------------------
def extract_features(base_dir):
    X, y = [], []
    class_names = sorted(os.listdir(base_dir))

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(base_dir, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            img = load_and_preprocess_image(img_path)
            img = np.expand_dims(img, axis=0)

            feature = model.predict(img, verbose=0)
            feature = feature.flatten()

            X.append(feature)
            y.append(label)

    return np.array(X), np.array(y), class_names

# ------------------------------
# Extract Training Features
# ------------------------------
print("⏳ Extracting training features...")
X_train, y_train, class_names = extract_features(
    os.path.join(DATASET_PATH, "Training")
)

# ------------------------------
# Extract Testing Features
# ------------------------------
print("⏳ Extracting testing features...")
X_test, y_test, _ = extract_features(
    os.path.join(DATASET_PATH, "Testing")
)

# ------------------------------
# Save Features
# ------------------------------
np.save(os.path.join(FEATURES_PATH, "X_train.npy"), X_train)
np.save(os.path.join(FEATURES_PATH, "y_train.npy"), y_train)
np.save(os.path.join(FEATURES_PATH, "X_test.npy"), X_test)
np.save(os.path.join(FEATURES_PATH, "y_test.npy"), y_test)
np.save(os.path.join(FEATURES_PATH, "class_names.npy"), class_names)

print("✅ Feature extraction completed")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
