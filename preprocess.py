import cv2
import os
import numpy as np

IMG_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img


def load_dataset(base_path):
    X = []
    y = []
    class_names = os.listdir(base_path)

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(base_path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = preprocess_image(img_path)
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), class_names


# -------- TEST FULL DATASET LOADING --------
if __name__ == "__main__":
    train_path = "Dataset/Training"
    X_train, y_train, class_names = load_dataset(train_path)

    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Classes:", class_names)
