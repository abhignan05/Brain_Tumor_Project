import os
import cv2

train_path = "Dataset/Training"
test_path = "Dataset/Testing"

def check_data(path, name):
    print(f"\n{name} Dataset:")
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            images = os.listdir(cls_path)
            print(f"{cls}: {len(images)} images")

            img = cv2.imread(os.path.join(cls_path, images[0]))
            print(f"Sample image shape for {cls}: {img.shape}")

check_data(train_path, "Training")
check_data(test_path, "Testing")
