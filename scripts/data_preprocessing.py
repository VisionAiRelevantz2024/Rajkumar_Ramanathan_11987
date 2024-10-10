import os
import cv2
import random
import albumentations as A

# Paths to your images and labels
IMAGE_DIR = "Dataset/train/images/"
LABEL_DIR = "Dataset/train/labels/"
OUTPUT_IMAGE_DIR = "data/preprocessed/images/"
OUTPUT_LABEL_DIR = "data/preprocessed/labels/"

# YOLO input image size
IMG_SIZE = 640

# Create output directories if they don't exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Albumentations data augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.2, p=0.7),
    A.Blur(p=0.3),
    A.CLAHE(p=0.5),
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, always_apply=True)
])

def preprocess_image(image_path, label_path):
    # Read image
    image = cv2.imread(image_path)
    
    # Load label
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Apply augmentations to the image
    augmented = transform(image=image)
    aug_image = augmented["image"]

    # Save the augmented image
    base_name = os.path.basename(image_path)
    new_image_path = os.path.join(OUTPUT_IMAGE_DIR, base_name)
    cv2.imwrite(new_image_path, aug_image)

    # Resize the label coordinates based on the new image size
    new_label_path = os.path.join(OUTPUT_LABEL_DIR, os.path.basename(label_path))
    with open(new_label_path, "w") as f:
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            
            # Since we resized to 640x640, the coordinates are the same.
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def process_dataset():
    # Get all image files
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png")]

    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(LABEL_DIR, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        if os.path.exists(label_path):
            preprocess_image(image_path, label_path)

if __name__ == '__main__':
    process_dataset()