import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.color_detection import detect_fire_regions


def load_and_preprocess_images(path, label):
    images = []
    # Lower and upper bounds for HSV color space for fire detection
    # Define the lower and upper bounds for the HSV values
    lower_bound = np.array([0, 74, 200])
    upper_bound = np.array([18, 166, 230])
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            fire_regions = detect_fire_regions(img, lower_bound, upper_bound)
            if fire_regions:
                region = fire_regions[0]
                region = cv2.resize(region, (224, 224))  # Resize
                region = region / 255.0  # Normalize
                images.append((region, label))
    return images


def load_and_preprocess_data():
    # directory paths
    fire_images_path = '../fire_dataset/fire_images'
    non_fire_images_path = '../fire_dataset/non_fire_images'

    # load and preprocess images
    fire_images = load_and_preprocess_images(fire_images_path, 1)
    non_fire_images = load_and_preprocess_images(non_fire_images_path, 0)

    # combine and shuffle all images
    all_images = fire_images + non_fire_images
    np.random.shuffle(all_images)

    # separate images and labels
    images, labels = zip(*all_images)
    images = np.array(images)
    labels = np.array(labels)

    # split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # print shape to confirm
    print("Training data shape:", X_train.shape)
    print("Validation data shape", X_val.shape)
    print("Test data shape", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test
