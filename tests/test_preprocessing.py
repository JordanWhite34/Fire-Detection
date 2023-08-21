import numpy as np
from src.preprocessing import load_and_preprocess_images


def test_load_and_preprocess_images():
    path = '../fire_dataset/fire_images'
    label = 1
    images = load_and_preprocess_images(path, label)

    # Check if images are loaded
    assert len(images) > 0

    # Check if images are resized and normalized
    for img, lbl in images:
        assert img.shape == (224, 224, 3)
        assert np.max(img) <= 1.0
        assert lbl == label
