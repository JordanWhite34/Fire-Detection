import os
import tempfile
import cv2
import numpy as np
from keras.models import load_model
from src.video_processing import read_video, preprocess_frame, apply_model, save_results


def test_read_video():
    # Test with valid video path
    video = read_video('../fire_dataset/fire_videos/fire224.mp4')
    assert video is not None
    assert isinstance(video, cv2.VideoCapture)

    # Test with invalid video path
    video = read_video('invalid/path')
    assert video is None


def test_preprocess_frame():
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    p_frame = preprocess_frame(frame)

    # Check if the frame is resized and normalized
    assert p_frame.shape == (224, 224, 3)
    assert np.max(p_frame) <= 1.0

    # Test with None input
    assert preprocess_frame(None) is None


def test_apply_model():
    # Load the model
    model = load_model('../src/models/saved_model.h5')

    # Create dummy frames
    frames = [np.zeros((224, 224, 3)) for _ in range(5)]
    frames = np.array(frames)

    predicted_labels = apply_model(model, frames)

    # Check if the predicted labels are binary
    assert predicted_labels.shape == (5, 1)
    assert set(predicted_labels.flatten()) <= {0, 1}


def test_save_results():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy results
        results = [True, False, True, False]

        # Path to the temporary results file
        results_path = os.path.join(temp_dir, 'results.txt')

        # Call the save_results function with the temporary path
        save_results(results, results_path)

        # Read the saved file and check if the results are correct
        with open(results_path, 'r') as file:
            saved_results = [line.strip() == 'True' for line in file.readlines()]
            assert saved_results == results
