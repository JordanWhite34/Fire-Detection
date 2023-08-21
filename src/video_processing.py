import cv2
import numpy as np
from keras.models import load_model

video_path = '../fire_dataset/fire_videos/fire224.mp4'


def load_testing_model():
    return load_model('models/best_model.h5')


def read_video(path):
    """
    Opens video stream for reading frames

    Parameter:
        path: file path of video

    Returns:
        video: The video capture object
    """
    # Create a VideoCapture object
    video = cv2.VideoCapture(path)

    # Check for successful opening of video
    if not video.isOpened():
        print("Could not read video")
        return None
    else:
        return video


def preprocess_frame(p_frame):
    """
    Prepares a single video frame for the model, applying same preprocessing used during training

    Parameter:
        frame: frame image

    Returns:
        p_frame: The processed
    """

    if p_frame is not None:
        p_frame = cv2.resize(p_frame, (224, 224))  # resize
        p_frame = p_frame / 255.0  # normalize
    return p_frame


def apply_model(model, frames):
    """
    Applies model to single frame of video

    Parameters:
        model: model being used to determine fire presence
        frames: frame image list

    Returns:
        label: True or False depending on if the model 'sees' fire in the frame
    """
    frames = np.array(frames)
    predictions = model.predict(frames)
    # Convert predictions to binary labels (fire or non-fire)
    predicted_labels = (predictions > 0.5).astype(int)
    return predicted_labels


def save_results(results, results_path='results/results.txt'):
    """
    Saves predictions of model in a .txt file

    Parameter:
        results: list of True's and False's showing if fire is detected for each frame
    """
    # Open the file in write mode, creating it if it does not exist
    with open(results_path, 'w') as file:
        # Iterate through the results list
        for result in results:
            # Write 'True' or 'False' in the file on a new line
            file.write(str(result) + '\n')


def main():
    # define model
    testing_model = load_testing_model()

    # Load and preprocess the data
    video = read_video(video_path)

    # Creating list to store fire detection results for each frame
    fire_detection = []

    while True:
        # Read next frame
        ret, frame = video.read()

        # Break loop at end of video
        if not ret:
            break

        # Preprocess frame
        p_frame = preprocess_frame(frame)

        # Put processed frame in list
        fire_detection.append(p_frame)

    # Run frames through model
    fire_detection_results = apply_model(testing_model, fire_detection)

    # Save results to a text file in /results/results.txt
    save_results(fire_detection_results)


if __name__ == "__main__":
    main()
