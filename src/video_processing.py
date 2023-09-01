import cv2
import numpy as np
from keras.models import load_model
from src.color_detection import detect_fire_region, visualize_detection, find_contour

video_path = '../fire_dataset/fire_videos/fire.mp4'


def load_testing_model():
    return load_model('models/saved_model.h5')


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
    if p_frame is None:
        return None

    if isinstance(p_frame, np.ndarray):
        p_frame = cv2.resize(p_frame, (224, 224))  # Resize
        p_frame = p_frame / 255.0  # Normalize
        return p_frame
    else:
        print(f"Unexpected type for fire_region: {type(p_frame)}")
        return None


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

    # Get the frame rate of the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # Calculate the delay between frames in milliseconds
    delay = int(1000 / frame_rate)

    # Creating list to store fire detection results for each frame
    fire_detection = []

    while True:
        # Read next frame
        ret, frame = video.read()

        # Break loop at end of video
        if not ret:
            break

        # Detect fire regions, getting largest contour and offsets
        fire_region, x_offset, y_offset = detect_fire_region(frame)

        # If a fire region is detected, visualize it
        if fire_region is not None:
            largest_contour = find_contour(cv2.cvtColor(fire_region, cv2.COLOR_BGR2GRAY))  # Find largest contour
            if largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                x += x_offset  # Adjust x coordinate
                y += y_offset  # Adjust y coordinate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle directly
            frame_with_detection = frame
        else:
            frame_with_detection = frame

        # Display visualized frame
        cv2.imshow('Fire Detection', frame_with_detection)
        cv2.waitKey(delay)

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
