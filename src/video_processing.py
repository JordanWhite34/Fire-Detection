import cv2
from keras.models import load_model


video_path = '../fire_dataset/fire_videos/fire.mp4'
testing_model = load_model('models/best_model.h5')


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


def preprocess_frame(frame):
    """
    Prepares a single video frame for the model, applying same preprocessing used during training

    Parameter:
        frame: frame image

    Returns:
        p_frame: The processed
    """

    # Right now I don't think this will work when the fire is outside the crop
    # TODO: Adjust it somehow so crop doesnt mess it up

    p_frame = frame

    if p_frame is not None:
        p_frame = cv2.resize(p_frame, (224, 224))  # resize
        p_frame = p_frame / 255.0  # normalize

    return p_frame


def apply_model(model, frame):
    """
    Applies model to single frame of video

    Parameter:
        model: used model
        frame: frame image

    Returns:
        label: Fire or fire depending on how sure the model is
    """
    prediction = model.predict(frame)
    label = "Fire" if prediction[0] > 0.5 else "No Fire"
    return label


def main():
    # Load and preprocess the data
    video = read_video(video_path)

    # Creating list to store fire detection results for each frame
    fire_detection_results = []

    while True:
        # Read next frame
        ret, frame = video.read()

        # Preprocess frame
        p_frame = preprocess_frame(frame)

        # Run frame through model
        fire_detection_results.append(apply_model(testing_model, p_frame))

        # Break loop at end of video
        if not ret:
            break


if __name__ == "__main__":
    main()
