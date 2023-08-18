import cv2

video_path = '../fire_dataset/fire_videos/fire.mp4'


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

    p_frame = frame

    if p_frame is not None:
        p_frame = cv2.resize(p_frame, (224, 224))  # resize
        p_frame = p_frame / 255.0  # normalize

    return p_frame



def main():
    # Load and preprocess the data
    read_video(video_path)


if __name__ == "__main__":
    main()
