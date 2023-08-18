import cv2

video_path = '../fire_dataset/fire_videos/fire.mp4'


def read_video(video_path):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: video could not be opened")
    else:
        # Loop through video frames
        while True:
            # Reading next frame
            ret, frame = video.read()
            if not ret:
                break  # Breaking loop if end of video

            cv2.imshow('Video', frame)  # Displaying frame

            if cv2.waitKey(30) and 0xFF == ord('q'):
                break

        # Releasing video object, closing windows
        video.release()
        cv2.destroyAllWindows()


def main():
    # Load and preprocess the data
    read_video(video_path)


if __name__ == "__main__":
    main()
