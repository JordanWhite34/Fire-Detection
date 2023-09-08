import cv2
from color_detection import detect_fire_region, visualize_detection
from keras.models import load_model
import numpy as np

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load your trained fire detection model
    model = load_model('path/to/your/saved/model.h5')

    # Loop to continuously get frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect potential fire regions
        fire_region, x, y = detect_fire_region(frame)

        if fire_region is not None:
            # Preprocess the fire_region for your model
            fire_region = cv2.resize(fire_region, (224, 224))
            fire_region = np.expand_dims(fire_region, axis=0)

            # Predict using your fire detection model
            prediction = model.predict(fire_region)

            # Visualize the detected fire regions
            frame_with_boxes = visualize_detection(frame, x, y, prediction)

            # Display the frame
            cv2.imshow('Real-Time Fire Detection', frame_with_boxes)

        else:
            cv2.imshow('Real-Time Fire Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
