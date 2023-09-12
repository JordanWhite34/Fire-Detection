import cv2
from color_detection import detect_fire_region, visualize_detection
from keras.models import load_model
import numpy as np


def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(1)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Enable detailed exceptions
    cap.setExceptionMode(True)

    # Load your trained fire detection model
    model = load_model('models/saved_model.h5')

    # Loop to continuously get frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect potential fire regions
        fire_region, x, y, largest_contour = detect_fire_region(frame)

        if fire_region is not None:
            # Preprocess the fire_region for your model
            fire_region = cv2.resize(fire_region, (224, 224))
            fire_region = np.expand_dims(fire_region, axis=0)

            # Predict using your fire detection model
            prediction = model.predict(fire_region)

            # Determine the color of the bounding box
            color = (0, 255, 0)  # Green by default
            if prediction[0][0] > 0.5:  # Assuming your model outputs a probability for the 'fire' class
                color = (0, 0, 255)  # Change to Red if fire is detected

            # Visualize the detected fire regions
            if x is not None and y is not None:
                x, y, w, h = cv2.boundingRect(
                    largest_contour)  # You'll need to get 'largest_contour' from your detect_fire_region function
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the frame
        cv2.imshow('Real-Time Fire Detection', frame)

        # Break the loop if 'q' is pressed by user
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
