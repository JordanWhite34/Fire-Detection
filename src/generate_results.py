from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os
from preprocessing import load_and_preprocess_data


def main():
    # Load the Model
    model_path = 'models/saved_model.h5'
    model = load_model(model_path)

    # Load and Preprocess Some Test Images
    _, _, X_test, _, _, y_test = load_and_preprocess_data()

    # Make Predictions
    predictions = model.predict(X_test)

    # Convert predictions to binary labels (fire or non-fire)
    predicted_labels = (predictions > 0.5).astype(int)

    # Create a results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Plot and save the first few images
    for i in range(5):
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor((X_test[i] * 255).astype('uint8'), cv2.COLOR_BGR2RGB)

        # Get the true and predicted labels as "fire" or "no fire"
        true_label = "fire" if y_test[i] == 1 else "no fire"
        predicted_label = "fire" if predicted_labels[i] == 1 else "no fire"

        plt.imshow(image_rgb)
        plt.title(f"True label: {true_label} - Predicted label: {predicted_label}")

        # Save the plot as an image file
        plt.savefig(os.path.join(results_dir, f'result_{i}.png'))

        # Close the plot to free up resources
        plt.close()

    print(f"Results saved in {results_dir} directory.")


if __name__ == "__main__":
    main()
