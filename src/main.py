from preprocessing import load_and_preprocess_data
from evaluation import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_model():
    """
    Builds and compiles a Convolutional Neural Network (CNN) for binary classification.
    The model consists of three convolutional layers, followed by a fully connected layer,
    and an examples layer with a single neuron using a sigmoid activation function.

    Returns:
        model: A compiled Keras model ready for training.
    """

    # Initializing Sequential Keras model
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the examples to feed into a Dense Layer
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout to prevent overfitting

    # Output layer with single neuron and sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with binary cross-entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # returning the model
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the given model using the provided training and validation data.

    - Utilizes a medium batch size of 32.
    - Trains for up to 100 epochs with early stopping.
    - Applies standard data augmentation techniques.
    - Uses a fixed validation split of 20%.
    - Includes standard callbacks for early stopping and model checkpointing.

    Parameters:
        model: The neural network model to be trained.
        X_train, y_train: The training data and corresponding labels.
        X_val, y_val: The validation data and corresponding labels.

    Returns:
        history: The training history object containing details about the training process.
    """

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Early Stopping and Model Checkpointing
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('models/saved_model.h5', save_best_only=True)
    ]

    # Training Configuration
    batch_size = 32
    epochs = 5
    # epochs = 100

    # Training the Model
    model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )

    # Returning the training history object
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the given model using the provided validation data

    Parameters:
        model: The trained neural network to be evaluated
        X_test, y_test: The testing data and corresponding labels

    Returns:
        loss: The loss value on the testing data
        accuracy: The accuracy value on the testing data
    """

    # Evaluating the Model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("Test Loss:", loss)
    print("Test accuracy", accuracy)

    return loss, accuracy


def save_model(model, file_path='models/saved_model.h5'):
    """
    Saves the given trained model to the specified file path.

    Parameters:
        model: The trained neural network model to be saved.
        file_path: The file path where the model will be saved (default is 'saved_model.h5').

    Returns:
        None
    """

    # Saving the model
    model.save(file_path)

    print("Model saved to ", file_path)


def main():
    # Load and preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Build the model
    model = build_model()

    # Train the model
    model = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model (optional)
    save_model(model)

    # Run metrics on model
    metrics('models/saved_model.h5', X_test, y_test)


if __name__ == "__main__":
    main()
