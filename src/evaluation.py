from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .preprocessing import load_and_preprocess_data


def metrics(model_path, X_test, y_test):
    """
    Evaluates the given model using the provided test data.

    Parameters:
        model_path: Path to the saved model file.
        X_test, y_test: The test data and corresponding labels.

    Prints:
        Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
    """

    # Load the Model
    model = load_model(model_path)

    # Make Predictions
    predictions = model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int)

    # Calculate Metrics
    accuracy = accuracy_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # Print Metrics
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix


# Example Usage
if __name__ == "__main__":
    model_path = 'models/saved_model.h5'
    _, _, X_test, _, _, y_test = load_and_preprocess_data()
    metrics(model_path, X_test, y_test)
