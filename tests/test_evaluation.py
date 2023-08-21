from src.evaluation import metrics
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def test_metrics():
    # Create a dummy model for testing
    model_path = '../src/models/saved_model.h5'
    model = Sequential([
        Dense(10, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    model.save(model_path)

    # Create dummy test data
    X_test = np.random.rand(100, 10)
    y_test = np.random.randint(2, size=100)

    # Call the metrics function
    accuracy, precision, recall, f1, conf_matrix = metrics(model_path, X_test, y_test)

    # Check if the returned metrics are in the expected range
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

    # Check if the confusion matrix has the correct shape
    assert conf_matrix.shape == (2, 2)
