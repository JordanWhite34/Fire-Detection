# Fire Detection using Convolutional Neural Networks (CNNs)

## Introduction

This repository contains a fire detection system implemented using Convolutional Neural Networks (CNNs). The system is
designed to classify images and video frames as containing fire or not. It includes preprocessing of images, building
and training a CNN, evaluating the model, generating visual results, and processing video inputs.

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV
- Keras

### Setup Environment

Clone the repository and navigate to the project directory. Then, create a conda environment using the
provided `environment.yml` file:

    git clone https://github.com/JordanWhite34/Fire-Detection.git
    cd Fire-Detection
    conda env create -f environment.yml
    conda activate fire-detection-env

## Usage

### Training the Model

Train the CNN model using the provided dataset:

    python src/main.py

### Evaluating the Model

Evaluate the trained model using various metrics (Accuracy, Precision, Recall, F1, Confusion Matrix):

    python src/evaluation.py

### Generating Results

Generate visual results for test images:

    python src/generate_results.py

### Processing Videos

Detect fire in video files:

    python src/video_processing.py

## Project Structure

- `src/`: Source code directory containing Python scripts for training, evaluation, result generation, and video
  processing.
- `fire_dataset/`: Dataset directory containing fire and non-fire images.
- `results/`: Directory for storing result images and text files.
- `environment.yml`: Conda environment file.
- `requirements.txt`: Requirements file for pip.

## Testing

Run the tests using pytest (and make sure you are in the tests directory):

    cd tests
    pytest

## Current Efforts

Right now I am working on implementing an object detection pre-step so when the preprocessing crops the photo, it won't
crop out the fire in the photo.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to
check [issues page](https://github.com/JordanWhite34/Fire-Detection/issues) or open a pull request.

## License

This project is MIT licensed.

## Acknowledgments

Special thanks to Kaggle for the dataset and anyone who finds this project useful.
