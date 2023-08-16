# Fire Detection

Fire Detection is a project aimed at detecting the presence of fire in photos. Utilizing machine learning models and
image processing techniques, it can analyze images and determine whether they contain fire.

## Possible Applications

The Fire Detection project, aimed at identifying the presence of fire in photographs, can have numerous applications
across various sectors. These applications leverage the ability to automatically and accurately detect fire in images to
enhance safety, monitoring, and response mechanisms. Here's a list of possible applications.

- Emergency Response: Real-time fire detection for quicker emergency services response and efficient resource
  allocation.
- Industrial Safety: Continuous monitoring in environments prone to fire hazards like chemical plants and refineries.
- Environmental Protection: Early detection of wildfires in forest areas, aiding in prevention and wildlife protection.
- Smart Building Management: Integration with building systems for enhanced fire safety protocols and insurance risk
  assessment.
- Research and Development: Fire behavior analysis, training simulations, and research in fire prevention.
- Consumer Applications: Use in home security systems for immediate fire alerts and community-based monitoring.
- Disaster Management: Planning and mitigation strategies in broader disaster management related to fire.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/fire_detection_project.git
   ```


2. Navigate to the project repository:
    ```bash
    cd fire_detection_project
    ```

3. Create and activate a conda environment using the 'environment.yml' file:
    ```
    conda env create -f environment.yml
    conda activate your_environment_name
    ```

## Usage

To train a model, run main.py from the Fire Detection root directory:

    python src/main.py

To test the model with a visualization of its results, run visualize_predictions.py:

    python src/visualize_predictions.py

To change the architecture of the model, change def build_model(), found in main.py.

## Examples

![Fire Example](src/results/result_0.png?raw=true "Fire Correctly Predicted")
![No Fire Example](src/results/result_1.png?raw=true "No Fire Correctly Predicted")

## Dataset Acknowledgment

The dataset used in this project, "Fire Dataset," was obtained from Kaggle. You can access the
dataset [here](https://www.kaggle.com/datasets/phylake1337/fire-dataset).

We would like to thank the creator and Kaggle for making this dataset publicly available.


## Contact

If you have any questions or need further assistance, please feel free to
contact [jwhite34@uw.edu](mailto:jwhite34@uw.edu).
