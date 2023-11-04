# User Engagement Prediction Model

## Overview

The primary goal of this project is to build a robust predictive model that estimates user engagement by analyzing the content of articles. Employing a blend of regression and classification training approaches allows the model to understand and predict user interaction levels based on the article's content.


## Code Structure

The codebase is organized as follows:

### Source Code (`src/`)

- **Data** (`data/`)
    - `dataset.py`: Python script containing dataset handling functions.
    - `__init__.py`: Initialization file for the data module.

- **Model** (`model/`)
    - `model.py`: Script with the implementation of the engagement prediction model.
    - `__init__.py`: Initialization file for the model module.

- `myutils.py`: Utility functions used across the project.
- `__init__.py`: Main initialization file for the source code.

- **Scripts** (`scripts/`)
    - `annotate_spiegel.py`: Script for annotating articles for training.
    - `download_dataset.py`: Script to download the required dataset.
    - `train.py`: Main training script for the engagement prediction model.

- **Storage** (`storage/`)
    - `storage_client.py`: Module handling storage-related operations.
    - `storage_exceptions.py`: Custom exceptions for storage operations.
    - `__init__.py`: Initialization file for the storage module.

- **Train** (`train/`)
    - `trainer.py`: Script defining the training process for the model.
    - `__init__.py`: Initialization file for the training module.

The codebase is structured to encapsulate specific functionalities within modules and scripts for ease of maintenance and organization.

## Usage

1. **Data Handling**: Utilize `dataset.py` to preprocess and handle the dataset.
2. **Model Building**: Implement the engagement prediction model using `model.py`.
3. **Training**: Execute `train.py` to train the model on the dataset.
4. **Storage**: Utilize `storage_client.py` for storage-related operations.

## Setup

To set up the environment and run the project:

1. Ensure Python 3.10 is installed.
2. Install the required dependencies by running: `pip install -r requirements.txt`.
3. Execute specific scripts in the `scripts/` directory for data exploration and training.


## License

This project is licensed under the [MIT License](LICENSE).