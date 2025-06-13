# Menstrual Phase Classifier

This project provides tools for classifying menstrual cycle phases (luteal vs. follicular) based on hormone measurements and simulating test data.

## Project Structure

- `phase_classifier.py`: Contains the `MenstrualPhaseClassifier` class that implements the phase classification logic
- `simulate_data.py`: Provides functions for generating synthetic hormone data for testing
- `requirements.txt`: Lists the project dependencies

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Simulation

The `simulate_data.py` script can be used to generate synthetic hormone data:

```python
from simulate_data import simulate_hormone_data, create_train_test_split

# Generate synthetic data
X, y = simulate_hormone_data(n_samples=1000)

# Split into train and test sets
X_train, X_test, y_train, y_test = create_train_test_split(X, y)
```

### Phase Classification

To use the classifier:

```python
from phase_classifier import MenstrualPhaseClassifier

# Define hormone ranges for the follicular phase
hormone_ranges = {
    'estrogen': (20, 200),
    'progesterone': (0.1, 1.5)
}

# Initialize and use the classifier
classifier = MenstrualPhaseClassifier(hormone_ranges=hormone_ranges)
predictions = classifier.predict(X_test)
```

## Customizing Hormone Ranges

You can customize the hormone ranges used for classification by modifying the `hormone_ranges` dictionary when initializing the classifier. The ranges should be specified as (min, max) tuples for the follicular phase.

## Data Simulation Parameters

The data simulation can be customized by providing hormone distribution parameters in the `hormone_params` dictionary. Each hormone should specify mean and standard deviation for both follicular and luteal phases. 