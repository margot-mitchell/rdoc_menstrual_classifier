# Menstrual Cycle Classifier

A comprehensive machine learning project for classifying menstrual cycle patterns based on hormone measurements and simulating realistic menstrual cycle data.

## Project Structure

```
rdoc_menstrual_classifier/
│
├── README.md                   # Project overview and instructions
├── requirements.txt            # Package dependencies
├── config/
│   ├── simulation_config.yaml   # Configuration for simulation
│   ├── training_config.yaml     # Configuration for model training
│   ├── prediction_config.yaml   # Configuration for predictions
│   └── cross_validation_config.yaml # Configuration for cross-validation
│
├── src/
│   ├── __init__.py             # Make src a package
│   ├── main/
│   │   ├── simulation.py       # Main simulation script
│   │   ├── train_model.py      # Model training script
│   │   ├── predict_model.py    # Prediction script
│   │   └── cross_validation.py   # Main cross-validation script
│   │
│   ├── utils/
│   │   ├── __init__.py         # Make utils a package
│   │   ├── data_loader.py      # Module for loading data
│   │   ├── evaluator.py         # Module for model evaluation metrics
│   │   └── model_utils.py       # Module for model saving/loading
│   │
│   ├── temporal_models/
│   │   ├── __init__.py         # Make temporal_models a package
│   │   ├── rule_based_prior.py # Rule-based prior model
│   │   └── temporal_predict.py  # Temporal prediction script
│   │
│   └── visualizations/
│       ├── __init__.py         # Make visualizations a package
│       ├── plotter.py          # Module for plotting
│       └── report_generator.py  # Module for generating reports
│
├── tests/
│   ├── __init__.py             # Make tests a package
│   ├── test_simulation.py      # Tests for simulation functionality
│   ├── test_classification.py   # Tests for classification functionality
│   └── test_cross_validation.py  # Tests for cross-validation functionality
│
├── outputs/
│   ├── data/                   # Directory for storing data outputs
│   ├── models/                 # Directory for storing trained models
│   ├── predictions/            # Directory for storing predictions
│   ├── figures/                # Directory for storing figures
│   └── reports/                # Directory for storing reports
│
└── scripts/                    # Shell scripts for running the project
    ├── run_simulation.sh       # Shell script to run the simulation
    ├── run_cross_validation.sh   # Shell script to run the cross-validation
    └── run_tests.sh            # Shell script to run tests
```

## Features

- **Data Simulation**: Generate realistic hormone and period data based on scientific literature
- **Multiple Classifiers**: Support for Random Forest, Logistic Regression, SVM, and Temporal models
- **Cross-Validation**: Comprehensive model evaluation with learning and validation curves
- **Feature Engineering**: Automatic generation of interaction and ratio features
- **Visualization**: Comprehensive plotting and report generation
- **Configuration Management**: YAML-based configuration for easy parameter tuning

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rdoc_menstrual_classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using Shell Scripts (Recommended)

1. **Run Simulation**:
```bash
./scripts/run_simulation.sh
```

2. **Train Models**:
```bash
python src/main/train_model.py
```

3. **Make Predictions**:
```bash
python src/main/predict_model.py
```

4. **Run Cross-Validation**:
```bash
./scripts/run_cross_validation.sh
```

### Using Python Scripts Directly

1. **Simulation**:
```bash
python src/main/simulation.py
```

2. **Train Models**:
```bash
python src/main/train_model.py
```

3. **Make Predictions**:
```bash
python src/main/predict_model.py
```

4. **Cross-Validation**:
```bash
python src/main/cross_validation.py
```

### Complete Pipeline

To run the entire pipeline from simulation to predictions:

```bash
# 1. Generate synthetic data
python src/main/simulation.py

# 2. Train classification models
python src/main/train_model.py

# 3. Make predictions on unlabeled data
python src/main/predict_model.py
```

## Configuration

The project uses YAML configuration files for easy parameter tuning:

### Simulation Configuration (`config/simulation_config.yaml`)
- Number of subjects and samples
- Hormone distribution parameters
- Phase duration settings
- Output settings

### Classification Configuration (`config/classification_config.yaml`)
- Model hyperparameters
- Feature engineering options
- Cross-validation settings
- Output preferences

### Cross-Validation Configuration (`config/cross_validation_config.yaml`)
- CV folds and repeats
- Evaluation metrics
- Data splitting strategy

## Data Simulation

The simulation generates realistic menstrual cycle data based on scientific literature:

- **Hormone Data**: Estradiol, progesterone, and testosterone levels across cycle phases
- **Period Data**: Menstrual period tracking with realistic timing
- **Survey Data**: Self-reported cycle information
- **Pattern Classification**: Regular, irregular, and anovulatory patterns

## Classification Models

The project supports multiple classification approaches:

1. **Random Forest**: Ensemble method with feature importance analysis
2. **Logistic Regression**: Linear model with interpretable coefficients
3. **Support Vector Machine**: Non-linear classification with kernel methods
4. **Temporal Model**: Neural network for sequence-based classification

## Outputs

The project generates comprehensive outputs:

### Data Files (`outputs/data/`)
- `hormone_data_unlabeled.csv`: Hormone measurements for classification
- `survey_responses.csv`: Self-reported cycle information
- `period_sleep_data.csv`: Period tracking data
- `menstrual_patterns.csv`: Ground truth pattern labels

### Figures (`outputs/figures/`)
- Hormone cycle plots
- Model performance comparisons
- Learning and validation curves
- Feature importance visualizations

### Reports (`outputs/reports/`)
- HTML reports with comprehensive analysis
- Model evaluation metrics
- Cross-validation results
- Classification reports

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_simulation.py
python -m pytest tests/test_classification.py
python -m pytest tests/test_cross_validation.py
```

## Customization

### Adding New Models

1. Create a new classifier class inheriting from `BaseClassifier`
2. Implement the required methods: `train()`, `predict()`, `get_feature_importance()`
3. Add configuration parameters to the YAML files
4. Update the main classification script

### Adding New Features

1. Modify the `DataProcessor` class in `src/utils/preprocessor.py`
2. Add feature engineering logic
3. Update configuration files to include new feature options

### Custom Visualizations

1. Add new plotting methods to `DataPlotter` class
2. Update report generation in `ReportGenerator` class
3. Modify configuration to enable new visualizations

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **torch**: Deep learning (for temporal models)
- **pyyaml**: YAML configuration parsing
- **joblib**: Model persistence
- **scipy**: Scientific computing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this project in your research, please cite:

[Add citation information here] 