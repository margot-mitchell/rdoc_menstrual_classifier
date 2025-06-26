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
│   │   ├── simulation.py       # Generate synthetic hormone and period data
│   │   ├── train_model.py      # Train all ML models on labeled data
│   │   ├── predict_model.py    # Make predictions on unlabeled data
│   │   ├── cross_validation.py # Evaluate models with cross-validation
│   │   ├── temporal_predict.py # Rule-based predictions (no ML training)
│   │   ├── compare_prior_weights.py # Compare prior weight performance
│   │   └── classification.py   # Classifier implementations (library)
│   │
│   ├── utils/
│   │   ├── __init__.py         # Make utils a package
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   ├── evaluator.py        # Model evaluation and metrics
│   │   └── model_utils.py      # Model saving/loading utilities
│   │
│   ├── temporal_models/
│   │   ├── __init__.py         # Make temporal_models a package
│   │   └── rule_based_prior.py # Rule-based prior model implementation
│   │
│   └── visualizations/
│       ├── __init__.py         # Make visualizations a package
│       ├── plotter.py          # Plotting utilities
│       └── report_generator.py # Report generation utilities
│
├── tests/
│   ├── __init__.py             # Make tests a package
│   ├── test_simulation.py      # Tests for simulation functionality
│   ├── test_classification.py  # Tests for classification functionality
│   └── test_cross_validation.py # Tests for cross-validation functionality
│
├── outputs/
│   ├── data/                   # Generated data files
│   ├── models/                 # Trained model bundles
│   ├── predictions/            # Prediction results (organized by model)
│   ├── figures/                # Generated plots and visualizations
│   └── reports/                # Evaluation reports (organized by model)
│
└── scripts/                    # Shell scripts for running the project
    ├── run_simulation.sh       # Shell script to run the simulation
    ├── run_cross_validation.sh # Shell script to run the cross-validation
    └── run_tests.sh            # Shell script to run tests
```

## Features

- **Data Simulation**: Generate realistic hormone and period data based on scientific literature
- **Multiple Classifiers**: Support for Random Forest, Logistic Regression, SVM, XGBoost, and LightGBM
- **Rule-Based Prior**: Temporal model using survey responses and period data
- **Cross-Validation**: Comprehensive model evaluation with learning and validation curves
- **Feature Engineering**: Automatic generation of interaction and ratio features
- **Organized Outputs**: Model-specific directories for predictions and reports
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

### Complete Pipeline

To run the entire pipeline from simulation to predictions:

```bash
# 1. Generate synthetic data
python src/main/simulation.py

# 2. Train classification models
python src/main/train_model.py

# 3. Make predictions on unlabeled data
python src/main/predict_model.py

# 4. Run cross-validation (optional)
python src/main/cross_validation.py
```

### Individual Scripts

1. **Simulation** (Generate test data):
```bash
python src/main/simulation.py
```

2. **Training** (Train all models):
```bash
python src/main/train_model.py
```

3. **Prediction** (Make predictions):
```bash
python src/main/predict_model.py
```

4. **Cross-Validation** (Evaluate models):
```bash
python src/main/cross_validation.py
```

5. **Temporal Prediction** (Rule-based, no training):
```bash
python src/main/temporal_predict.py
```

6. **Prior Weight Comparison** (Optimize prior weights):
```bash
python src/main/compare_prior_weights.py
```

## Script Functions and Outputs

### `simulation.py`
- **Purpose**: Generates synthetic hormone and period data for testing
- **Outputs**: 
  - `outputs/data/full_hormone_data_labeled.csv` (labeled training data)
  - `outputs/data/hormone_data_unlabeled.csv` (unlabeled prediction data)
  - `outputs/data/period_sleep_data.csv` (period tracking data)
  - `outputs/data/survey_responses.csv` (survey responses)

### `train_model.py`
- **Purpose**: Trains all ML models (Random Forest, Logistic Regression, SVM, XGBoost, LightGBM)
- **Outputs**:
  - `outputs/models/*_bundle.joblib` (trained model bundles)
  - `outputs/reports/model_name/` (model-specific evaluation results)
  - `outputs/reports/training_results.csv` (overall training summary)

### `predict_model.py`
- **Purpose**: Makes predictions on unlabeled data using all trained models
- **Outputs**:
  - `outputs/predictions/model_name/` (model-specific predictions)
  - `outputs/reports/combined_prediction_summary.json` (aggregated results)

### `cross_validation.py`
- **Purpose**: Evaluates model performance using cross-validation
- **Outputs**:
  - `outputs/reports/model_name/cross_validation_results.json`
  - `outputs/reports/model_name/learning_curves.png`
  - `outputs/reports/model_name/validation_curves.png`

### `temporal_predict.py`
- **Purpose**: Makes rule-based predictions using survey responses and period data
- **Outputs**:
  - `outputs/predictions/rule_based_predictions.csv`
  - Rule-based prediction summaries

### `compare_prior_weights.py`
- **Purpose**: Compares model performance across different prior weights
- **Outputs**:
  - `outputs/reports/prior_weight_comparison/` (comparison plots and results)

## Configuration

The project uses YAML configuration files for easy parameter tuning:

### Simulation Configuration (`config/simulation_config.yaml`)
- Number of subjects and samples
- Hormone distribution parameters
- Phase duration settings
- Output settings

### Training Configuration (`config/training_config.yaml`)
- Model hyperparameters for all classifiers
- Feature engineering options
- Prior weight settings
- Output preferences

### Prediction Configuration (`config/prediction_config.yaml`)
- Model selection for prediction
- Prior weight settings
- Output format preferences

### Cross-Validation Configuration (`config/cross_validation_config.yaml`)
- CV folds and repeats
- Evaluation metrics
- Data splitting strategy

## Classification Models

The project supports multiple classification approaches:

1. **Random Forest**: Ensemble method with feature importance analysis
2. **Logistic Regression**: Linear model with interpretable coefficients
3. **Support Vector Machine**: Non-linear classification with kernel methods
4. **XGBoost**: Gradient boosting with advanced features
5. **LightGBM**: Light gradient boosting machine
6. **Rule-Based Prior**: Temporal model using survey and period data

## Output Organization

The project generates organized outputs with model-specific directories:

### Models (`outputs/models/`)
- `*_bundle.joblib`: Complete model bundles (model + preprocessor + config)

### Predictions (`outputs/predictions/model_name/`)
- `predictions.csv`: Predicted phases
- `probabilities.csv`: Prediction probabilities
- `summary.json`: Model-specific prediction summary

### Reports (`outputs/reports/model_name/`)
- `training_results.json`: Training metrics
- `cross_validation_results.json`: CV results
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.png`: Confusion matrix plot
- `feature_importance.png`: Feature importance plot
- `learning_curves.png`: Learning curves
- `validation_curves.png`: Validation curves

### Combined Results
- `outputs/reports/combined_prediction_summary.json`: Aggregated results from all models
- `outputs/reports/training_results.csv`: Overall training summary
- `outputs/reports/cross_validation_results.csv`: Overall CV summary

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

1. Modify the `DataPreprocessor` class in `src/utils/data_loader.py`
2. Add feature engineering logic
3. Update configuration files to include new feature options

### Custom Visualizations

1. Add new plotting methods to the visualization modules
2. Update report generation
3. Modify configuration to enable new visualizations

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **xgboost**: XGBoost gradient boosting
- **lightgbm**: LightGBM gradient boosting
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