# Menstrual Cycle Phase Classification

A comprehensive machine learning system for classifying menstrual cycle phases using hormone measurements combined with a rule-based prior model. The system integrates traditional ML models with domain knowledge to achieve high accuracy in menstrual cycle phase prediction.

## 🏗️ Project Structure

```
rdoc_menstrual_classifier/
│
├── README.md                   # Project overview and instructions
├── requirements.txt            # Package dependencies
├── config/
│   ├── simulation_config.yaml   # Configuration for data simulation
│   ├── training_config.yaml     # Configuration for model training
│   ├── prediction_config.yaml   # Configuration for predictions
│   └── cross_validation_config.yaml # Configuration for cross-validation
│
├── src/
│   ├── __init__.py             # Make src a package
│   ├── main/
│   │   ├── simulation.py       # Generate synthetic hormone and period data
│   │   ├── train_model.py      # Train ML models with/without prior features
│   │   ├── predict_model.py    # Make predictions on unlabeled data
│   │   ├── cross_validation.py # Evaluate models with cross-validation
│   │   ├── compare_prior_weights.py # Compare different prior weights
│   │   └── classification.py   # Classifier implementations
│   │
│   ├── utils/
│   │   ├── __init__.py         # Make utils a package
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   ├── evaluator.py        # Model evaluation and metrics
│   │   └── model_utils.py      # Model saving/loading utilities
│   │
│   ├── classification/
│   │   ├── __init__.py         # Make classification a package
│   │   └── rule_based_prior.py # Rule-based prior model implementation
│   │
│   └── visualizations/
│       ├── __init__.py         # Make visualizations a package
│       ├── plotter.py          # Plotting utilities
│       └── report_generator.py # Report generation utilities
│
├── debug/                      # Debugging scripts and outputs
│   ├── debug_prior.py          # Analyze prior performance
│   ├── debug_prior_predictions.py # Debug prior predictions
│   ├── debug_combination_logic.py # Test combination logic
│   ├── debug_accuracy_calculation.py # Debug accuracy calculations
│   ├── debug_cv_data_alignment.py # Debug cross-validation data alignment
│   └── debug_outputs/          # Debug script outputs
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

## 🚀 Features

- **Hybrid ML + Rule-Based System**: Combines traditional ML models with domain-specific rule-based prior
- **Multiple Classifiers**: Support for Random Forest, Logistic Regression, SVM, XGBoost, and LightGBM
- **Rule-Based Prior**: Temporal model using survey responses and period data for perimenstruation detection
- **Prior Integration**: Models can be trained with prior features or used in ensemble predictions
- **Cross-Validation**: Comprehensive model evaluation using GroupKFold (keeps subjects together)
- **Feature Engineering**: Automatic generation of hormone ratios and interaction features
- **Organized Outputs**: Model-specific directories for predictions, confusion matrices, and reports
- **Configuration Management**: YAML-based configuration for easy parameter tuning
- **Debug Tools**: Comprehensive debugging scripts for troubleshooting

## 📦 Installation

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

## 🎯 Quick Start

### Complete Pipeline

To run the entire pipeline from simulation to predictions:

```bash
# 1. Generate synthetic data
python src/main/simulation.py

# 2. Train classification models (with and without prior features)
python src/main/train_model.py

# 3. Make predictions on unlabeled data
python src/main/predict_model.py

# 4. Run cross-validation evaluation
python src/main/cross_validation.py
```

## 📊 Script Functions and Outputs

### `simulation.py`
- **Purpose**: Generates synthetic hormone and period data for testing
- **Outputs**: 
  - `outputs/data/full_hormone_data_labeled.csv` (labeled training data)
  - `outputs/data/hormone_data_unlabeled.csv` (unlabeled prediction data)
  - `outputs/data/period_sleep_data.csv` (period tracking data)
  - `outputs/data/survey_responses.csv` (survey responses)

### `train_model.py`
- **Purpose**: Trains all ML models with and without prior features
- **Features**: 
  - Trains models with prior features integrated as additional features
  - Trains models without prior features for comparison
  - Generates confusion matrices for test sets
- **Outputs**:
  - `outputs/models/*_bundle.joblib` (trained model bundles)
  - `outputs/models/*_with_prior_bundle.joblib` (models with prior features)
  - `outputs/models/*_no_prior_bundle.joblib` (models without prior features)
  - `outputs/reports/<model_name>/` (model-specific evaluation results)
  - `outputs/reports/training_results.csv` (overall training summary)

### `predict_model.py`
- **Purpose**: Makes predictions on unlabeled data using all trained models
- **Features**:
  - Processes all models in `outputs/models/` directory
  - Generates confusion matrices by matching with labeled data
  - Handles XGBoost numeric predictions automatically
- **Outputs**:
  - `outputs/predictions/<model_name>/` (model-specific predictions)
    - `predictions.csv` - Original data + predicted phases + confidence
    - `prediction_probabilities.csv` - Raw probability scores
    - `prediction_summary.json` - Summary statistics
    - `prediction_distribution.png` - Bar chart of predictions
    - `confidence_distribution.png` - Confidence score histogram
    - `confusion_matrix.png` - Confusion matrix plot
    - `confusion_matrix_metrics.json` - Accuracy, precision, recall, F1
  - `outputs/predictions/combined_prediction_summary.json` (aggregated results)

### `cross_validation.py`
- **Purpose**: Evaluates model performance using cross-validation with full labeled dataset
- **Features**:
  - Uses GroupKFold to keep subjects together
  - Evaluates ML-only, prior-only, and combined predictions
  - Uses full labeled dataset
- **Outputs**:
  - `outputs/reports/<model_name>/` (model-specific results)
    - `confusion_matrix_cv.png` - ML-only confusion matrix
    - `combined_confusion_matrix.png` - ML + prior combined confusion matrix
    - `cv_results.json` - Cross-validation metrics
  - `outputs/reports/prior/prior_confusion_matrix.png` - Prior-only confusion matrix
  - `outputs/reports/deployment_prior_predictions.csv` - Prior predictions on unlabeled data
  - `outputs/reports/prior_testing_results.json` - Prior testing summary

### `compare_prior_weights.py`
- **Purpose**: Compares model performance across different prior weights
- **Features**: Tests different prior weights (0.0 to 1.0) to find optimal combination
- **Outputs**:
  - `outputs/reports/prior_weight_comparison/` (comparison plots and results)

## 🔧 Configuration

The project uses YAML configuration files for easy parameter tuning:

### Cross-Validation Configuration (`config/cross_validation_config.yaml`)

Key options for prior testing:

```yaml
prior_testing:
  enabled: true        # Enable prior testing
  prior_weight: 0.7   # Weight for prior in combined predictions (0.0-1.0)
```

### Training Configuration (`config/training_config.yaml`)
- Model hyperparameters for all classifiers
- Feature engineering options
- Prior feature integration settings
- Output preferences

### Prediction Configuration (`config/prediction_config.yaml`)
- Model selection for prediction
- Prior weight settings
- Output format preferences

### Simulation Configuration (`config/simulation_config.yaml`)
- Number of subjects and samples
- Hormone distribution parameters
- Phase duration settings
- Output settings

## 🧠 Classification Models

The project supports multiple classification approaches:

1. **Random Forest**: Ensemble method with feature importance analysis
2. **Logistic Regression**: Linear model with interpretable coefficients
3. **Support Vector Machine**: Non-linear classification with kernel methods
4. **XGBoost**: Gradient boosting with advanced features
5. **LightGBM**: Light gradient boosting machine
6. **Rule-Based Prior**: Temporal model using survey and period data

### Prior Model Features

The rule-based prior takes into account:
- **Survey Data**: Date of last period, cycle length, menstrual regularity
- **Period Data**: Actual period dates for precise perimenstruation detection
- **Hormone Patterns**: Estradiol, progesterone, and testosterone levels (not currently enabled - commented out)
- **Perfect Prior Logic**: 100% weight to perimenstruation when period data indicates menstruation

## 📁 Output Organization

The project generates organized outputs with model-specific directories:

### Models (`outputs/models/`)
- `*_bundle.joblib`: Complete model bundles (model + preprocessor + config)
- `*_with_prior_bundle.joblib`: Models trained with prior features
- `*_no_prior_bundle.joblib`: Models trained without prior features

### Predictions (`outputs/predictions/<model_name>/`)
- `predictions.csv`: Original data + predicted phases + confidence scores
- `prediction_probabilities.csv`: Raw probability scores for each class
- `prediction_summary.json`: Model-specific prediction summary
- `prediction_distribution.png`: Bar chart of prediction counts per phase
- `confidence_distribution.png`: Histogram of confidence scores
- `confusion_matrix.png`: Confusion matrix plot (by matching with labeled data)
- `confusion_matrix_metrics.json`: Accuracy, precision, recall, F1 scores

### Reports (`outputs/reports/<model_name>/`)
- `training_results.json`: Training metrics and metadata
- `confusion_matrix.png`: Test set confusion matrix
- `feature_importance.png`: Feature importance plot (if available)
- `confusion_matrix_cv.png`: Cross-validation ML-only confusion matrix
- `combined_confusion_matrix.png`: Cross-validation ML + prior confusion matrix

### Debug Outputs (`debug/debug_outputs/`)
- Temporary data files and analysis results from debugging scripts

## 🐛 Debugging

The project includes comprehensive debugging tools:

### Debug Scripts (`debug/`)
- `debug_prior.py`: Analyze prior performance and generate confusion matrices
- `debug_prior_predictions.py`: Debug prior predictions and combination logic
- `debug_combination_logic.py`: Test probability-based combination logic
- `debug_accuracy_calculation.py`: Debug accuracy calculation and label formats
- `debug_cv_data_alignment.py`: Debug cross-validation data alignment

### Running Debug Scripts
```bash
# Analyze prior performance
python debug/debug_prior.py

# Debug combination logic
python debug/debug_combination_logic.py

# Check data alignment
python debug/debug_cv_data_alignment.py
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_simulation.py
python -m pytest tests/test_classification.py
python -m pytest tests/test_cross_validation.py
```

## 🔧 Customization

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

## 📦 Dependencies

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

[Add your license information here]

## 📚 Citation

If you use this project in your research, please cite:

[Add citation information here] 