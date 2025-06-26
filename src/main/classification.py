#!/usr/bin/env python3
"""
Main classification script for menstrual pattern classification.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.simulation.temporal_model import TemporalModel
from src.utils.data_loader import load_config, load_and_split_data
from src.utils.preprocessor import DataProcessor
from src.utils.evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        from sklearn.ensemble import RandomForestClassifier as RF
        rf_config = config['models']['random_forest']
        self.model = RF(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            random_state=rf_config['random_state']
        )
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, path: str) -> None:
        """Save the trained model and feature names."""
        import joblib
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model and feature names."""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        from sklearn.linear_model import LogisticRegression
        lr_config = config['models']['logistic_regression']
        self.model = LogisticRegression(
            C=lr_config['C'],
            max_iter=lr_config['max_iter'],
            random_state=lr_config['random_state'],
            solver=lr_config.get('solver', 'lbfgs'),
            tol=lr_config.get('tol', 1e-6)
        )
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Logistic Regression model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (coefficients)."""
        if not hasattr(self.model, 'coef_'):
            return {}
        return dict(zip(self.feature_names, np.abs(self.model.coef_[0])))


class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        from sklearn.svm import SVC
        svm_config = config['models']['support_vector_machine']
        self.model = SVC(
            C=svm_config['C'],
            kernel=svm_config['kernel'],
            gamma=svm_config['gamma'],
            random_state=svm_config['random_state'],
            probability=True  # Enable probability estimates
        )
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the SVM model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (not available for SVM)."""
        return {}


class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier implementation with label encoding."""
    
    def __init__(self, config: Dict[str, Any]):
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        xgb_config = config['models']['xgboost']
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            random_state=xgb_config['random_state'],
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model with label encoding."""
        self.feature_names = X.columns
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model and decode labels."""
        y_pred_encoded = self.model.predict(X)
        if self.is_fitted:
            return self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred_encoded
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save_model(self, path: str) -> None:
        """Save the trained model, feature names, and label encoder."""
        import joblib
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, path)

    def load_model(self, path: str) -> None:
        """Load a trained model, feature names, and label encoder."""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.label_encoder = model_data.get('label_encoder', None)
        self.is_fitted = self.label_encoder is not None


class LightGBMClassifier(BaseClassifier):
    """LightGBM classifier implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        import lightgbm as lgb
        lgb_config = config['models']['lightgbm']
        self.model = lgb.LGBMClassifier(
            n_estimators=lgb_config['n_estimators'],
            max_depth=lgb_config['max_depth'],
            learning_rate=lgb_config['learning_rate'],
            subsample=lgb_config['subsample'],
            colsample_bytree=lgb_config['colsample_bytree'],
            random_state=lgb_config['random_state'],
            verbose=-1
        )
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LightGBM model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))


class TemporalClassifier(BaseClassifier):
    """Temporal model classifier implementation."""
    
    def __init__(self, config: Dict[str, Any], sequence_length: int = 70):
        self.model = TemporalModel(sequence_length=sequence_length)
        self.feature_names = None
        self.output_dir = os.path.join(config['output']['results_dir'], 'temporal')
        os.makedirs(self.output_dir, exist_ok=True)
        self.period_df = None
        self.survey_df = None
        self.config = config
    
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Train the temporal model."""
        # For temporal model, X should be the labeled_df
        if y is not None:
            logger.warning("Temporal model ignores target variable y, using labeled data directly")
        
        # Load period and survey data
        data_config = self.config['data']
        self.period_df = pd.read_csv(data_config['period_data_path'])
        self.survey_df = pd.read_csv(data_config['survey_data_path'])
        
        # Convert dates to datetime
        X['date'] = pd.to_datetime(X['date'])
        self.period_df['date'] = pd.to_datetime(self.period_df['date'])
        
        # Train model on full sequences with cycle structure
        logger.info("Training temporal model...")
        self.model.train(X, self.period_df, self.survey_df)
        
        # Save trained model
        self.model.save_model(os.path.join(self.output_dir, 'model.pt'))
        logger.info(f"Saved trained model to {self.output_dir}/model.pt")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the temporal model."""
        # Create a DataFrame with the required format for temporal model
        predict_df = pd.DataFrame({
            'subject_id': X.index.get_level_values('subject_id') if isinstance(X.index, pd.MultiIndex) else range(len(X)),
            'date': X.index.get_level_values('date') if isinstance(X.index, pd.MultiIndex) else pd.date_range(start='2024-01-01', periods=len(X)),
            'estradiol': X['estradiol'] if 'estradiol' in X.columns else X.iloc[:, 0],
            'progesterone': X['progesterone'] if 'progesterone' in X.columns else X.iloc[:, 1],
            'testosterone': X['testosterone'] if 'testosterone' in X.columns else X.iloc[:, 2]
        })
        
        # Convert dates to datetime
        predict_df['date'] = pd.to_datetime(predict_df['date'])
        
        # Load period and survey data if not already loaded
        data_config = self.config['data']
        if self.period_df is None:
            self.period_df = pd.read_csv(data_config['period_data_path'])
            self.period_df['date'] = pd.to_datetime(self.period_df['date'])
        if self.survey_df is None:
            self.survey_df = pd.read_csv(data_config['survey_data_path'])
        
        # Add period information to predict_df
        for subject_id in predict_df['subject_id'].unique():
            subject_periods = self.period_df[
                (self.period_df['subject_id'] == subject_id) & 
                (self.period_df['period'] == 1)
            ]
            if not subject_periods.empty:
                # Add period column to predict_df for this subject
                subject_mask = predict_df['subject_id'] == subject_id
                predict_df.loc[subject_mask, 'period'] = 0  # Default to 0
                for _, period_row in subject_periods.iterrows():
                    period_date = period_row['date']
                    predict_df.loc[
                        (subject_mask) & (predict_df['date'] == period_date),
                        'period'
                    ] = 1
        
        # Get predictions for each subject
        predictions = []
        
        for subject_id in predict_df['subject_id'].unique():
            subject_data = predict_df[predict_df['subject_id'] == subject_id].copy()
            
            # Get predictions for the sequence
            cycle_positions = self.model.predict_cycle_position(subject_data)
            
            # Handle both scalar and array predictions
            if np.isscalar(cycle_positions):
                cycle_positions = [cycle_positions]
            
            # Convert cycle positions to phases
            for pos in cycle_positions:
                # Map cycle day to phase based on hormone_config.py ranges
                if 1 <= pos <= 14:  # follicular phase (days 1-14)
                    predictions.append('follicular')
                else:  # luteal phase (days 15-28)
                    predictions.append('luteal')
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (not available for temporal model)."""
        return {}


def run_classification(config: Dict[str, Any]):
    """Run the classification pipeline."""
    logger.info("Starting classification pipeline...")
    
    # Load and split data using the new function
    data_config = config['data']
    hormone_data_path = data_config['hormone_data_path']
    
    logger.info(f"Loading data from: {hormone_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(
        hormone_data_path, 
        test_size=data_config['test_size'], 
        random_state=data_config['random_state']
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config['output']['results_dir'])
    
    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(config),
        'Logistic Regression': LogisticRegressionClassifier(config),
        'SVM': SVMClassifier(config),
        'XGBoost': XGBoostClassifier(config),
        'LightGBM': LightGBMClassifier(config)
    }
    
    # Train and evaluate each classifier
    results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"\nTraining {name}...")
        
        try:
            # Train the model
            classifier.train(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            # Evaluate the model
            metrics = evaluator.evaluate(y_test, y_pred, name, X_test)
            results[name] = metrics
            
            # Save model if configured
            if config['output']['save_model']:
                model_path = os.path.join(config['output']['results_dir'], f'{name.lower().replace(" ", "_")}_model.pkl')
                if hasattr(classifier, 'save_model'):
                    classifier.save_model(model_path)
            
            # Generate feature importance plot if available
            if config['output']['generate_feature_importance']:
                feature_importance = classifier.get_feature_importance()
                if feature_importance:
                    evaluator.plot_feature_importance(feature_importance, name)
            
            # Generate confusion matrix
            if config['output']['generate_confusion_matrix']:
                evaluator._plot_confusion_matrix(y_test, y_pred, name)
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(config['output']['results_dir'], 'classification_results.csv'))
    
    logger.info("\nClassification pipeline completed!")
    return results


def main():
    """Main function to run the classification."""
    # Load configuration
    config = load_config('config/classification_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['figures_dir'], exist_ok=True)
    
    # Run classification
    results = run_classification(config)
    
    # Print summary
    print("\n=== CLASSIFICATION RESULTS SUMMARY ===")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print()


if __name__ == '__main__':
    main() 