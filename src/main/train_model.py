#!/usr/bin/env python3
"""
Training script for menstrual pattern classification models.
Trains models on labeled data and saves them for later use.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config, preprocess_data_with_prior, preprocess_data, DataPreprocessor
from src.utils.evaluator import ModelEvaluator
from src.utils.model_utils import save_model_bundle
from src.temporal_models.rule_based_prior import RuleBasedPrior
from src.main.classification import (
    RandomForestClassifier,
    LogisticRegressionClassifier,
    SVMClassifier,
    XGBoostClassifier,
    LightGBMClassifier
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_results_dir(results_dir: str, model_name: str) -> str:
    """
    Get model-specific results directory.
    
    Args:
        results_dir (str): Base results directory
        model_name (str): Name of the model
        
    Returns:
        str: Path to model-specific results directory
    """
    # Clean model name for directory name
    clean_name = model_name.lower().replace(" ", "_").replace("-", "_")
    model_results_dir = os.path.join(results_dir, clean_name)
    os.makedirs(model_results_dir, exist_ok=True)
    return model_results_dir


def train_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train classification models on labeled data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Training results and model information
    """
    logger.info("Starting model training...")
    
    # Load data
    data_config = config['data']
    labeled_data_path = data_config['labeled_data_path']
    
    logger.info(f"Loading labeled data from: {labeled_data_path}")
    
    # Load raw data
    data = pd.read_csv(labeled_data_path)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Preprocess data with prior features if enabled
    if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
        logger.info("Using rule-based prior as features...")
        X, y = preprocess_data_with_prior(data, config, preprocessor)
    else:
        logger.info("Using standard preprocessing without prior...")
        X, y = preprocess_data(data, preprocessor)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=data_config['test_size'], 
        random_state=data_config['random_state']
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Number of features: {len(X_train.columns)}")
    
    # Save the fitted preprocessor
    preprocessor_path = os.path.join(config['output']['models_dir'], 'data_preprocessor.joblib')
    preprocessor.save(preprocessor_path)
    logger.info(f"Saved fitted preprocessor to: {preprocessor_path}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config['output']['results_dir'])
    
    # Initialize classifiers
    classifiers = {
        'random_forest': RandomForestClassifier(config),
        'logistic_regression': LogisticRegressionClassifier(config),
        'svm': SVMClassifier(config),
        'xgb': XGBoostClassifier(config),
        'lgbm': LightGBMClassifier(config)
    }
    
    # Train and evaluate each classifier
    training_results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"\nTraining {name}...")
        
        try:
            # Train the model
            classifier.train(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            # Evaluate the model
            metrics = evaluator.evaluate(y_test, y_pred, name, X_test)
            training_results[name] = metrics
            
            # Get model-specific results directory
            model_results_dir = get_model_results_dir(config['output']['results_dir'], name)
            
            # Save model-specific training results
            model_results = {
                'model_name': name,
                'training_metrics': metrics,
                'feature_names': list(X_train.columns),
                'n_features': len(X_train.columns),
                'n_classes': len(np.unique(y_train)),
                'classes': list(np.unique(y_train)),
                'training_date': pd.Timestamp.now().isoformat(),
                'used_prior_features': config.get('models', {}).get('temporal', {}).get('use_as_prior', False),
                'test_size': len(X_test),
                'train_size': len(X_train)
            }
            
            # Save model-specific results
            model_results_path = os.path.join(model_results_dir, 'training_results.json')
            with open(model_results_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
            
            # Save model with metadata
            metadata = {
                'training_metrics': metrics,
                'feature_names': list(X_train.columns),
                'n_features': len(X_train.columns),
                'n_classes': len(np.unique(y_train)),
                'classes': list(np.unique(y_train)),
                'training_date': pd.Timestamp.now().isoformat(),
                'config': config,
                'used_prior_features': config.get('models', {}).get('temporal', {}).get('use_as_prior', False),
                'preprocessor_path': preprocessor_path
            }
            
            model_path = save_model_bundle(
                model=classifier.model,
                preprocessor=preprocessor,
                config=config,
                training_metrics=metrics,
                feature_names=list(X_train.columns),
                model_name=name,
                output_dir=config['output']['models_dir'],
                label_encoder=getattr(classifier, 'label_encoder', None)
            )
            
            # Generate feature importance plot if available
            if config['output']['generate_feature_importance']:
                feature_importance = classifier.get_feature_importance()
                if feature_importance:
                    evaluator.plot_feature_importance(feature_importance, name)
            
            # Generate confusion matrix
            if config['output']['generate_confusion_matrix']:
                evaluator._plot_confusion_matrix(y_test, y_pred, name)
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Model results saved to: {model_results_dir}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            continue
    
    # Save training results summary
    if training_results:
        results_df = pd.DataFrame(training_results).T
        results_path = os.path.join(config['output']['results_dir'], 'training_results.csv')
        results_df.to_csv(results_path)
        logger.info(f"Training results saved to: {results_path}")
    
    logger.info("\nModel training completed!")
    return training_results


def main():
    """Main function to run model training."""
    # Load configuration
    config = load_config('config/training_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['figures_dir'], exist_ok=True)
    
    # Train models
    results = train_models(config)
    
    # Print summary
    print("\n=== TRAINING RESULTS SUMMARY ===")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print()
    
    print(f"Models saved to: {config['output']['models_dir']}")
    print(f"Results saved to: {config['output']['results_dir']}")


if __name__ == '__main__':
    main() 