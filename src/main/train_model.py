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
from src.classification.rule_based_prior import RuleBasedPrior
from src.classification.classification import (
    RandomForestClassifier,
    LogisticRegressionClassifier,
    SVMClassifier,
    XGBoostClassifier,
    LightGBMClassifier
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_predictions_with_perfect_prior(ml_predictions: np.ndarray, prior_predictions: np.ndarray, 
                                         ml_probs: np.ndarray, prior_probs: np.ndarray, 
                                         prior_weight: float = 0.3) -> np.ndarray:
    """
    Combine ML and prior predictions using smart weighted voting.
    
    The prior gets 100% weight for phases it's known to be perfect at (perimenstruation),
    and the configured weight for other phases.
    
    Args:
        ml_predictions: Array of ML predictions
        prior_predictions: Array of prior predictions
        ml_probs: ML probability vectors (n_samples, n_classes)
        prior_probs: Prior probability vectors (n_samples, n_classes)
        prior_weight: Weight for prior predictions (0.0 to 1.0) for non-perfect phases
        
    Returns:
        np.ndarray: Combined predictions
    """
    # Define phase order
    phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
    
    # Phases where prior has perfect accuracy (should get 100% weight)
    perfect_prior_phases = ['perimenstruation']
    
    combined_predictions = []
    
    for i in range(len(ml_predictions)):
        ml_pred = ml_predictions[i]
        prior_pred = prior_predictions[i]
        
        # If prior predicts a phase it's perfect at, use prior prediction
        if prior_pred in perfect_prior_phases:
            combined_predictions.append(prior_pred)
        # If ML predicts a phase the prior is perfect at, use prior prediction
        elif ml_pred in perfect_prior_phases:
            combined_predictions.append(prior_pred)
        # Otherwise, use weighted combination of probabilities
        else:
            # Combine probability vectors
            combined_probs = (1 - prior_weight) * ml_probs[i] + prior_weight * prior_probs[i]
            
            # Get final prediction
            combined_pred = phases[np.argmax(combined_probs)]
            combined_predictions.append(combined_pred)
    
    return np.array(combined_predictions)


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


def train_single_model(classifier, X_train, y_train, X_test, y_test, name, config, preprocessor, evaluator, model_results_dir, use_prior_features: bool, original_data: pd.DataFrame = None, test_indices: np.ndarray = None) -> Tuple[Dict[str, Any], str]:
    """
    Train a single model and return results and model path.
    
    Args:
        classifier: The classifier to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        name: Model name
        config: Configuration dictionary
        preprocessor: Fitted preprocessor
        evaluator: Model evaluator
        model_results_dir: Directory to save results
        use_prior_features: Whether prior features were used
        original_data: Original unprocessed data (needed for prior predictions)
        test_indices: Indices of test samples in original data
        
    Returns:
        Tuple of (metrics, model_path)
    """
    logger.info(f"Training {name} {'with' if use_prior_features else 'without'} prior features...")
    
    try:
        # Train the model
        classifier.train(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # If using prior features, also get ensemble predictions
        if use_prior_features and original_data is not None and test_indices is not None:
            # Get ML probabilities
            ml_probs = classifier.predict_proba(X_test)
            
            # Get prior predictions and probabilities using original data
            rule_prior = RuleBasedPrior(config)
            rule_prior.load_data()
            
            # Get the correct test original data using indices
            test_original_data = original_data.iloc[test_indices].reset_index(drop=True)
            
            prior_predictions = rule_prior.predict_phases(test_original_data)
            prior_probs = rule_prior.get_prior_probabilities(test_original_data)
            
            # Combine predictions using perfect prior logic
            prior_weight = config.get('models', {}).get('temporal', {}).get('prior_weight', 0.3)
            y_pred_combined = combine_predictions_with_perfect_prior(
                y_pred, prior_predictions, ml_probs, prior_probs, prior_weight
            )
            
            # Use combined predictions for evaluation
            y_pred = y_pred_combined
        
        # Evaluate the model
        metrics = evaluator.evaluate(y_test, y_pred, name, X_test)
        
        # Save model-specific training results
        model_results = {
            'model_name': name,
            'training_metrics': metrics,
            'feature_names': list(X_train.columns),
            'n_features': len(X_train.columns),
            'n_classes': len(np.unique(y_train)),
            'classes': list(np.unique(y_train)),
            'training_date': pd.Timestamp.now().isoformat(),
            'used_prior_features': use_prior_features,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        # Save model-specific results with prior indicator
        prior_suffix = "_with_prior" if use_prior_features else "_no_prior"
        model_results_path = os.path.join(model_results_dir, f'training_results{prior_suffix}.json')
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
            'used_prior_features': use_prior_features,
            'preprocessor_path': os.path.join(config['output']['models_dir'], 'data_preprocessor.joblib')
        }
        
        # Save model with prior indicator in name
        model_name_with_prior = f"{name}{prior_suffix}"
        model_path = save_model_bundle(
            model=classifier.model,
            preprocessor=preprocessor,
            config=config,
            training_metrics=metrics,
            feature_names=list(X_train.columns),
            model_name=model_name_with_prior,
            output_dir=config['output']['models_dir'],
            label_encoder=getattr(classifier, 'label_encoder', None)
        )
        
        # Generate feature importance plot if available
        if config['output']['generate_feature_importance']:
            feature_importance = classifier.get_feature_importance()
            if feature_importance:
                evaluator.plot_feature_importance(feature_importance, model_name_with_prior)
        
        # Generate confusion matrix
        if config['output']['generate_confusion_matrix']:
            evaluator._plot_confusion_matrix(y_test, y_pred, model_name_with_prior)
        
        logger.info(f"{model_name_with_prior} - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Model saved to: {model_path}")
        
        return metrics, model_path
        
    except Exception as e:
        logger.error(f"Error training {name}: {str(e)}")
        return None, None


def compare_prior_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train models with and without prior features and compare performance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Comparison results
    """
    logger.info("Starting prior feature comparison training...")
    
    # Load data
    data_config = config['data']
    labeled_data_path = data_config['labeled_data_path']
    
    logger.info(f"Loading labeled data from: {labeled_data_path}")
    
    # Load raw data
    data = pd.read_csv(labeled_data_path)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
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
    
    comparison_results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"\n=== Training {name} with and without prior features ===")
        
        comparison_results[name] = {}
        
        # Train with prior features
        logger.info(f"Training {name} WITH prior features...")
        X_with_prior, y_with_prior = preprocess_data_with_prior(data, config, preprocessor)
        
        # Create indices for tracking
        n_samples = len(X_with_prior)
        indices = np.arange(n_samples)
        X_train_prior, X_test_prior, y_train_prior, y_test_prior, train_indices_prior, test_indices_prior = train_test_split(
            X_with_prior, y_with_prior, indices,
            test_size=data_config['test_size'], 
            random_state=data_config['random_state']
        )
        
        # Create fresh classifier instance for prior training
        classifier_with_prior = type(classifier)(config)
        model_results_dir = get_model_results_dir(config['output']['results_dir'], name)
        
        metrics_with_prior, model_path_with_prior = train_single_model(
            classifier_with_prior, X_train_prior, y_train_prior, X_test_prior, y_test_prior,
            name, config, preprocessor, evaluator, model_results_dir, use_prior_features=True, 
            original_data=data, test_indices=test_indices_prior
        )
        
        if metrics_with_prior:
            comparison_results[name]['with_prior'] = {
                'metrics': metrics_with_prior,
                'model_path': model_path_with_prior,
                'n_features': len(X_train_prior.columns)
            }
        
        # Train without prior features
        logger.info(f"Training {name} WITHOUT prior features...")
        X_no_prior, y_no_prior = preprocess_data(data, preprocessor)
        
        # Create indices for tracking
        n_samples = len(X_no_prior)
        indices = np.arange(n_samples)
        X_train_no_prior, X_test_no_prior, y_train_no_prior, y_test_no_prior, train_indices_no_prior, test_indices_no_prior = train_test_split(
            X_no_prior, y_no_prior, indices,
            test_size=data_config['test_size'], 
            random_state=data_config['random_state']
        )
        
        # Create fresh classifier instance for no-prior training
        classifier_no_prior = type(classifier)(config)
        
        metrics_no_prior, model_path_no_prior = train_single_model(
            classifier_no_prior, X_train_no_prior, y_train_no_prior, X_test_no_prior, y_test_no_prior,
            name, config, preprocessor, evaluator, model_results_dir, use_prior_features=False, 
            original_data=data, test_indices=test_indices_no_prior
        )
        
        if metrics_no_prior:
            comparison_results[name]['no_prior'] = {
                'metrics': metrics_no_prior,
                'model_path': model_path_no_prior,
                'n_features': len(X_train_no_prior.columns)
            }
    
    # Save comparison results
    comparison_path = os.path.join(config['output']['results_dir'], 'prior_comparison_results.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"Prior comparison results saved to: {comparison_path}")
    
    return comparison_results


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
    n_samples = len(X)
    indices = np.arange(n_samples)
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices,
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
            
            # If using prior features, also get ensemble predictions
            if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
                # Get ML probabilities
                ml_probs = classifier.predict_proba(X_test)
                
                # Get prior predictions and probabilities using original data
                rule_prior = RuleBasedPrior(config)
                rule_prior.load_data()
                
                # Get the correct test original data using indices
                test_original_data = data.iloc[test_indices].reset_index(drop=True)
                
                prior_predictions = rule_prior.predict_phases(test_original_data)
                prior_probs = rule_prior.get_prior_probabilities(test_original_data)
                
                # Combine predictions using perfect prior logic
                prior_weight = config.get('models', {}).get('temporal', {}).get('prior_weight', 0.3)
                y_pred_combined = combine_predictions_with_perfect_prior(
                    y_pred, prior_predictions, ml_probs, prior_probs, prior_weight
                )
                
                # Use combined predictions for evaluation
                y_pred = y_pred_combined
            
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
    
    # Check if comparison mode is enabled
    if config.get('training', {}).get('compare_prior_features', False):
        logger.info("Running prior feature comparison training...")
        comparison_results = compare_prior_training(config)
        
        # Print comparison summary
        print("\n=== PRIOR FEATURE COMPARISON SUMMARY ===")
        for model_name, results in comparison_results.items():
            print(f"\n{model_name.upper()}:")
            
            if 'with_prior' in results and 'no_prior' in results:
                with_prior_acc = results['with_prior']['metrics']['accuracy']
                no_prior_acc = results['no_prior']['metrics']['accuracy']
                improvement = with_prior_acc - no_prior_acc
                
                print(f"  With prior features:    {with_prior_acc:.4f} ({results['with_prior']['n_features']} features)")
                print(f"  Without prior features: {no_prior_acc:.4f} ({results['no_prior']['n_features']} features)")
                print(f"  Improvement:            {improvement:+.4f}")
                
                if improvement > 0:
                    print(f"  ✅ Prior features improve performance")
                elif improvement < 0:
                    print(f"  ❌ Prior features hurt performance")
                else:
                    print(f"  ➖ No difference in performance")
            else:
                print(f"  ⚠️  Incomplete results for comparison")
        
        print(f"\nModels saved to: {config['output']['models_dir']}")
        print(f"Comparison results saved to: {config['output']['results_dir']}/prior_comparison_results.json")
        
    else:
        # Run standard training
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