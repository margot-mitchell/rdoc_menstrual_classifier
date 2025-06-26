#!/usr/bin/env python3
"""
Main cross-validation script for model evaluation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import (
    StratifiedKFold, 
    cross_val_score, 
    learning_curve, 
    validation_curve
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.utils.data_loader import load_config, load_and_split_data
from src.utils.preprocessor import DataProcessor
from src.utils.evaluator import ModelEvaluator
from src.main.classification import (
    RandomForestClassifier,
    LogisticRegressionClassifier,
    SVMClassifier,
    XGBoostClassifier,
    LightGBMClassifier
)
from src.temporal_models.rule_based_prior import RuleBasedPrior

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


def run_cross_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run cross-validation experiments."""
    logger.info("Starting cross-validation experiments...")
    
    # Load and split data using the new function
    data_config = config['data']
    hormone_data_path = data_config['hormone_data_path']
    
    logger.info(f"Loading data from: {hormone_data_path}")
    X_train, X_test, y_train, y_test = load_and_split_data(
        hormone_data_path, 
        test_size=data_config['test_size'], 
        random_state=data_config['random_state']
    )
    
    # For cross-validation, we'll use the combined training and test data
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    
    logger.info(f"Total data size for cross-validation: {len(X)}")
    
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
    
    # Cross-validation settings
    cv_config = config['cross_validation']
    cv_folds = cv_config['cv_folds']
    cv_repeats = cv_config.get('cv_repeats', 1)
    scoring = cv_config['scoring']
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_config['random_state'])
    
    # Store results
    cv_results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"\nRunning cross-validation for {name}...")
        
        try:
            # Train the model first
            classifier.train(X, y)
            
            # Calculate additional metrics for each fold
            fold_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'roc_auc': []
            }
            
            cv_scores = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train on this fold using the custom classifier
                classifier.train(X_train_fold, y_train_fold)
                y_pred_fold = classifier.predict(X_test_fold)
                
                # Calculate metrics with proper error handling
                fold_metrics['accuracy'].append(accuracy_score(y_test_fold, y_pred_fold))
                
                # Handle precision warnings
                try:
                    precision = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                    fold_metrics['precision'].append(precision)
                except:
                    fold_metrics['precision'].append(0.0)
                
                fold_metrics['recall'].append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
                fold_metrics['f1'].append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
                
                # ROC AUC (if binary classification)
                if len(np.unique(y)) == 2:
                    try:
                        y_pred_proba = classifier.predict_proba(X_test_fold)[:, 1]
                        fold_metrics['roc_auc'].append(roc_auc_score(y_test_fold, y_pred_proba))
                    except:
                        fold_metrics['roc_auc'].append(np.nan)
                else:
                    fold_metrics['roc_auc'].append(np.nan)
                
                # Store CV score
                cv_scores.append(fold_metrics['accuracy'][-1])
            
            cv_scores = np.array(cv_scores)
            
            # Store results
            cv_results[name] = {
                'cv_scores': cv_scores,
                'fold_metrics': fold_metrics,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'mean_accuracy': np.mean(fold_metrics['accuracy']),
                'mean_precision': np.mean(fold_metrics['precision']),
                'mean_recall': np.mean(fold_metrics['recall']),
                'mean_f1': np.mean(fold_metrics['f1']),
                'mean_roc_auc': np.nanmean(fold_metrics['roc_auc'])
            }
            
            logger.info(f"{name} - Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Generate learning curves if configured
            if config['output']['generate_learning_curves']:
                # For XGBoost, use the custom classifier instead of raw model
                if 'XGBoost' in name:
                    generate_learning_curves(classifier, X, y, name, config)
                else:
                    generate_learning_curves(classifier.model, X, y, name, config)
            
            # Generate validation curves if configured
            if config['output']['generate_validation_curves']:
                # For XGBoost, use the custom classifier instead of raw model
                if 'XGBoost' in name:
                    generate_validation_curves(classifier, X, y, name, config)
                else:
                    generate_validation_curves(classifier.model, X, y, name, config)
            
        except Exception as e:
            logger.error(f"Error in cross-validation for {name}: {str(e)}")
            continue
    
    # Save cross-validation results
    if config['output']['save_cv_results']:
        save_cv_results(cv_results, config)
    
    # Save best model
    if config['output']['save_best_model']:
        save_best_model(cv_results, classifiers, config)
    
    logger.info("\nCross-validation experiments completed!")
    return cv_results


def generate_learning_curves(model, X, y, model_name: str, config: Dict[str, Any]):
    """Generate learning curves for the model."""
    logger.info(f"Generating learning curves for {model_name}...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['cross_validation']['random_state'])
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, 
        scoring='accuracy', n_jobs=config['cross_validation'].get('n_jobs', -1)
    )
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='r')
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='g')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves for {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot in model-specific directory
    model_results_dir = get_model_results_dir(config['output']['results_dir'], model_name)
    output_path = os.path.join(model_results_dir, 'learning_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning curves saved to {output_path}")


def generate_validation_curves(model, X, y, model_name: str, config: Dict[str, Any]):
    """Generate validation curves for the model."""
    logger.info(f"Generating validation curves for {model_name}...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['cross_validation']['random_state'])
    
    # Define parameter ranges based on model type
    if 'Random Forest' in model_name:
        param_name = 'n_estimators'
        param_range = [10, 50, 100, 200, 300]
    elif 'Logistic Regression' in model_name:
        param_name = 'C'
        param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    elif 'SVM' in model_name:
        param_name = 'C'
        param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    elif 'XGBoost' in model_name:
        param_name = 'n_estimators'
        param_range = [10, 50, 100, 200, 300]
    elif 'LightGBM' in model_name:
        param_name = 'n_estimators'
        param_range = [10, 50, 100, 200, 300]
    else:
        logger.warning(f"No validation curve parameters defined for {model_name}")
        return
    
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=config['cross_validation'].get('n_jobs', -1)
    )
    
    # Plot validation curves
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    plt.semilogx(param_range, np.mean(val_scores, axis=1), 'o-', color='g', label='Cross-validation score')
    plt.fill_between(param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='r')
    plt.fill_between(param_range, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='g')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curves for {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save plot in model-specific directory
    model_results_dir = get_model_results_dir(config['output']['results_dir'], model_name)
    output_path = os.path.join(model_results_dir, 'validation_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Validation curves saved to {output_path}")


def save_cv_results(cv_results: Dict[str, Any], config: Dict[str, Any]):
    """Save cross-validation results to file."""
    results_data = []
    
    for model_name, results in cv_results.items():
        # Save model-specific results
        model_results_dir = get_model_results_dir(config['output']['results_dir'], model_name)
        model_results_path = os.path.join(model_results_dir, 'cross_validation_results.json')
        
        model_results = {
            'model_name': model_name,
            'cv_scores': results['cv_scores'].tolist(),
            'fold_metrics': results['fold_metrics'],
            'mean_cv_score': float(results['mean_cv_score']),
            'std_cv_score': float(results['std_cv_score']),
            'mean_accuracy': float(results['mean_accuracy']),
            'mean_precision': float(results['mean_precision']),
            'mean_recall': float(results['mean_recall']),
            'mean_f1': float(results['mean_f1']),
            'mean_roc_auc': float(results['mean_roc_auc']) if not np.isnan(results['mean_roc_auc']) else None
        }
        
        import json
        with open(model_results_path, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        
        logger.info(f"Cross-validation results for {model_name} saved to {model_results_path}")
        
        # Add to summary data
        results_data.append({
            'model': model_name,
            'mean_cv_score': results['mean_cv_score'],
            'std_cv_score': results['std_cv_score'],
            'mean_accuracy': results['mean_accuracy'],
            'mean_precision': results['mean_precision'],
            'mean_recall': results['mean_recall'],
            'mean_f1': results['mean_f1'],
            'mean_roc_auc': results['mean_roc_auc']
        })
    
    # Save overall summary
    results_df = pd.DataFrame(results_data)
    output_path = os.path.join(config['output']['results_dir'], 'cross_validation_results.csv')
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Overall cross-validation results saved to {output_path}")


def save_best_model(cv_results: Dict[str, Any], classifiers: Dict[str, Any], config: Dict[str, Any]):
    """Save the best performing model."""
    if not cv_results:
        logger.warning("No cross-validation results available to determine best model")
        return
    
    # Find best model based on mean CV score
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean_cv_score'])
    best_score = cv_results[best_model_name]['mean_cv_score']
    
    logger.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
    
    # Save best model
    if best_model_name in classifiers and hasattr(classifiers[best_model_name], 'save_model'):
        model_path = os.path.join(config['output']['results_dir'], 'best_model.pkl')
        classifiers[best_model_name].save_model(model_path)
        logger.info(f"Best model saved to {model_path}")


def run_prior_testing(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run prior testing experiments."""
    logger.info("Starting prior testing experiments...")
    
    # Load data
    data_config = config['data']
    hormone_data_path = data_config['hormone_data_path']
    survey_data_path = data_config['survey_data_path']
    period_data_path = data_config['period_data_path']
    
    logger.info(f"Loading data from: {hormone_data_path}")
    
    # Load original data for prior
    original_data = pd.read_csv(hormone_data_path)
    
    # Load preprocessed data for ML models
    X_train, X_test, y_train, y_test = load_and_split_data(
        hormone_data_path, 
        test_size=data_config['test_size'], 
        random_state=data_config['random_state']
    )
    
    # For cross-validation, we'll use the combined training and test data
    X = pd.concat([X_train, X_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    
    logger.info(f"Total data size for prior testing: {len(X)}")
    
    # Initialize prior
    prior = RuleBasedPrior(config)
    prior.load_data()
    
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
    
    # Cross-validation settings
    cv_config = config['cross_validation']
    cv_folds = cv_config['cv_folds']
    prior_weight = config['prior_testing']['prior_weight']
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_config['random_state'])
    
    # Store results
    prior_results = {}
    
    for name, classifier in classifiers.items():
        logger.info(f"\nRunning prior testing for {name}...")
        
        try:
            # Train the model first
            classifier.train(X, y)
            
            # Calculate metrics for each fold
            fold_metrics = {
                'ml_only_accuracy': [],
                'prior_only_accuracy': [],
                'combined_accuracy': [],
                'ml_only_precision': [],
                'prior_only_precision': [],
                'combined_precision': [],
                'ml_only_recall': [],
                'prior_only_recall': [],
                'combined_recall': [],
                'ml_only_f1': [],
                'prior_only_f1': [],
                'combined_f1': []
            }
            
            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Get corresponding original data for this fold
                original_test_fold = original_data.iloc[test_idx]
                
                # Train on this fold
                classifier.train(X_train_fold, y_train_fold)
                
                # Get ML predictions
                y_pred_ml = classifier.predict(X_test_fold)
                
                # Get prior predictions using original data
                y_pred_prior = prior.predict_phases(original_test_fold)
                
                # Combine predictions
                y_pred_combined = combine_predictions(
                    y_pred_ml, y_pred_prior, prior_weight
                )
                
                # Calculate metrics for ML-only
                fold_metrics['ml_only_accuracy'].append(accuracy_score(y_test_fold, y_pred_ml))
                fold_metrics['ml_only_precision'].append(
                    precision_score(y_test_fold, y_pred_ml, average='weighted', zero_division=0)
                )
                fold_metrics['ml_only_recall'].append(
                    recall_score(y_test_fold, y_pred_ml, average='weighted')
                )
                fold_metrics['ml_only_f1'].append(
                    f1_score(y_test_fold, y_pred_ml, average='weighted')
                )
                
                # Calculate metrics for prior-only
                fold_metrics['prior_only_accuracy'].append(accuracy_score(y_test_fold, y_pred_prior))
                fold_metrics['prior_only_precision'].append(
                    precision_score(y_test_fold, y_pred_prior, average='weighted', zero_division=0)
                )
                fold_metrics['prior_only_recall'].append(
                    recall_score(y_test_fold, y_pred_prior, average='weighted')
                )
                fold_metrics['prior_only_f1'].append(
                    f1_score(y_test_fold, y_pred_prior, average='weighted')
                )
                
                # Calculate metrics for combined
                fold_metrics['combined_accuracy'].append(accuracy_score(y_test_fold, y_pred_combined))
                fold_metrics['combined_precision'].append(
                    precision_score(y_test_fold, y_pred_combined, average='weighted', zero_division=0)
                )
                fold_metrics['combined_recall'].append(
                    recall_score(y_test_fold, y_pred_combined, average='weighted')
                )
                fold_metrics['combined_f1'].append(
                    f1_score(y_test_fold, y_pred_combined, average='weighted')
                )
            
            # Store results
            prior_results[name] = {
                'ml_only': {
                    'mean_accuracy': np.mean(fold_metrics['ml_only_accuracy']),
                    'mean_precision': np.mean(fold_metrics['ml_only_precision']),
                    'mean_recall': np.mean(fold_metrics['ml_only_recall']),
                    'mean_f1': np.mean(fold_metrics['ml_only_f1']),
                    'std_accuracy': np.std(fold_metrics['ml_only_accuracy'])
                },
                'prior_only': {
                    'mean_accuracy': np.mean(fold_metrics['prior_only_accuracy']),
                    'mean_precision': np.mean(fold_metrics['prior_only_precision']),
                    'mean_recall': np.mean(fold_metrics['prior_only_recall']),
                    'mean_f1': np.mean(fold_metrics['prior_only_f1']),
                    'std_accuracy': np.std(fold_metrics['prior_only_accuracy'])
                },
                'combined': {
                    'mean_accuracy': np.mean(fold_metrics['combined_accuracy']),
                    'mean_precision': np.mean(fold_metrics['combined_precision']),
                    'mean_recall': np.mean(fold_metrics['combined_recall']),
                    'mean_f1': np.mean(fold_metrics['combined_f1']),
                    'std_accuracy': np.std(fold_metrics['combined_accuracy'])
                }
            }
            
            logger.info(f"{name} - ML Only: {prior_results[name]['ml_only']['mean_accuracy']:.4f} (+/- {prior_results[name]['ml_only']['std_accuracy'] * 2:.4f})")
            logger.info(f"{name} - Prior Only: {prior_results[name]['prior_only']['mean_accuracy']:.4f} (+/- {prior_results[name]['prior_only']['std_accuracy'] * 2:.4f})")
            logger.info(f"{name} - Combined: {prior_results[name]['combined']['mean_accuracy']:.4f} (+/- {prior_results[name]['combined']['std_accuracy'] * 2:.4f})")
            
        except Exception as e:
            logger.error(f"Error in prior testing for {name}: {str(e)}")
            continue
    
    # Save prior testing results
    save_prior_results(prior_results, config)
    
    logger.info("\nPrior testing experiments completed!")
    return prior_results


def save_prior_results(prior_results: Dict[str, Any], config: Dict[str, Any]):
    """Save prior testing results to file."""
    results_data = []
    
    for model_name, results in prior_results.items():
        # Save model-specific results
        model_results_dir = get_model_results_dir(config['output']['results_dir'], model_name)
        model_results_path = os.path.join(model_results_dir, 'prior_testing_results.json')
        
        model_results = {
            'model_name': model_name,
            'ml_only': results['ml_only'],
            'prior_only': results['prior_only'],
            'combined': results['combined']
        }
        
        import json
        with open(model_results_path, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        
        logger.info(f"Prior testing results for {model_name} saved to {model_results_path}")
        
        # Add to summary data
        results_data.append({
            'model': model_name,
            'ml_only_accuracy': results['ml_only']['mean_accuracy'],
            'ml_only_precision': results['ml_only']['mean_precision'],
            'ml_only_recall': results['ml_only']['mean_recall'],
            'ml_only_f1': results['ml_only']['mean_f1'],
            'prior_only_accuracy': results['prior_only']['mean_accuracy'],
            'prior_only_precision': results['prior_only']['mean_precision'],
            'prior_only_recall': results['prior_only']['mean_recall'],
            'prior_only_f1': results['prior_only']['mean_f1'],
            'combined_accuracy': results['combined']['mean_accuracy'],
            'combined_precision': results['combined']['mean_precision'],
            'combined_recall': results['combined']['mean_recall'],
            'combined_f1': results['combined']['mean_f1']
        })
    
    # Save overall summary
    results_df = pd.DataFrame(results_data)
    output_path = os.path.join(config['output']['results_dir'], 'prior_testing_results.csv')
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Overall prior testing results saved to {output_path}")


def combine_predictions(ml_predictions: np.ndarray, prior_predictions: np.ndarray, prior_weight: float = 0.3) -> np.ndarray:
    """
    Combine ML and prior predictions using weighted voting.
    
    Args:
        ml_predictions: Array of ML predictions
        prior_predictions: Array of prior predictions
        prior_weight: Weight for prior predictions (0.0 to 1.0)
        
    Returns:
        np.ndarray: Combined predictions
    """
    # Define phase order
    phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
    
    # Convert predictions to indices
    ml_indices = np.array([phases.index(pred) if pred in phases else 0 for pred in ml_predictions])
    prior_indices = np.array([phases.index(pred) if pred in phases else 0 for pred in prior_predictions])
    
    # Create probability matrices
    n_samples = len(ml_predictions)
    n_phases = len(phases)
    
    ml_probs = np.zeros((n_samples, n_phases))
    prior_probs = np.zeros((n_samples, n_phases))
    
    # Set probabilities based on predictions
    for i in range(n_samples):
        ml_probs[i, ml_indices[i]] = 1.0
        prior_probs[i, prior_indices[i]] = 1.0
    
    # Combine probabilities
    combined_probs = (1 - prior_weight) * ml_probs + prior_weight * prior_probs
    
    # Get final predictions
    combined_predictions = np.array([phases[np.argmax(probs)] for probs in combined_probs])
    
    return combined_predictions


def main():
    """Main function to run cross-validation and/or prior testing."""
    # Load configuration
    config = load_config('config/cross_validation_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['figures_dir'], exist_ok=True)
    
    # Check if prior testing is enabled
    prior_config = config.get('prior_testing', {})
    prior_enabled = prior_config.get('enabled', False)
    test_both = prior_config.get('test_both', True)
    
    if prior_enabled:
        if test_both:
            logger.info("Running both ML-only cross-validation and prior testing...")
            # Run standard cross-validation
            cv_results = run_cross_validation(config)
            
            # Run prior testing
            prior_results = run_prior_testing(config)
            
            # Print summary
            print("\n=== CROSS-VALIDATION RESULTS SUMMARY ===")
            for name, metrics in cv_results.items():
                print(f"{name}:")
                print(f"  Mean CV Score: {metrics['mean_cv_score']:.4f} (+/- {metrics['std_cv_score'] * 2:.4f})")
                print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
                print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
                print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
                print(f"  Mean F1-Score: {metrics['mean_f1']:.4f}")
                print()
            
            print("\n=== PRIOR TESTING RESULTS SUMMARY ===")
            for name, results in prior_results.items():
                print(f"{name}:")
                print(f"  ML Only - Accuracy: {results['ml_only']['mean_accuracy']:.4f} (+/- {results['ml_only']['std_accuracy'] * 2:.4f})")
                print(f"  Prior Only - Accuracy: {results['prior_only']['mean_accuracy']:.4f} (+/- {results['prior_only']['std_accuracy'] * 2:.4f})")
                print(f"  Combined - Accuracy: {results['combined']['mean_accuracy']:.4f} (+/- {results['combined']['std_accuracy'] * 2:.4f})")
                print()
        else:
            logger.info("Running prior testing only...")
            prior_results = run_prior_testing(config)
            
            # Print summary
            print("\n=== PRIOR TESTING RESULTS SUMMARY ===")
            for name, results in prior_results.items():
                print(f"{name}:")
                print(f"  ML Only - Accuracy: {results['ml_only']['mean_accuracy']:.4f} (+/- {results['ml_only']['std_accuracy'] * 2:.4f})")
                print(f"  Prior Only - Accuracy: {results['prior_only']['mean_accuracy']:.4f} (+/- {results['prior_only']['std_accuracy'] * 2:.4f})")
                print(f"  Combined - Accuracy: {results['combined']['mean_accuracy']:.4f} (+/- {results['combined']['std_accuracy'] * 2:.4f})")
                print()
    else:
        logger.info("Running ML-only cross-validation...")
        # Run standard cross-validation
        results = run_cross_validation(config)
        
        # Print summary
        print("\n=== CROSS-VALIDATION RESULTS SUMMARY ===")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Mean CV Score: {metrics['mean_cv_score']:.4f} (+/- {metrics['std_cv_score'] * 2:.4f})")
            print(f"  Mean Accuracy: {metrics['mean_accuracy']:.4f}")
            print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
            print(f"  Mean Recall: {metrics['mean_recall']:.4f}")
            print(f"  Mean F1-Score: {metrics['mean_f1']:.4f}")
            print()


if __name__ == '__main__':
    main() 