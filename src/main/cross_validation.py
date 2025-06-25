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
    SVMClassifier
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cross_validation(config: Dict[str, Any]):
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
        'SVM': SVMClassifier(config)
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
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                classifier.model, 
                X, 
                y, 
                cv=cv, 
                scoring=scoring,
                n_jobs=cv_config.get('n_jobs', -1)
            )
            
            # Calculate additional metrics for each fold
            fold_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'roc_auc': []
            }
            
            for train_idx, test_idx in cv.split(X, y):
                X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train on this fold
                classifier.model.fit(X_train_fold, y_train_fold)
                y_pred_fold = classifier.model.predict(X_test_fold)
                
                # Calculate metrics
                fold_metrics['accuracy'].append(accuracy_score(y_test_fold, y_pred_fold))
                fold_metrics['precision'].append(precision_score(y_test_fold, y_pred_fold, average='weighted'))
                fold_metrics['recall'].append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
                fold_metrics['f1'].append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
                
                # ROC AUC (if binary classification)
                if len(np.unique(y)) == 2:
                    try:
                        y_pred_proba = classifier.model.predict_proba(X_test_fold)[:, 1]
                        fold_metrics['roc_auc'].append(roc_auc_score(y_test_fold, y_pred_proba))
                    except:
                        fold_metrics['roc_auc'].append(np.nan)
                else:
                    fold_metrics['roc_auc'].append(np.nan)
            
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
                generate_learning_curves(classifier.model, X, y, name, config)
            
            # Generate validation curves if configured
            if config['output']['generate_validation_curves']:
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
    
    # Save plot
    output_path = os.path.join(config['output']['figures_dir'], f'{model_name.lower().replace(" ", "_")}_learning_curves.png')
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
    
    # Save plot
    output_path = os.path.join(config['output']['figures_dir'], f'{model_name.lower().replace(" ", "_")}_validation_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Validation curves saved to {output_path}")


def save_cv_results(cv_results: Dict[str, Any], config: Dict[str, Any]):
    """Save cross-validation results to file."""
    results_data = []
    
    for model_name, results in cv_results.items():
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
    
    results_df = pd.DataFrame(results_data)
    output_path = os.path.join(config['output']['results_dir'], 'cross_validation_results.csv')
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Cross-validation results saved to {output_path}")


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


def main():
    """Main function to run cross-validation."""
    # Load configuration
    config = load_config('config/cross_validation_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['figures_dir'], exist_ok=True)
    
    # Run cross-validation
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