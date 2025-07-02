#!/usr/bin/env python3
"""
Prediction script for making predictions on unlabeled hormone data.
Loads trained models and predicts menstrual phases.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config, preprocess_unlabeled_data_with_prior, preprocess_unlabeled_data, DataPreprocessor
from src.utils.model_utils import load_model_bundle, get_bundle_info, list_available_bundles
from src.classification.rule_based_prior import RuleBasedPrior

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_unlabeled_data(data_path: str, config: Dict[str, Any], model_path: str) -> pd.DataFrame:
    """
    Load and preprocess unlabeled data for prediction using the saved preprocessor.
    
    Args:
        data_path: Path to unlabeled data file
        config: Configuration dictionary
        model_path: Path to the trained model (to get preprocessor path)
        
    Returns:
        pd.DataFrame: Preprocessed feature matrix
    """
    logger.info(f"Loading unlabeled data from: {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Load the model bundle to get the preprocessor
    bundle = load_model_bundle(model_path)
    preprocessor = bundle.preprocessor
    
    logger.info(f"Loaded preprocessor from model bundle")
    
    # Preprocess data with prior features if enabled
    if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
        logger.info("Using rule-based prior as features...")
        X = preprocess_unlabeled_data_with_prior(data, config, preprocessor)
    else:
        logger.info("Using standard preprocessing without prior...")
        X = preprocess_unlabeled_data(data, preprocessor)
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    return X


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


def make_predictions(model_path: str, X: pd.DataFrame, original_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions using a trained model.
    
    Args:
        model_path: Path to the trained model
        X: Preprocessed feature matrix
        original_data: Original unprocessed data (for rule-based prior)
        config: Configuration dictionary
        
    Returns:
        dict: Prediction results
    """
    logger.info(f"Making predictions using model: {model_path}")
    
    # Load the model bundle
    bundle = load_model_bundle(model_path)
    logger.info(f"Loaded model bundle: {bundle.model_name}")
    
    # Get model info
    model_info = bundle.get_info()
    logger.info(f"Model info: {model_info}")
    
    # Check if model was trained with prior features
    model_used_prior_features = model_info.get('config_summary', {}).get('models_config', {}).get('temporal', {}).get('use_as_prior', False)
    current_using_prior_features = config.get('models', {}).get('temporal', {}).get('use_as_prior', False)
    
    # Prepare features for model prediction
    if model_used_prior_features and current_using_prior_features:
        # Both training and prediction use prior features - use all features
        X_for_model = X
        logger.info("Using all features (including prior features) for prediction")
    elif not model_used_prior_features and current_using_prior_features:
        # Model trained without prior, but prediction wants to use prior
        # Use all features except prior features
        non_prior_features = [col for col in X.columns if not col.startswith('prior_')]
        X_for_model = X[non_prior_features]
        logger.info(f"Model trained without prior features. Using all non-prior features: {non_prior_features}")
    elif model_used_prior_features and not current_using_prior_features:
        # Model trained with prior, but prediction doesn't want to use prior
        # This case needs the prior features to be present
        logger.warning("Model was trained with prior features but prediction config doesn't use them")
        X_for_model = X
    else:
        # Neither training nor prediction use prior features
        X_for_model = X
        logger.info("Using standard features for prediction")
    
    # Validate features before making predictions
    try:
        bundle.validate_prediction_data(X_for_model)
        logger.info(f"Feature validation passed for {bundle.model_name}")
    except Exception as e:
        logger.error(f"Feature validation failed for {bundle.model_name}: {str(e)}")
        raise ValueError(f"Feature validation failed for {bundle.model_name}. Please check your input data and model configuration. Error: {str(e)}")
    
    # Make predictions using the bundle
    predictions = bundle.predict(X_for_model)
    
    # Get prediction probabilities if available
    probabilities = None
    try:
        probabilities = bundle.predict_proba(X_for_model)
        logger.info("Successfully obtained prediction probabilities")
    except Exception as e:
        logger.warning(f"Could not get prediction probabilities: {str(e)}")
    
    # Calculate confidence scores
    confidence_scores = None
    if probabilities is not None:
        confidence_scores = np.max(probabilities, axis=1)
    
    # If using rule-based prior for post-processing (not as features), combine predictions
    if current_using_prior_features and not model_used_prior_features:
        logger.info("Combining model predictions with rule-based prior...")
        
        # Initialize rule-based prior
        rule_prior = RuleBasedPrior(config)
        rule_prior.load_data()
        
        # Get prior predictions using original data
        prior_predictions = rule_prior.predict_phases(original_data)
        prior_probs = rule_prior.get_prior_probabilities(original_data)
        
        # Combine predictions using perfect prior logic
        prior_weight = config.get('models', {}).get('temporal', {}).get('prior_weight', 0.3)
        combined_predictions = combine_predictions_with_perfect_prior(
            predictions, prior_predictions, probabilities, prior_probs, prior_weight
        )
        
        predictions = combined_predictions
        
        # Combine probabilities if available
        if probabilities is not None:
            combined_probs = (1 - prior_weight) * probabilities + prior_weight * prior_probs
            probabilities = combined_probs
            confidence_scores = np.max(combined_probs, axis=1)
        
        logger.info(f"Combined predictions with rule-based prior (weight: {prior_weight})")
    
    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'confidence_scores': confidence_scores,
        'model_info': model_info
    }
    
    return results


def generate_combined_summary(config: Dict[str, Any]) -> None:
    """
    Generate a combined summary comparing all models' predictions.
    """
    results_dir = config['output']['results_dir']
    predictions_dir = config['output']['predictions_dir']
    
    # Load all prediction summaries from model subdirectories
    summaries = {}
    all_predictions = {}
    
    # Look for model subdirectories
    for model_dir in os.listdir(predictions_dir):
        model_path = os.path.join(predictions_dir, model_dir)
        if os.path.isdir(model_path):
            # Check for prediction summary in this model directory
            summary_file = os.path.join(model_path, 'prediction_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summaries[model_dir] = json.load(f)
            
            # Check for predictions CSV in this model directory
            predictions_file = os.path.join(model_path, 'predictions.csv')
            if os.path.exists(predictions_file):
                predictions_df = pd.read_csv(predictions_file)
                all_predictions[model_dir] = predictions_df
    
    # Fallback: also check for old-style files in the main predictions directory
    if not summaries:
        summary_files = glob.glob(os.path.join(results_dir, 'prediction_summary_*.json'))
        for summary_file in summary_files:
            model_name = os.path.basename(summary_file).replace('prediction_summary_', '').replace('.json', '')
            with open(summary_file, 'r') as f:
                summaries[model_name] = json.load(f)
    
    if not all_predictions:
        prediction_files = glob.glob(os.path.join(predictions_dir, 'predictions_*.csv'))
        for pred_file in prediction_files:
            model_name = os.path.basename(pred_file).replace('predictions_', '').replace('.csv', '')
            predictions_df = pd.read_csv(pred_file)
            all_predictions[model_name] = predictions_df
    
    # Create combined summary
    combined_summary = {
        'total_models': len(summaries),
        'models_compared': list(summaries.keys()),
        'total_predictions': summaries[list(summaries.keys())[0]]['total_predictions'] if summaries else 0,
        'model_performance': {},
        'prediction_agreement': {},
        'confidence_comparison': {},
        'best_model_by_metric': {}
    }
    
    # Model performance comparison
    for model_name, summary in summaries.items():
        combined_summary['model_performance'][model_name] = {
            'mean_confidence': summary['mean_confidence'],
            'high_confidence_predictions': summary['high_confidence_predictions'],
            'prediction_distribution': summary['prediction_distribution']
        }
    
    # Prediction agreement analysis
    if len(all_predictions) > 1:
        # Get all unique phases across all models
        all_phases = set()
        for predictions_df in all_predictions.values():
            all_phases.update(predictions_df['predicted_phase'].unique())
        
        # Create agreement matrix
        agreement_matrix = {}
        for phase in all_phases:
            agreement_matrix[phase] = {}
            for model1 in all_predictions.keys():
                agreement_matrix[phase][model1] = {}
                for model2 in all_predictions.keys():
                    if model1 != model2:
                        # Count how many times both models predicted the same phase
                        pred1 = all_predictions[model1]['predicted_phase']
                        pred2 = all_predictions[model2]['predicted_phase']
                        agreement_count = ((pred1 == phase) & (pred2 == phase)).sum()
                        total_count = (pred1 == phase).sum()
                        agreement_rate = agreement_count / total_count if total_count > 0 else 0
                        agreement_matrix[phase][model1][model2] = {
                            'agreement_count': int(agreement_count),
                            'agreement_rate': float(agreement_rate)
                        }
        
        combined_summary['prediction_agreement'] = agreement_matrix
    
    # Confidence comparison
    models_with_confidence = {}
    for name, summary in summaries.items():
        mean_conf = summary['mean_confidence']
        if mean_conf is not None:
            try:
                # Convert to float if it's a string
                mean_conf_float = float(mean_conf) if isinstance(mean_conf, str) else mean_conf
                models_with_confidence[name] = summary.copy()
                models_with_confidence[name]['mean_confidence'] = mean_conf_float
            except (ValueError, TypeError):
                # Skip models with invalid confidence values
                continue
    
    if models_with_confidence:
        confidence_ranking = sorted(models_with_confidence.items(), 
                                  key=lambda x: x[1]['mean_confidence'], reverse=True)
        combined_summary['confidence_comparison'] = {
            'ranking': [{'model': name, 'mean_confidence': summary['mean_confidence']} 
                       for name, summary in confidence_ranking],
            'highest_confidence_model': confidence_ranking[0][0],
            'lowest_confidence_model': confidence_ranking[-1][0]
        }
    
    # Best model by different metrics
    if models_with_confidence:
        combined_summary['best_model_by_metric'] = {
            'highest_mean_confidence': confidence_ranking[0][0],
            'most_high_confidence_predictions': max(summaries.items(), 
                key=lambda x: x[1]['high_confidence_predictions'] or 0)[0]
        }
    
    # Before saving combined_summary, convert all dict keys to str for JSON compatibility
    def convert_keys_to_str(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys_to_str(i) for i in obj]
        else:
            return obj
    combined_summary = convert_keys_to_str(combined_summary)
    
    # Save combined summary
    combined_summary_path = os.path.join(results_dir, 'combined_prediction_summary.json')
    with open(combined_summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    
    # Print combined summary
    print("\n" + "="*60)
    print("COMBINED PREDICTION SUMMARY")
    print("="*60)
    print(f"Total models compared: {combined_summary['total_models']}")
    print(f"Models: {', '.join(combined_summary['models_compared'])}")
    print(f"Total predictions per model: {combined_summary['total_predictions']}")
    
    if combined_summary['confidence_comparison']:
        print(f"\nConfidence Ranking:")
        for i, item in enumerate(combined_summary['confidence_comparison']['ranking'], 1):
            print(f"  {i}. {item['model']}: {item['mean_confidence']:.3f}")
    
    if combined_summary['best_model_by_metric']:
        print(f"\nBest Models:")
        for metric, model in combined_summary['best_model_by_metric'].items():
            print(f"  {metric}: {model}")
    
    print(f"\nDetailed results saved to: {combined_summary_path}")
    print("="*60)
    
    logger.info(f"Combined summary saved to: {combined_summary_path}")


def predict_with_all_models(config: Dict[str, Any]):
    """
    Run prediction pipeline for all available model bundles.
    """
    # Load and cache original data
    data_path = config['data']['unlabeled_data_path']
    original_data = pd.read_csv(data_path)
    models_dir = config['output']['models_dir']
    bundle_files = list_available_bundles(models_dir)
    if not bundle_files:
        raise FileNotFoundError(f"No model bundles found in {models_dir}")
    
    for bundle_file in bundle_files:
        model_path = os.path.join(models_dir, bundle_file)
        bundle = load_model_bundle(model_path)
        model_name = bundle.model_name
        print(f"\n=== Predicting with {model_name} ===")
        # Preprocess data using the bundle's preprocessor
        preprocessor = bundle.preprocessor
        if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
            X = preprocess_unlabeled_data_with_prior(original_data, config, preprocessor)
        else:
            X = preprocess_unlabeled_data(original_data, preprocessor)
        # Make predictions
        results = make_predictions(model_path, X, original_data, config)
        # Save predictions with model-specific filenames
        save_predictions(
            results['predictions'],
            original_data,
            results,
            config,
            model_name=model_name
        )
    
    # Generate combined summary
    generate_combined_summary(config)


def save_predictions(predictions: np.ndarray, original_data: pd.DataFrame, 
                    results: Dict[str, Any], config: Dict[str, Any], model_name: str = None) -> None:
    """
    Save prediction results to files, with optional model-specific filenames.
    Ensures all predicted_phase values are string labels.
    """
    output_config = config['output']
    
    # Create model-specific subdirectory
    if model_name:
        model_dir = os.path.join(output_config['predictions_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created model-specific directory: {model_dir}")
    else:
        model_dir = output_config['predictions_dir']
    
    # Create predictions DataFrame
    predictions_df = original_data.copy()
    
    # Convert numeric predictions to string labels if needed
    classes = results['model_info'].get('classes', None)
    if classes is not None:
        # Check if we need to convert numeric predictions to string labels
        # This handles cases where the model outputs numeric indices but we want string labels
        first_pred = predictions[0] if len(predictions) > 0 else None
        first_class = classes[0] if len(classes) > 0 else None
        
        # Convert numpy types to regular types for comparison
        if hasattr(first_pred, 'item'):
            first_pred = first_pred.item()
        if hasattr(first_class, 'item'):
            first_class = first_class.item()
        
        # If predictions are numeric and classes are strings, map them
        if (isinstance(first_pred, (int, float)) and isinstance(first_class, str)):
            try:
                predictions = [classes[int(idx)] for idx in predictions]
                logger.info(f"Converted numeric predictions to string labels for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to convert predictions for {model_name}: {e}")
    
    predictions_df['predicted_phase'] = predictions
    if results['confidence_scores'] is not None:
        predictions_df['confidence'] = results['confidence_scores']
    
    # Save predictions
    if output_config['save_predictions']:
        predictions_path = os.path.join(model_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
    
    # Save probabilities if available
    if output_config['save_probabilities'] and results['probabilities'] is not None:
        proba_df = pd.DataFrame(
            results['probabilities'],
            columns=results['model_info'].get('classes', [f'class_{i}' for i in range(results['probabilities'].shape[1])])
        )
        proba_path = os.path.join(model_dir, 'prediction_probabilities.csv')
        proba_df.to_csv(proba_path, index=False)
        logger.info(f"Prediction probabilities saved to: {proba_path}")
    
    # Generate prediction summary
    if output_config['generate_prediction_summary']:
        generate_prediction_summary(predictions_df, results, config, model_name, model_dir)
    
    # Generate prediction plots
    if output_config['save_prediction_plots']:
        generate_prediction_plots(predictions_df, results, config, model_name, model_dir)
    
    # Generate confusion matrix by matching with labeled data
    if output_config.get('generate_confusion_matrix', True):
        generate_confusion_matrix(predictions_df, config, model_name, model_dir)


def generate_prediction_summary(predictions_df: pd.DataFrame, results: Dict[str, Any], config: Dict[str, Any], model_name: str = None, model_dir: str = None) -> None:
    """
    Generate a summary of predictions, with optional model-specific filename.
    """
    summary = {
        'total_predictions': len(predictions_df),
        'model_used': results['model_info']['model_name'],
        'prediction_distribution': predictions_df['predicted_phase'].value_counts().to_dict(),
        'mean_confidence': predictions_df['confidence'].mean() if 'confidence' in predictions_df.columns else None,
        'high_confidence_predictions': None
    }
    if 'confidence' in predictions_df.columns:
        summary['high_confidence_predictions'] = int((predictions_df['confidence'] > config['prediction']['confidence_threshold']).sum())
    
    # Save summary
    if model_dir:
        summary_path = os.path.join(model_dir, 'prediction_summary.json')
    else:
        suffix = f"_{model_name}" if model_name else ""
        summary_path = os.path.join(config['output']['results_dir'], f'prediction_summary{suffix}.json')
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Prediction summary saved to: {summary_path}")
    
    # Print summary
    print(f"\n=== PREDICTION SUMMARY ({model_name}) ===")
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Model used: {summary['model_used']}")
    print(f"Mean confidence: {summary['mean_confidence']}")
    print(f"High confidence predictions: {summary['high_confidence_predictions']}")
    print("\nPrediction distribution:")
    for phase, count in summary['prediction_distribution'].items():
        print(f"  {phase}: {count}")


def generate_prediction_plots(predictions_df: pd.DataFrame, results: Dict[str, Any], config: Dict[str, Any], model_name: str = None, model_dir: str = None) -> None:
    """
    Generate plots for prediction results, with optional model-specific filename.
    """
    # Prediction distribution plot
    plt.figure(figsize=(10, 6))
    predictions_df['predicted_phase'].value_counts().plot(kind='bar')
    plt.title(f'Prediction Distribution ({model_name})')
    plt.xlabel('Predicted Phase')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if model_dir:
        plot_path = os.path.join(model_dir, 'prediction_distribution.png')
    else:
        suffix = f"_{model_name}" if model_name else ""
        plot_path = os.path.join(config['output']['predictions_dir'], f'prediction_distribution{suffix}.png')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Prediction distribution plot saved to: {plot_path}")
    
    # Confidence distribution plot (if available)
    if 'confidence' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['confidence'], bins=20, alpha=0.7)
        plt.title(f'Prediction Confidence Distribution ({model_name})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.axvline(config['prediction']['confidence_threshold'], color='red', linestyle='--', 
                   label=f'Threshold ({config['prediction']['confidence_threshold']})')
        plt.legend()
        plt.tight_layout()
        
        if model_dir:
            conf_path = os.path.join(model_dir, 'confidence_distribution.png')
        else:
            suffix = f"_{model_name}" if model_name else ""
            conf_path = os.path.join(config['output']['predictions_dir'], f'confidence_distribution{suffix}.png')
        
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confidence distribution plot saved to: {conf_path}")


def generate_confusion_matrix(predictions_df: pd.DataFrame, config: Dict[str, Any], model_name: str = None, model_dir: str = None) -> None:
    """
    Generate confusion matrix by matching predictions on unlabeled data with true phases from labeled data.
    Matches samples by subject_id and date.
    """
    try:
        # Load labeled data to get true phases
        labeled_data_path = config['data'].get('hormone_data_path', 'outputs/data/full_hormone_data_labeled.csv')
        labeled_data = pd.read_csv(labeled_data_path)
        labeled_data['date'] = pd.to_datetime(labeled_data['date'])
        
        # Ensure predictions_df has date as datetime
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Merge predictions with labeled data on subject_id and date
        merged_data = predictions_df.merge(
            labeled_data[['subject_id', 'date', 'phase']], 
            on=['subject_id', 'date'], 
            how='inner',
            suffixes=('', '_true')
        )
        
        if len(merged_data) == 0:
            logger.warning(f"No matching samples found between unlabeled predictions and labeled data for {model_name}")
            return
        
        logger.info(f"Found {len(merged_data)} matching samples for confusion matrix generation")
        
        # Get true and predicted phases
        y_true = merged_data['phase']
        y_pred = merged_data['predicted_phase']
        
        # Handle XGBoost numeric predictions by converting to string phase names
        if y_pred.dtype in ['int64', 'int32', 'float64', 'float32'] or any(isinstance(x, (int, float)) for x in y_pred):
            logger.info(f"Converting numeric predictions to string phase names for {model_name}")
            
            # Define the mapping from numeric indices to phase names
            # This should match the order used in the model's label encoder
            phase_mapping = {
                0: 'early_luteal',
                1: 'mid_follicular', 
                2: 'mid_late_luteal',
                3: 'perimenstruation',
                4: 'periovulation'
            }
            
            # Convert numeric predictions to string phase names
            y_pred_strings = []
            for pred in y_pred:
                if pred in phase_mapping:
                    y_pred_strings.append(phase_mapping[pred])
                else:
                    # Fallback for unexpected values
                    logger.warning(f"Unexpected prediction value: {pred}, using 'unknown'")
                    y_pred_strings.append('unknown')
            
            y_pred = pd.Series(y_pred_strings)
        
        # Define phase order for consistent confusion matrix
        phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=phases)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=phases, yticklabels=phases)
        plt.xlabel('Predicted Phase')
        plt.ylabel('True Phase')
        plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        plt.tight_layout()
        
        # Save confusion matrix plot
        if model_dir:
            cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        else:
            suffix = f"_{model_name}" if model_name else ""
            cm_path = os.path.join(config['output']['predictions_dir'], f'confusion_matrix{suffix}.png')
        
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        # Save confusion matrix metrics
        cm_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_samples': len(merged_data),
            'confusion_matrix': cm.tolist(),
            'phase_labels': phases
        }
        
        if model_dir:
            metrics_path = os.path.join(model_dir, 'confusion_matrix_metrics.json')
        else:
            suffix = f"_{model_name}" if model_name else ""
            metrics_path = os.path.join(config['output']['predictions_dir'], f'confusion_matrix_metrics{suffix}.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(cm_metrics, f, indent=2, default=str)
        logger.info(f"Confusion matrix metrics saved to: {metrics_path}")
        
        # Print metrics
        print(f"\n=== CONFUSION MATRIX METRICS ({model_name}) ===")
        print(f"Total matched samples: {len(merged_data)}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix for {model_name}: {str(e)}")


def main():
    """Main function to run predictions for all models."""
    # Load configuration
    config = load_config('config/prediction_config.yaml')
    # Create output directories
    os.makedirs(config['output']['predictions_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    # Predict with all models
    predict_with_all_models(config)


if __name__ == '__main__':
    main() 