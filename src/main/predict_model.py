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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config, preprocess_unlabeled_data_with_prior
from src.utils.model_utils import load_model, predict_with_model, predict_proba_with_model, get_model_info
from src.temporal_models.rule_based_prior import RuleBasedPrior

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_unlabeled_data(data_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and preprocess unlabeled data for prediction.
    
    Args:
        data_path: Path to unlabeled data file
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame: Preprocessed feature matrix
    """
    logger.info(f"Loading unlabeled data from: {data_path}")
    
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} samples")
    
    # Preprocess data with prior features if enabled
    if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
        logger.info("Using rule-based prior as features...")
        X = preprocess_unlabeled_data_with_prior(data, config)
    else:
        logger.info("Using standard preprocessing without prior...")
        X = preprocess_unlabeled_data(data)
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    return X


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
    
    # Get model info
    model_info = get_model_info(model_path)
    logger.info(f"Model info: {model_info}")
    
    # Check if model was trained with prior features
    model_used_prior_features = model_info.get('metadata', {}).get('used_prior_features', False)
    current_using_prior_features = config.get('models', {}).get('temporal', {}).get('use_as_prior', False)
    
    # Prepare features for model prediction
    if model_used_prior_features and current_using_prior_features:
        # Both training and prediction use prior features - use all features
        X_for_model = X
        logger.info("Using all features (including prior features) for prediction")
    elif not model_used_prior_features and current_using_prior_features:
        # Model trained without prior, but prediction wants to use prior
        # Extract only the original hormone features for model prediction
        original_features = ['estradiol', 'progesterone', 'testosterone']
        X_for_model = X[original_features]
        logger.info(f"Model trained without prior features. Using only original features: {original_features}")
    elif model_used_prior_features and not current_using_prior_features:
        # Model trained with prior, but prediction doesn't want to use prior
        # This case needs the prior features to be present
        logger.warning("Model was trained with prior features but prediction config doesn't use them")
        X_for_model = X
    else:
        # Neither training nor prediction use prior features
        X_for_model = X
        logger.info("Using standard features for prediction")
    
    # Make predictions
    predictions = predict_with_model(model_path, X_for_model)
    
    # Get prediction probabilities if available
    probabilities = None
    try:
        probabilities = predict_proba_with_model(model_path, X_for_model)
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
        
        # Combine predictions (simple ensemble for now)
        prior_weight = config.get('models', {}).get('temporal', {}).get('prior_weight', 0.3)
        combined_predictions = []
        
        for i in range(len(predictions)):
            if np.random.random() < prior_weight:
                combined_predictions.append(prior_predictions[i])
            else:
                combined_predictions.append(predictions[i])
        
        predictions = np.array(combined_predictions)
        
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


def save_predictions(predictions: np.ndarray, original_data: pd.DataFrame, 
                    results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Save prediction results to files.
    
    Args:
        predictions: Model predictions
        original_data: Original unlabeled data
        results: Complete prediction results
        config: Configuration dictionary
    """
    output_config = config['output']
    
    # Create predictions DataFrame
    predictions_df = original_data.copy()
    predictions_df['predicted_phase'] = predictions
    
    if results['confidence_scores'] is not None:
        predictions_df['confidence'] = results['confidence_scores']
    
    # Save predictions
    if output_config['save_predictions']:
        predictions_path = os.path.join(output_config['predictions_dir'], 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
    
    # Save probabilities if available
    if output_config['save_probabilities'] and results['probabilities'] is not None:
        proba_df = pd.DataFrame(
            results['probabilities'],
            columns=results['model_info'].get('classes', [f'class_{i}' for i in range(results['probabilities'].shape[1])])
        )
        proba_path = os.path.join(output_config['predictions_dir'], 'prediction_probabilities.csv')
        proba_df.to_csv(proba_path, index=False)
        logger.info(f"Prediction probabilities saved to: {proba_path}")
    
    # Generate prediction summary
    if output_config['generate_prediction_summary']:
        generate_prediction_summary(predictions_df, results, config)
    
    # Generate prediction plots
    if output_config['save_prediction_plots']:
        generate_prediction_plots(predictions_df, results, config)


def generate_prediction_summary(predictions_df: pd.DataFrame, results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Generate a summary of predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        results: Prediction results
        config: Configuration dictionary
    """
    summary = {
        'total_predictions': len(predictions_df),
        'model_used': results['model_info']['model_name'],
        'prediction_distribution': predictions_df['predicted_phase'].value_counts().to_dict(),
        'mean_confidence': predictions_df.get('confidence', pd.Series([np.nan])).mean(),
        'high_confidence_predictions': len(predictions_df[predictions_df.get('confidence', 0) > config['prediction']['confidence_threshold']])
    }
    
    # Save summary
    summary_path = os.path.join(config['output']['results_dir'], 'prediction_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Prediction summary saved to: {summary_path}")
    
    # Print summary
    print("\n=== PREDICTION SUMMARY ===")
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Model used: {summary['model_used']}")
    print(f"Mean confidence: {summary['mean_confidence']:.3f}")
    print(f"High confidence predictions: {summary['high_confidence_predictions']}")
    print("\nPrediction distribution:")
    for phase, count in summary['prediction_distribution'].items():
        print(f"  {phase}: {count}")


def generate_prediction_plots(predictions_df: pd.DataFrame, results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Generate plots for prediction results.
    
    Args:
        predictions_df: DataFrame with predictions
        results: Prediction results
        config: Configuration dictionary
    """
    # Prediction distribution plot
    plt.figure(figsize=(10, 6))
    predictions_df['predicted_phase'].value_counts().plot(kind='bar')
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Phase')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(config['output']['predictions_dir'], 'prediction_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Prediction distribution plot saved to: {plot_path}")
    
    # Confidence distribution plot (if available)
    if 'confidence' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['confidence'], bins=20, alpha=0.7)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.axvline(config['prediction']['confidence_threshold'], color='red', linestyle='--', 
                   label=f'Threshold ({config["prediction"]["confidence_threshold"]})')
        plt.legend()
        plt.tight_layout()
        
        conf_path = os.path.join(config['output']['predictions_dir'], 'confidence_distribution.png')
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confidence distribution plot saved to: {conf_path}")


def main():
    """Main function to run predictions."""
    # Load configuration
    config = load_config('config/prediction_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['predictions_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    # Load and preprocess unlabeled data
    data_path = config['data']['unlabeled_data_path']
    original_data = pd.read_csv(data_path)
    X = load_and_preprocess_unlabeled_data(data_path, config)
    
    # Make predictions
    model_path = config['data']['model_path']
    results = make_predictions(model_path, X, original_data, config)
    
    # Save predictions
    save_predictions(results['predictions'], original_data, results, config)
    
    logger.info("Prediction pipeline completed!")


if __name__ == '__main__':
    main() 