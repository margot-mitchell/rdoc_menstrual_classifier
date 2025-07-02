#!/usr/bin/env python3
"""
Script to compare model performance at different prior weights.
Tests the rule-based prior as a post-processing ensemble with various weights.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config, preprocess_unlabeled_data
from src.utils.model_utils import load_model, get_bundle_info, load_model_bundle
from src.classification.rule_based_prior import RuleBasedPrior
from src.utils.evaluator import ModelEvaluator

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


def evaluate_prior_weight(model_path: str, X: pd.DataFrame, original_data: pd.DataFrame, 
                         prior_weight: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model performance with a specific prior weight.
    
    Args:
        model_path: Path to the trained model
        X: Preprocessed feature matrix
        original_data: Original unprocessed data
        prior_weight: Weight for rule-based prior (0.0 to 1.0)
        config: Configuration dictionary
        
    Returns:
        dict: Performance metrics
    """
    logger.info(f"Testing prior weight: {prior_weight}")
    
    # Get model info
    model_bundle = load_model_bundle(model_path)
    model_info = get_bundle_info(model_bundle)
    
    # Use all preprocessed features for model prediction
    X_for_model = X
    
    # Make base model predictions
    base_predictions = model_bundle.predict(X_for_model)
    
    # Use ModelBundle.predict_proba instead of deprecated function
    base_probabilities = model_bundle.predict_proba(X_for_model)
    
    # Initialize rule-based prior
    rule_prior = RuleBasedPrior(config)
    rule_prior.load_data()
    
    # Get prior predictions
    prior_predictions = rule_prior.predict_phases(original_data)
    prior_probs = rule_prior.get_prior_probabilities(original_data)
    
    # Combine predictions based on prior weight
    if prior_weight == 0.0:
        # No prior influence
        final_predictions = base_predictions
        final_probabilities = base_probabilities
    elif prior_weight == 1.0:
        # Only prior influence
        final_predictions = prior_predictions
        final_probabilities = prior_probs
    else:
        # Combined using perfect prior logic
        final_predictions = combine_predictions_with_perfect_prior(
            base_predictions, prior_predictions, base_probabilities, prior_probs, prior_weight
        )
        
        # Calculate confidence scores from combined probabilities
        phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
        perfect_prior_phases = ['perimenstruation']
        
        confidence_scores = []
        for i in range(len(base_predictions)):
            ml_pred = base_predictions[i]
            prior_pred = prior_predictions[i]
            
            if prior_pred in perfect_prior_phases or ml_pred in perfect_prior_phases:
                # Use prior confidence for perfect phases
                confidence_scores.append(np.max(prior_probs[i]))
            else:
                # Use combined confidence for other phases
                combined_probs = (1 - prior_weight) * base_probabilities[i] + prior_weight * prior_probs[i]
                confidence_scores.append(np.max(combined_probs))
        
        confidence_scores = np.array(confidence_scores)
    
    # Calculate basic metrics (assuming we have ground truth for evaluation)
    # For now, we'll calculate prediction distribution and confidence metrics
    prediction_distribution = pd.Series(final_predictions).value_counts().to_dict()
    mean_confidence = np.mean(confidence_scores)
    high_confidence_count = np.sum(confidence_scores > 0.7)

    # NEW: Calculate accuracy if ground truth is available
    accuracy = None
    if 'phase' in original_data.columns:
        accuracy = accuracy_score(original_data['phase'], final_predictions)

    results = {
        'prior_weight': prior_weight,
        'prediction_distribution': prediction_distribution,
        'mean_confidence': mean_confidence,
        'high_confidence_count': high_confidence_count,
        'total_predictions': len(final_predictions),
        'base_predictions': base_predictions,
        'prior_predictions': prior_predictions,
        'final_predictions': final_predictions,
        'confidence_scores': confidence_scores,
        'accuracy': accuracy  # NEW: add accuracy to results
    }
    
    return results


def compare_prior_weights(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare model performance across different prior weights.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame: Comparison results
    """
    logger.info("Starting prior weight comparison...")
    
    # Load and preprocess data
    data_path = config['data']['unlabeled_data_path']
    original_data = pd.read_csv(data_path)
    
    # Load the model bundle to get the preprocessor
    model_bundle = load_model_bundle(config['data']['model_path'])
    preprocessor = model_bundle.preprocessor
    X = preprocess_unlabeled_data(original_data, preprocessor)
    
    # Define prior weights to test
    prior_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Test each prior weight
    results_list = []
    for weight in prior_weights:
        try:
            results = evaluate_prior_weight(
                config['data']['model_path'], 
                X, 
                original_data, 
                weight, 
                config
            )
            results_list.append(results)
            logger.info(f"Prior weight {weight}: Mean confidence = {results['mean_confidence']:.3f}")
        except Exception as e:
            logger.error(f"Error testing prior weight {weight}: {str(e)}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)
    
    # NEW: Find best accuracy if available
    if 'accuracy' in comparison_df.columns and comparison_df['accuracy'].notnull().any():
        best_acc_idx = comparison_df['accuracy'].idxmax()
        best_acc_weight = comparison_df.loc[best_acc_idx, 'prior_weight']
        best_acc_score = comparison_df.loc[best_acc_idx, 'accuracy']
        logger.info(f"Best accuracy: {best_acc_score:.4f} at prior weight {best_acc_weight}")
        comparison_df.attrs['best_accuracy'] = best_acc_score
        comparison_df.attrs['best_accuracy_weight'] = best_acc_weight
    else:
        comparison_df.attrs['best_accuracy'] = None
        comparison_df.attrs['best_accuracy_weight'] = None

    return comparison_df


def plot_prior_weight_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create plots comparing performance across prior weights.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Mean confidence vs prior weight
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(comparison_df['prior_weight'], comparison_df['mean_confidence'], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Prior Weight')
    plt.ylabel('Mean Confidence')
    plt.title('Mean Prediction Confidence vs Prior Weight')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: High confidence predictions vs prior weight
    plt.subplot(2, 2, 2)
    plt.plot(comparison_df['prior_weight'], comparison_df['high_confidence_count'], 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Prior Weight')
    plt.ylabel('High Confidence Predictions (>0.7)')
    plt.title('High Confidence Predictions vs Prior Weight')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Prediction distribution heatmap
    plt.subplot(2, 2, 3)
    phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
    distribution_matrix = []
    
    for _, row in comparison_df.iterrows():
        dist = row['prediction_distribution']
        row_dist = [dist.get(phase, 0) for phase in phases]
        distribution_matrix.append(row_dist)
    
    distribution_matrix = np.array(distribution_matrix)
    im = plt.imshow(distribution_matrix.T, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Number of Predictions')
    plt.xlabel('Prior Weight Index')
    plt.ylabel('Phase')
    plt.xticks(range(len(comparison_df)), [f"{w:.1f}" for w in comparison_df['prior_weight']], rotation=45)
    plt.yticks(range(len(phases)), phases)
    plt.title('Prediction Distribution Heatmap')
    
    # Plot 4: Confidence distribution
    plt.subplot(2, 2, 4)
    for i, weight in enumerate(comparison_df['prior_weight']):
        confidence_scores = comparison_df.iloc[i]['confidence_scores']
        plt.hist(confidence_scores, bins=20, alpha=0.5, label=f'Weight {weight:.1f}')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution by Prior Weight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'prior_weight_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plots saved to: {plot_path}")


def save_comparison_results(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save comparison results to files.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'prior_weight_comparison.csv')
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"Detailed results saved to: {results_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'prior_weight_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("PRIOR WEIGHT COMPARISON SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Best performing weights:\n")
        f.write("-" * 20 + "\n")
        
        # Find best confidence
        best_conf_idx = comparison_df['mean_confidence'].idxmax()
        best_conf_weight = comparison_df.loc[best_conf_idx, 'prior_weight']
        best_conf_score = comparison_df.loc[best_conf_idx, 'mean_confidence']
        f.write(f"Highest mean confidence: {best_conf_weight:.1f} (score: {best_conf_score:.3f})\n")
        
        # Find most high-confidence predictions
        best_high_conf_idx = comparison_df['high_confidence_count'].idxmax()
        best_high_conf_weight = comparison_df.loc[best_high_conf_idx, 'prior_weight']
        best_high_conf_count = comparison_df.loc[best_high_conf_idx, 'high_confidence_count']
        f.write(f"Most high-confidence predictions: {best_high_conf_weight:.1f} (count: {best_high_conf_count})\n")

        # NEW: Best accuracy
        if 'accuracy' in comparison_df.columns and comparison_df['accuracy'].notnull().any():
            best_acc_idx = comparison_df['accuracy'].idxmax()
            best_acc_weight = comparison_df.loc[best_acc_idx, 'prior_weight']
            best_acc_score = comparison_df.loc[best_acc_idx, 'accuracy']
            f.write(f"Best accuracy: {best_acc_weight:.1f} (accuracy: {best_acc_score:.4f})\n")
        else:
            f.write("No accuracy calculated (no ground truth labels).\n")
        
        f.write("\nDetailed results:\n")
        f.write("-" * 15 + "\n")
        for _, row in comparison_df.iterrows():
            f.write(f"Weight {row['prior_weight']:.1f}: ")
            f.write(f"Confidence={row['mean_confidence']:.3f}, ")
            f.write(f"High-conf={row['high_confidence_count']}")
            if 'accuracy' in row and row['accuracy'] is not None:
                f.write(f", Accuracy={row['accuracy']:.4f}")
            f.write("\n")
    
    logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main function to run prior weight comparison."""
    # Load configuration
    config = load_config('config/prediction_config.yaml')
    
    # Create output directory
    output_dir = os.path.join(config['output']['results_dir'], 'prior_weight_comparison')
    
    # Run comparison
    comparison_df = compare_prior_weights(config)
    
    # Create plots
    plot_prior_weight_comparison(comparison_df, output_dir)
    
    # Save results
    save_comparison_results(comparison_df, output_dir)
    
    # Print summary
    print("\n=== PRIOR WEIGHT COMPARISON SUMMARY ===")
    print(f"Tested {len(comparison_df)} different prior weights")
    print(f"Best mean confidence: {comparison_df['mean_confidence'].max():.3f}")
    print(f"Best high-confidence count: {comparison_df['high_confidence_count'].max()}")
    # NEW: Print best accuracy if available
    if 'accuracy' in comparison_df.columns and comparison_df['accuracy'].notnull().any():
        best_acc_idx = comparison_df['accuracy'].idxmax()
        best_acc_weight = comparison_df.loc[best_acc_idx, 'prior_weight']
        best_acc_score = comparison_df.loc[best_acc_idx, 'accuracy']
        print(f"Best accuracy: {best_acc_score:.4f} at prior weight {best_acc_weight}")
    else:
        print("No accuracy calculated (no ground truth labels).")
    print(f"\nResults saved to: {output_dir}")
    
    logger.info("Prior weight comparison completed!")


if __name__ == '__main__':
    main() 