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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config, preprocess_unlabeled_data
from src.utils.model_utils import load_model, predict_with_model, predict_proba_with_model, get_model_info
from src.temporal_models.rule_based_prior import RuleBasedPrior
from src.utils.evaluator import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    model_info = get_model_info(model_path)
    
    # Extract original features for model prediction
    original_features = ['estradiol', 'progesterone', 'testosterone']
    X_for_model = X[original_features]
    
    # Make base model predictions
    base_predictions = predict_with_model(model_path, X_for_model)
    base_probabilities = predict_proba_with_model(model_path, X_for_model)
    
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
        # Weighted combination
        combined_predictions = []
        combined_probs = []
        
        for i in range(len(base_predictions)):
            # Simple voting scheme for predictions
            if np.random.random() < prior_weight:
                combined_predictions.append(prior_predictions[i])
            else:
                combined_predictions.append(base_predictions[i])
            
            # Weighted average for probabilities
            combined_prob = (1 - prior_weight) * base_probabilities[i] + prior_weight * prior_probs[i]
            combined_probs.append(combined_prob)
        
        final_predictions = np.array(combined_predictions)
        final_probabilities = np.array(combined_probs)
    
    # Calculate confidence scores
    confidence_scores = np.max(final_probabilities, axis=1)
    
    # Calculate basic metrics (assuming we have ground truth for evaluation)
    # For now, we'll calculate prediction distribution and confidence metrics
    prediction_distribution = pd.Series(final_predictions).value_counts().to_dict()
    mean_confidence = np.mean(confidence_scores)
    high_confidence_count = np.sum(confidence_scores > 0.7)
    
    results = {
        'prior_weight': prior_weight,
        'prediction_distribution': prediction_distribution,
        'mean_confidence': mean_confidence,
        'high_confidence_count': high_confidence_count,
        'total_predictions': len(final_predictions),
        'base_predictions': base_predictions,
        'prior_predictions': prior_predictions,
        'final_predictions': final_predictions,
        'confidence_scores': confidence_scores
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
    X = preprocess_unlabeled_data(original_data)
    
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
        
        f.write("\nDetailed results:\n")
        f.write("-" * 15 + "\n")
        for _, row in comparison_df.iterrows():
            f.write(f"Weight {row['prior_weight']:.1f}: ")
            f.write(f"Confidence={row['mean_confidence']:.3f}, ")
            f.write(f"High-conf={row['high_confidence_count']}\n")
    
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
    print(f"\nResults saved to: {output_dir}")
    
    logger.info("Prior weight comparison completed!")


if __name__ == '__main__':
    main() 