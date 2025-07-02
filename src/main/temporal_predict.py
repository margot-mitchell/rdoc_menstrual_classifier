#!/usr/bin/env python3
"""
Temporal prediction script for making rule-based predictions using survey responses
and period data without requiring training.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.utils.data_loader import load_config
from src.classification.rule_based_prior import RuleBasedPrior

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_temporal_prediction(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run rule-based temporal prediction pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Prediction results
    """
    logger.info("Starting rule-based temporal prediction...")
    
    # Initialize rule-based prior model
    rule_prior = RuleBasedPrior(config)
    
    # Load data
    rule_prior.load_data()
    
    # Load hormone data for prediction
    data_config = config['data']
    hormone_data_path = data_config['unlabeled_data_path']
    
    if not os.path.exists(hormone_data_path):
        raise FileNotFoundError(f"Hormone data file not found: {hormone_data_path}")
    
    hormone_data = pd.read_csv(hormone_data_path)
    logger.info(f"Loaded hormone data: {len(hormone_data)} samples")
    
    # Make predictions
    predictions = rule_prior.predict_phases(hormone_data)
    
    # Get prior probabilities
    prior_probs = rule_prior.get_prior_probabilities(hormone_data)
    
    # Save predictions
    output_config = config['output']
    rule_prior.save_predictions(hormone_data, output_config['predictions_dir'])
    
    # Save prior probabilities
    phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
    proba_df = pd.DataFrame(prior_probs, columns=phases)
    proba_path = os.path.join(output_config['predictions_dir'], 'rule_based_prior_probabilities.csv')
    proba_df.to_csv(proba_path, index=False)
    logger.info(f"Prior probabilities saved to: {proba_path}")
    
    # Generate summary
    results = {
        'predictions': predictions,
        'prior_probabilities': prior_probs,
        'n_samples': len(hormone_data),
        'phase_distribution': pd.Series(predictions).value_counts().to_dict()
    }
    
    logger.info("Rule-based temporal prediction completed!")
    return results


def main():
    """Main function to run temporal prediction."""
    # Load configuration
    config = load_config('config/prediction_config.yaml')
    
    # Create output directories
    os.makedirs(config['output']['predictions_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    # Run temporal prediction
    results = run_temporal_prediction(config)
    
    # Print summary
    print("\n=== RULE-BASED TEMPORAL PREDICTION SUMMARY ===")
    print(f"Total predictions: {results['n_samples']}")
    print("\nPhase distribution:")
    for phase, count in results['phase_distribution'].items():
        percentage = (count / results['n_samples']) * 100
        print(f"  {phase}: {count} ({percentage:.1f}%)")
    
    print(f"\nPredictions saved to: {config['output']['predictions_dir']}")


if __name__ == '__main__':
    main() 