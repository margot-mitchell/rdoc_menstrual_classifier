import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.simulation.subject import (
    get_menstrual_patterns,
    get_start_dates,
    get_phase_duration_multiplier,
    initialize_subject
)
from src.simulation.hormones import generate_all_hormone_data
from src.simulation.periods import generate_all_period_data
from src.simulation.visualization import plot_hormone_cycles
from src.simulation.metrics import calculate_metrics

def simulate_hormone_and_period_data(n_subjects=100, n_hormone_samples=70, n_period_days=150):
    """
    Simulate hormone and period data for multiple subjects.
    
    Args:
        n_subjects (int): Number of subjects to simulate
        n_hormone_samples (int): Number of days of hormone data to generate
        n_period_days (int): Number of days of period data to generate
        
    Returns:
        tuple: (hormone_data, period_data, menstrual_patterns) DataFrames containing simulated data and patterns
    """
    # Generate subject patterns and start dates
    patterns = get_menstrual_patterns(n_subjects)
    start_dates = get_start_dates(n_subjects)
    
    # Initialize all subjects
    subjects = []
    for subject_id in range(n_subjects):
        pattern = patterns[subject_id]
        start_date = start_dates[subject_id]
        phase_duration_sd_multiplier = get_phase_duration_multiplier(pattern)
        
        subject_params = initialize_subject(
            subject_id=subject_id,
            pattern=pattern,
            start_date=start_date,
            phase_duration_sd_multiplier=phase_duration_sd_multiplier
        )
        subjects.append(subject_params)
    
    # Generate hormone data
    hormone_df = generate_all_hormone_data(subjects, n_hormone_samples)
    
    # Generate period data
    period_df = generate_all_period_data(subjects, n_period_days)
    
    # Create DataFrame of subject patterns
    pattern_df = pd.DataFrame({
        'subject_id': range(n_subjects),
        'menstrual_pattern': patterns
    })
    
    return hormone_df, period_df, pattern_df

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Simulate data
    hormone_df, period_df, pattern_df = simulate_hormone_and_period_data()
    
    # Save data to CSV files
    hormone_df.to_csv('output/full_hormone_data_labeled.csv', index=False)
    period_df.to_csv('output/period_sleep_data.csv', index=False)
    pattern_df.to_csv('output/menstrual_patterns.csv', index=False)
    
    # Create unlabeled hormone data (10 samples per subject)
    unlabeled_df = pd.concat([
        group.iloc[::7][:10]  # Take every 7th sample, up to 10 samples
        for _, group in hormone_df.groupby('subject_id', group_keys=False)
    ])
    unlabeled_df.to_csv('output/hormone_data_unlabeled.csv', index=False)
    
    # Generate survey responses
    survey_df = pd.DataFrame({
        'subject_id': range(len(pattern_df)),
        'menstrual_pattern': pattern_df['menstrual_pattern'],
        'cycle_length': np.random.normal(28, 2, len(pattern_df)).round().astype(int)
    })
    survey_df.to_csv('output/survey_responses.csv', index=False)
    
    # Calculate and save metrics
    metrics = calculate_metrics(hormone_df)
    with open('output/simulated_metrics.txt', 'w') as f:
        f.write(metrics)
    
    # Plot hormone cycles for first 5 subjects
    for subject_id in range(5):
        plot_hormone_cycles(
            hormone_df,
            subject_id=subject_id,
            output_path=f'output/hormone_cycles_subject_{subject_id}.png'
        )
    
    print("Data generation complete. Files saved in 'output' directory.")

if __name__ == '__main__':
    main() 