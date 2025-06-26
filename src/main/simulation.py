#!/usr/bin/env python3
"""
Main simulation script for generating hormone and period data.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from src.utils.data_loader import load_config


def simulate_hormone_and_period_data(config):
    """
    Simulate hormone and period data for multiple subjects.
    
    Args:
        config (dict): Configuration dictionary loaded from YAML
        
    Returns:
        tuple: (hormone_data, period_data, menstrual_patterns, survey_data) DataFrames containing simulated data and patterns
    """
    # Extract parameters from config
    n_subjects = config['simulation']['n_subjects']
    n_hormone_samples = config['simulation']['n_hormone_samples']
    n_period_days = config['simulation']['n_period_days']
    
    # Set random seed if specified
    if 'random_seed' in config['simulation']:
        np.random.seed(config['simulation']['random_seed'])
    
    # Generate subject patterns and start dates
    patterns = get_menstrual_patterns(n_subjects)
    start_dates = get_start_dates(n_subjects)
    
    # Initialize all subjects with realistic phase durations first
    subjects = []
    survey_data = []
    
    for subject_id in range(n_subjects):
        pattern = patterns[subject_id]
        start_date = start_dates[subject_id]
        phase_duration_sd_multiplier = get_phase_duration_multiplier(pattern)
        
        # First, get realistic phase durations (this gives us the proportions)
        subject_params = initialize_subject(
            subject_id=subject_id,
            pattern=pattern,
            start_date=start_date,
            phase_duration_sd_multiplier=phase_duration_sd_multiplier,
            config_path='config/simulation_config.yaml'
        )
        
        # Sample cycle length from distribution for NC women (from Fehring)
        cycle_config = config['cycle_length']
        cycle_length = int(round(np.clip(
            np.random.normal(cycle_config['mean'], cycle_config['sd']), 
            cycle_config['min'], 
            cycle_config['max']
        )))
        
        # Scale the phase durations to match the cycle length while preserving proportions
        original_phase_durations = subject_params['phase_durations']
        original_total = sum(original_phase_durations.values())
        scale_factor = cycle_length / original_total
        
        # Scale phase durations proportionally
        adjusted_phase_durations = {}
        for phase, duration in original_phase_durations.items():
            adjusted_phase_durations[phase] = int(round(duration * scale_factor))
        
        # Ensure the total matches exactly by adjusting the largest phase
        total_adjusted = sum(adjusted_phase_durations.values())
        if total_adjusted != cycle_length:
            # Find the phase with the largest duration and adjust it
            largest_phase = max(adjusted_phase_durations.items(), key=lambda x: x[1])[0]
            adjusted_phase_durations[largest_phase] += (cycle_length - total_adjusted)
        
        # Update subject parameters with adjusted phase durations
        subject_params['phase_durations'] = adjusted_phase_durations
        subject_params['total_cycle_length'] = cycle_length
        
        subjects.append(subject_params)
        
        # Store survey data
        survey_data.append({
            'subject_id': subject_id,
            'menstrual_pattern': pattern,
            'cycle_length': cycle_length
        })
    
    # Generate hormone data
    hormone_df = generate_all_hormone_data(subjects, n_hormone_samples)
    
    # Generate period data
    period_df = generate_all_period_data(subjects, n_period_days)
    
    # Create DataFrame of subject patterns
    pattern_df = pd.DataFrame({
        'subject_id': range(n_subjects),
        'menstrual_pattern': patterns
    })
    
    # Create survey DataFrame
    survey_df = pd.DataFrame(survey_data)
    
    return hormone_df, period_df, pattern_df, survey_df


def generate_survey_responses(period_df, survey_df):
    """
    Generate survey responses using the survey data from simulation.
    
    Args:
        period_df (pd.DataFrame): Period data
        survey_df (pd.DataFrame): Survey data with basic info
        
    Returns:
        pd.DataFrame: Survey responses with dates
    """
    survey_data = []
    for _, survey_row in survey_df.iterrows():
        subject_id = survey_row['subject_id']
        cycle_length = survey_row['cycle_length']
        menstrual_pattern = survey_row['menstrual_pattern']
        
        # Get subject's period data and convert dates to datetime
        subject_periods = period_df[period_df['subject_id'] == subject_id].copy()
        subject_periods['date'] = pd.to_datetime(subject_periods['date'])
        subject_periods = subject_periods.sort_values('date')
        
        # Find actual periods (where period == 'Yes')
        actual_periods = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
        
        if len(actual_periods) < 1:
            # If no periods, use default values
            date_of_last_period = pd.Timestamp('2025-01-01')
        else:
            # Set date_of_response to a random date between rows 14-21
            response_idx = np.random.randint(14, 22)  # 14-21 inclusive
            date_of_response = subject_periods.iloc[response_idx]['date']
            
            # Find periods before the response date
            periods_before_response = [p for p in actual_periods if p <= date_of_response]
            
            if periods_before_response:
                # Use the first day of the most recent period before response date
                date_of_last_period = min(periods_before_response)
            else:
                # If no periods before response, calculate date_of_last_period as cycle_length days before the first period
                first_period_date = actual_periods[0]
                date_of_last_period = first_period_date - pd.Timedelta(days=cycle_length)
        
        survey_data.append({
            'subject_id': subject_id,
            'menstrual_pattern': menstrual_pattern,
            'cycle_length': cycle_length,
            'date_of_response': date_of_response.strftime('%Y-%m-%d'),
            'date_of_last_period': date_of_last_period.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(survey_data)


def main():
    """Main function to run the simulation."""
    # Load configuration
    config = load_config('config/simulation_config.yaml')
    
    # Create output directories
    output_config = config['output']
    os.makedirs(output_config['data_dir'], exist_ok=True)
    os.makedirs(output_config['figures_dir'], exist_ok=True)
    os.makedirs(output_config['reports_dir'], exist_ok=True)
    
    print("Starting hormone and period data simulation...")
    
    # Simulate data
    hormone_df, period_df, pattern_df, survey_df = simulate_hormone_and_period_data(config)
    
    # Save data to CSV files
    if output_config['save_hormone_data']:
        hormone_df.to_csv(os.path.join(output_config['data_dir'], 'full_hormone_data_labeled.csv'), index=False)
    
    if output_config['save_period_data']:
        # Remove phase and cycle_day columns from period data
        period_df_clean = period_df.drop(columns=['phase', 'cycle_day'])
        period_df_clean.to_csv(os.path.join(output_config['data_dir'], 'period_sleep_data.csv'), index=False)
    
    if output_config['save_patterns']:
        pattern_df.to_csv(os.path.join(output_config['data_dir'], 'menstrual_patterns.csv'), index=False)
    
    # Create unlabeled hormone data (7 samples per subject)
    # Remove cycle_day and phase from unlabeled data to force model to learn cycle position
    unlabeled_df = pd.concat([
        group.iloc[::10][:7].drop(columns=['cycle_day', 'phase'])  # Take every 10th sample, up to 7 samples
        for _, group in hormone_df.groupby('subject_id', group_keys=False)
    ])
    unlabeled_df.to_csv(os.path.join(output_config['data_dir'], 'hormone_data_unlabeled.csv'), index=False)
    
    # Generate survey responses
    if output_config['save_survey_data']:
        survey_responses_df = generate_survey_responses(period_df, survey_df)
        survey_responses_df.to_csv(os.path.join(output_config['data_dir'], 'survey_responses.csv'), index=False)
    
    # Calculate and save metrics
    metrics = calculate_metrics(hormone_df)
    with open(os.path.join(output_config['reports_dir'], 'simulated_metrics.txt'), 'w') as f:
        f.write(metrics)
    
    # Plot hormone cycles
    if output_config['plot_hormone_cycles']:
        n_plot_subjects = output_config.get('n_plot_subjects', 5)
        for subject_id in range(n_plot_subjects):
            plot_hormone_cycles(
                hormone_df,
                subject_id=subject_id,
                output_path=os.path.join(output_config['figures_dir'], f'hormone_cycles_subject_{subject_id}.png')
            )
    
    print("Data generation complete. Files saved in output directories.")


if __name__ == '__main__':
    main() 