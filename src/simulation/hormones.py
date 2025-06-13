import pandas as pd
from datetime import timedelta
from src.config.hormone_config import (
    generate_estradiol_value,
    generate_progesterone_value,
    sample_lognormal_from_distribution,
    get_testosterone_distribution
)
from src.config.phase_config import get_phase_from_cycle_day

def generate_hormone_data(subject_params, n_hormone_samples):
    """
    Generate hormone data for a single subject.
    
    Args:
        subject_params (dict): Subject parameters from initialize_subject
        n_hormone_samples (int): Number of days of hormone data to generate
        
    Returns:
        list: List of hormone data points
    """
    hormone_data = []
    
    for day in range(n_hormone_samples):
        # Calculate cycle day (1 to total_cycle_length)
        cycle_day = ((day + subject_params['start_idx']) % subject_params['total_cycle_length']) + 1
        
        # Get current phase
        current_phase = get_phase_from_cycle_day(cycle_day, subject_params['phase_durations'])
        
        # Generate hormone values
        estradiol = generate_estradiol_value(cycle_day, subject_params['baselines']['estradiol'])
        progesterone = generate_progesterone_value(cycle_day, subject_params['baselines']['progesterone'])
        testosterone = sample_lognormal_from_distribution(get_testosterone_distribution(cycle_day=cycle_day))
        
        hormone_data.append({
            'subject_id': subject_params['subject_id'],
            'date': (subject_params['start_date'] + timedelta(days=day)).strftime('%Y-%m-%d'),
            'cycle_day': cycle_day,
            'estradiol': estradiol,
            'progesterone': progesterone,
            'testosterone': testosterone,
            'phase': current_phase
        })
    
    return hormone_data

def generate_all_hormone_data(subjects, n_hormone_samples):
    """
    Generate hormone data for all subjects.
    
    Args:
        subjects (list): List of subject parameters
        n_hormone_samples (int): Number of days of hormone data to generate
        
    Returns:
        pd.DataFrame: DataFrame containing hormone data for all subjects
    """
    all_hormone_data = []
    
    for subject_params in subjects:
        subject_hormone_data = generate_hormone_data(subject_params, n_hormone_samples)
        all_hormone_data.extend(subject_hormone_data)
    
    return pd.DataFrame(all_hormone_data) 