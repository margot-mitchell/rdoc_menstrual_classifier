import numpy as np
import pandas as pd
from src.simulation.utils import get_phase_from_cycle_day

def generate_period_data(subject_params, n_days):
    """
    Generate period data for a subject.
    
    Args:
        subject_params (dict): Subject parameters from initialize_subject
        n_days (int): Number of days to generate
        
    Returns:
        pd.DataFrame: Period data
    """
    data = []
    current_date = subject_params['start_date']
    cycle_day = subject_params['start_idx'] + 1
    
    for day in range(n_days):
        current_phase = get_phase_from_cycle_day(cycle_day, subject_params['phase_durations'])
        
        # Determine if it's a period day (perimenstruation phase)
        period = 'Yes' if current_phase == 'perimenstruation' else 'No'
        
        # Calculate phase day (day within the current phase)
        phase_day = cycle_day - sum(d for p, d in subject_params['phase_durations'].items() if p != 'perimenstruation')
        
        data.append({
            'subject_id': subject_params['subject_id'],
            'date': current_date.strftime('%Y-%m-%d'),
            'period': period,
            'phase': current_phase,
            'cycle_day': cycle_day
        })
        
        current_date += pd.Timedelta(days=1)
        cycle_day += 1
        
        # Reset cycle day when we reach the end of the cycle
        if cycle_day > subject_params['total_cycle_length']:
            cycle_day = 1
    
    return pd.DataFrame(data)

def generate_all_period_data(subjects, n_days):
    """
    Generate period data for all subjects.
    
    Args:
        subjects (list): List of subject parameters
        n_days (int): Number of days to generate
        
    Returns:
        pd.DataFrame: Combined period data for all subjects
    """
    all_data = []
    for subject_params in subjects:
        subject_data = generate_period_data(subject_params, n_days)
        all_data.append(subject_data)
    
    return pd.concat(all_data, ignore_index=True) 