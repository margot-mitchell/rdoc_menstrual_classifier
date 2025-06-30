import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.simulation.utils import (
    generate_estradiol_value,
    generate_progesterone_value,
    generate_testosterone_value
)
from src.simulation.utils import get_phase_from_cycle_day

def generate_hormone_data(subject_params, n_samples):
    """
    Generate hormone data for a subject.
    
    Args:
        subject_params (dict): Subject parameters from initialize_subject
        n_samples (int): Number of hormone samples to generate
        
    Returns:
        pd.DataFrame: Hormone data
    """
    data = []
    current_date = subject_params['start_date']
    cycle_day = subject_params['start_idx'] + 1
    
    for sample in range(n_samples):
        current_phase = get_phase_from_cycle_day(cycle_day, subject_params['phase_durations'])
        
        # Generate hormone values
        estradiol = generate_estradiol_value(cycle_day, subject_params['baselines']['estradiol'], subject_params['phase_durations'])
        progesterone = generate_progesterone_value(cycle_day, subject_params['baselines']['progesterone'], subject_params['phase_durations'])
        testosterone = generate_testosterone_value(cycle_day, subject_params['baselines']['testosterone'], subject_params['phase_durations'])
        
        data.append({
            'subject_id': subject_params['subject_id'],
            'date': current_date.strftime('%Y-%m-%d'),
            'cycle_day': cycle_day,
            'phase': current_phase,
            'estradiol': estradiol,
            'progesterone': progesterone,
            'testosterone': testosterone
        })
        
        current_date += pd.Timedelta(days=1)
        cycle_day += 1
        
        # Reset cycle day when we reach the end of the cycle
        if cycle_day > subject_params['total_cycle_length']:
            cycle_day = 1
    
    return pd.DataFrame(data)

def generate_all_hormone_data(subjects, n_samples):
    """
    Generate hormone data for all subjects.
    
    Args:
        subjects (list): List of subject parameters
        n_samples (int): Number of hormone samples to generate
        
    Returns:
        pd.DataFrame: Combined hormone data for all subjects
    """
    all_data = []
    for subject_params in subjects:
        subject_data = generate_hormone_data(subject_params, n_samples)
        all_data.append(subject_data)
    
    return pd.concat(all_data, ignore_index=True) 