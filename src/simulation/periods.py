import numpy as np
import pandas as pd
from datetime import timedelta
from src.config.phase_config import get_phase_from_cycle_day

def generate_period_data(subject_params, n_period_days):
    """
    Generate period and sleep data for a single subject.
    
    Args:
        subject_params (dict): Subject parameters from initialize_subject
        n_period_days (int): Number of days of period data to generate
        
    Returns:
        list: List of period data points
    """
    period_data = []
    
    for day in range(n_period_days):
        cycle_day = ((day + subject_params['start_idx']) % subject_params['total_cycle_length']) + 1
        current_phase = get_phase_from_cycle_day(cycle_day, subject_params['phase_durations'])
        
        # Determine if it's a period day (perimenstruation phase)
        is_period = current_phase == 'perimenstruation'
        
        # Generate flow value
        flow = 'N/A'
        if is_period:
            phase_day = cycle_day - sum(d for p, d in subject_params['phase_durations'].items() if p != 'perimenstruation')
            if phase_day == 1:
                flow = 'Heavy'
            else:
                flow = np.random.choice(['Light', 'Medium'])
        
        # Generate spotting value
        spotting = 'No'
        if not is_period:
            if current_phase == 'periovulation':
                spotting = 'Yes' if np.random.random() < 0.3 else 'No'  # 30% chance
            elif current_phase == 'mid_late_luteal' and cycle_day == subject_params['total_cycle_length']:
                spotting = 'Yes' if np.random.random() < 0.2 else 'No'  # 20% chance
        
        # Generate sleep data
        sleep_hours = np.random.uniform(4, 11)
        
        period_data.append({
            'subject_id': subject_params['subject_id'],
            'date': (subject_params['start_date'] + timedelta(days=day)).strftime('%Y-%m-%d'),
            'cycle_day': cycle_day,
            'period': 'Yes' if is_period else 'No',
            'flow': flow,
            'spotting': spotting,
            'sleep_hours': sleep_hours,
            'phase': current_phase
        })
    
    return period_data

def generate_all_period_data(subjects, n_period_days):
    """
    Generate period data for all subjects.
    
    Args:
        subjects (list): List of subject parameters
        n_period_days (int): Number of days of period data to generate
        
    Returns:
        pd.DataFrame: DataFrame containing period data for all subjects
    """
    all_period_data = []
    
    for subject_params in subjects:
        subject_period_data = generate_period_data(subject_params, n_period_days)
        all_period_data.extend(subject_period_data)
    
    return pd.DataFrame(all_period_data) 