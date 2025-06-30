import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_menstrual_patterns(n_subjects):
    """
    Generate menstrual patterns for subjects.
    
    Args:
        n_subjects (int): Number of subjects
        
    Returns:
        np.ndarray: Array of menstrual patterns
    """
    pattern_options = [
        'Extremely regular (no more than 1-2 days before or after expected)',
        'Very regular (within 3-4 days)',
        'Regular (within 5-7 days)',
    ]
    return np.random.choice(pattern_options, size=n_subjects)

def get_start_dates(n_subjects):
    """
    Generate random start dates for subjects.
    
    Args:
        n_subjects (int): Number of subjects
        
    Returns:
        list: List of start dates
    """
    start_dates = pd.date_range(start='2025-01-01', end='2026-12-31', periods=n_subjects).date.tolist()
    np.random.shuffle(start_dates)
    return start_dates

def get_phase_duration_multiplier(pattern):
    """
    Get phase duration variability multiplier based on menstrual pattern.
    
    Args:
        pattern (str): Menstrual pattern
        
    Returns:
        float: Phase duration multiplier
    """
    if pattern == 'Extremely regular (no more than 1-2 days before or after expected)':
        return 0.3
    elif pattern == 'Very regular (within 3-4 days)':
        return 0.6
    else:  # 'Regular (within 5-7 days)'
        return 1.0

def initialize_subject(subject_id, pattern, start_date, phase_duration_sd_multiplier, config_path='config/simulation_config.yaml'):
    """
    Initialize a subject with their basic parameters.
    
    Args:
        subject_id (int): Subject ID
        pattern (str): Menstrual pattern
        start_date (datetime): Start date
        phase_duration_sd_multiplier (float): Phase duration multiplier
        config_path (str): Path to config file
        
    Returns:
        dict: Subject parameters
    """
    from src.simulation.utils import sample_baseline_values
    from src.simulation.utils import generate_phase_durations
    
    baselines = sample_baseline_values(config_path)
    phase_durations = generate_phase_durations(sd_multiplier=phase_duration_sd_multiplier, config_path=config_path)
    total_cycle_length = sum(phase_durations.values())
    start_idx = np.random.randint(0, total_cycle_length)
    
    return {
        'subject_id': subject_id,
        'pattern': pattern,
        'start_date': start_date,
        'baselines': baselines,
        'phase_durations': phase_durations,
        'total_cycle_length': total_cycle_length,
        'start_idx': start_idx
    } 