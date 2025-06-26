import numpy as np
import yaml
import os

def load_phase_durations_from_config(config_path='config/simulation_config.yaml'):
    """
    Load phase durations from YAML config file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Phase durations dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('phase_durations', {})

# Fallback hardcoded values (only used if config file is missing)
FALLBACK_PHASE_DURATIONS = {
    'perimenstruation': {
        'mean': 3.0,  # ~10.7% of cycle
        'sd': 1.0     # Scaled proportionally
    },
    'mid_follicular': {
        'mean': 8.4,  # ~30% of cycle
        'sd': 1.8     # Scaled proportionally
    },
    'periovulation': {
        'mean': 3.5,  # ~12.5% of cycle
        'sd': 1.0     # Scaled proportionally
    },
    'early_luteal': {
        'mean': 4.8,  # ~17.1% of cycle
        'sd': 2.0     # Scaled proportionally
    },
    'mid_late_luteal': {
        'mean': 8.3,  # ~29.6% of cycle
        'sd': 3.0     # Scaled proportionally
    }
}

def get_phase_duration(phase, phase_durations=None):
    """
    Get the duration of a phase by sampling from its distribution.
    
    Args:
        phase (str): Phase name
        phase_durations (dict): Phase durations dictionary (optional)
    
    Returns:
        int: Duration in days (rounded to nearest integer)
    """
    if phase_durations is None:
        try:
            phase_durations = load_phase_durations_from_config()
        except FileNotFoundError:
            phase_durations = FALLBACK_PHASE_DURATIONS
    
    params = phase_durations[phase]
    duration = np.random.normal(params['mean'], params['sd'])
    return max(1, round(duration))  # Ensure at least 1 day

def get_phase_from_cycle_day(cycle_day, phase_durations):
    """
    Determine the menstrual cycle phase based on cycle day and phase durations.
    
    Args:
        cycle_day (int): Day of the menstrual cycle
        phase_durations (dict): Dictionary of phase durations
    
    Returns:
        str: Phase name
    """
    current_day = 0
    for phase, duration in phase_durations.items():
        current_day += duration
        if cycle_day <= current_day:
            return phase
    return 'mid_late_luteal'  # Default to last phase if something goes wrong

def generate_phase_durations(sd_multiplier=1.0, config_path='config/simulation_config.yaml'):
    """
    Generate phase durations for a complete menstrual cycle.
    
    Args:
        sd_multiplier (float): Multiplier for standard deviation to adjust variability
        config_path (str): Path to config file with phase durations
    
    Returns:
        dict: Dictionary mapping phase names to their durations in days
    """
    # Load phase durations from config
    try:
        phase_durations = load_phase_durations_from_config(config_path)
    except FileNotFoundError:
        phase_durations = FALLBACK_PHASE_DURATIONS
    
    durations = {}
    for phase, params in phase_durations.items():
        # Apply the sd_multiplier to adjust variability
        duration = np.random.normal(params['mean'], params['sd'] * sd_multiplier)
        durations[phase] = max(1, round(duration))  # Ensure at least 1 day
    return durations 