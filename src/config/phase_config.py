import numpy as np

# Phase durations in days (mean, SD) from Gloe
# Original proportions maintained but scaled to achieve ~28 day cycle
PHASE_DURATIONS = {
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

def get_phase_duration(phase):
    """
    Get the duration of a phase by sampling from its distribution.
    
    Args:
        phase (str): Phase name
    
    Returns:
        int: Duration in days (rounded to nearest integer)
    """
    params = PHASE_DURATIONS[phase]
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

def generate_phase_durations(sd_multiplier=1.0):
    """
    Generate phase durations for a complete menstrual cycle.
    
    Args:
        sd_multiplier (float): Multiplier for standard deviation to adjust variability
    
    Returns:
        dict: Dictionary mapping phase names to their durations in days
    """
    durations = {}
    for phase, params in PHASE_DURATIONS.items():
        # Apply the sd_multiplier to adjust variability
        duration = np.random.normal(params['mean'], params['sd'] * sd_multiplier)
        durations[phase] = max(1, round(duration))  # Ensure at least 1 day
    return durations 