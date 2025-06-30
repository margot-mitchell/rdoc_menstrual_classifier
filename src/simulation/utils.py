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
    phase_durations = load_phase_durations_from_config(config_path)
    
    durations = {}
    for phase, params in phase_durations.items():
        # Apply the sd_multiplier to adjust variability
        duration = np.random.normal(params['mean'], params['sd'] * sd_multiplier)
        durations[phase] = max(1, round(duration))  # Ensure at least 1 day
    return durations 

def load_hormone_distributions_from_config(config_path='config/simulation_config.yaml'):
    """
    Load hormone distributions from YAML config file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('hormones', {})

def get_estradiol_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    hormone_distributions = load_hormone_distributions_from_config(config_path)
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day, load_phase_durations_from_config(config_path))
    return hormone_distributions.get('estradiol', {}).get(phase)

def get_progesterone_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    hormone_distributions = load_hormone_distributions_from_config(config_path)
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day, load_phase_durations_from_config(config_path))
    return hormone_distributions.get('progesterone', {}).get(phase)

def get_testosterone_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    hormone_distributions = load_hormone_distributions_from_config(config_path)
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day, load_phase_durations_from_config(config_path))
    return hormone_distributions.get('testosterone', {}).get(phase)

def sample_lognormal_from_distribution(dist_params):
    mu = np.log(dist_params['mean']**2 / np.sqrt(dist_params['mean']**2 + dist_params['sd']**2))
    sigma = np.sqrt(np.log(1 + dist_params['sd']**2 / dist_params['mean']**2))
    value = np.random.lognormal(mu, sigma)
    value = max(dist_params['min'], min(dist_params['max'], value))
    return value

def sample_baseline_values(config_path='config/simulation_config.yaml'):
    hormone_distributions = load_hormone_distributions_from_config(config_path)
    return {
        'estradiol': sample_lognormal_from_distribution(hormone_distributions['estradiol']['perimenstruation']),
        'progesterone': sample_lognormal_from_distribution(hormone_distributions['progesterone']['perimenstruation']),
        'testosterone': sample_lognormal_from_distribution(hormone_distributions['testosterone']['perimenstruation'])
    }

def get_subphase_boundaries(phase_durations):
    boundaries = {}
    current_day = 0
    for phase, duration in phase_durations.items():
        start_day = current_day + 1
        end_day = current_day + duration
        boundaries[phase] = (start_day, end_day)
        current_day = end_day
    return boundaries

def generate_progesterone_value(cycle_day, baseline, phase_durations):
    boundaries = get_subphase_boundaries(phase_durations)
    mid_late_start, mid_late_end = boundaries['mid_late_luteal']
    mid_late_duration = mid_late_end - mid_late_start + 1
    peak_day = mid_late_start + (mid_late_duration // 2)
    rising_start = peak_day * 0.6
    falling_end = phase_durations.get('mid_late_luteal', 8) + mid_late_start
    if cycle_day <= rising_start:
        value = baseline + np.random.normal(0, 0.5 * baseline * 0.1)
    elif rising_start < cycle_day <= peak_day:
        peak = 4 * baseline
        progress = (cycle_day - rising_start) / (peak_day - rising_start)
        value = baseline + (peak - baseline) * progress
        value += np.random.normal(0, 0.1 * baseline)
    elif peak_day < cycle_day <= falling_end:
        peak = 4 * baseline
        progress = (cycle_day - peak_day) / (falling_end - peak_day)
        value = peak - (peak - baseline) * progress
        value += np.random.normal(0, 0.1 * baseline)
    else:
        value = baseline
    return max(0, value)

def generate_estradiol_value(cycle_day, baseline, phase_durations):
    boundaries = get_subphase_boundaries(phase_durations)
    periov_start, periov_end = boundaries['periovulation']
    primary_peak_day = (periov_start + periov_end) // 2
    mid_late_start, mid_late_end = boundaries['mid_late_luteal']
    mid_late_duration = mid_late_end - mid_late_start + 1
    secondary_peak_day = mid_late_start + (mid_late_duration // 2)
    early_phase_end = primary_peak_day * 0.4
    primary_peak_end = primary_peak_day + (secondary_peak_day - primary_peak_day) * 0.3
    secondary_peak_end = secondary_peak_day + (phase_durations.get('mid_late_luteal', 8) - mid_late_duration // 2)
    if cycle_day <= early_phase_end:
        value = baseline + np.random.normal(0, 0.5 * baseline * 0.1)
    elif early_phase_end < cycle_day <= primary_peak_day:
        primary_peak = 5.0 * baseline
        progress = (cycle_day - early_phase_end) / (primary_peak_day - early_phase_end)
        value = baseline + (primary_peak - baseline) * progress
        value += np.random.normal(0, 0.1 * baseline)
    elif primary_peak_day < cycle_day <= primary_peak_end:
        primary_peak = 5.0 * baseline
        inflection = baseline
        progress = (cycle_day - primary_peak_day) / (primary_peak_end - primary_peak_day)
        value = primary_peak - (primary_peak - inflection) * progress
        value += np.random.normal(0, 0.1 * baseline)
    elif primary_peak_end < cycle_day <= secondary_peak_day:
        trough = baseline
        secondary_peak = 2.5 * baseline
        progress = (cycle_day - primary_peak_end) / (secondary_peak_day - primary_peak_end)
        value = trough + (secondary_peak - trough) * progress
        value += np.random.normal(0, 0.1 * baseline)
    elif secondary_peak_day < cycle_day <= sum(phase_durations.values()):
        secondary_peak = 2.5 * baseline
        progress = (cycle_day - secondary_peak_day) / (sum(phase_durations.values()) - secondary_peak_day)
        value = secondary_peak - (secondary_peak - baseline) * progress
        value += np.random.normal(0, 0.1 * baseline)
    else:
        value = baseline
    return max(0, value)

def generate_testosterone_value(cycle_day, baseline, phase_durations):
    boundaries = get_subphase_boundaries(phase_durations)
    periov_start, periov_end = boundaries['periovulation']
    periov_center = (periov_start + periov_end) // 2
    early_luteal_start, early_luteal_end = boundaries['early_luteal']
    mid_late_start, mid_late_end = boundaries['mid_late_luteal']
    luteal_center = (early_luteal_start + mid_late_end) // 2
    periov_distance = abs(cycle_day - periov_center)
    luteal_distance = abs(cycle_day - luteal_center)
    value = baseline
    if periov_distance <= 2:
        periov_factor = 1.0 - (periov_distance / 2.0)
        value += baseline * 0.2 * periov_factor
    if luteal_distance <= 5:
        luteal_factor = 1.0 - (luteal_distance / 5.0)
        value -= baseline * 0.15 * luteal_factor
    value += np.random.normal(0, 0.1 * baseline)
    return max(0, value) 