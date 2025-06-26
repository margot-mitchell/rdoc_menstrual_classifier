import numpy as np
from scipy.stats import lognorm
import yaml
import os

def load_hormone_distributions_from_config(config_path='config/simulation_config.yaml'):
    """
    Load hormone distributions from YAML config file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Hormone distributions dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('hormones', {})

# Fallback hardcoded values (only used if config file is missing)
FALLBACK_ESTRADIOL_DISTRIBUTIONS = {
    'perimenstruation': {  # Days 1-5
        'min': 0.28,
        'max': 2.64,
        'mean': 1.28,
        'sd': 0.54
    },
    'mid_follicular': {    # Days 6-11
        'min': 0.28,
        'max': 2.64,
        'mean': 1.28,
        'sd': 0.54
    },
    'periovulation': {     # Days 12-14
        'min': 0.48,
        'max': 4.3,
        'mean': 2.0,
        'sd': 0.76
    },
    'early_luteal': {      # Days 15-21
        'min': 0.3,
        'max': 3.14,
        'mean': 1.59,
        'sd': 0.6
    },
    'mid_late_luteal': {   # Days 22-28
        'min': 0.3,
        'max': 2.63,
        'mean': 1.27,
        'sd': 0.57
    }
}

FALLBACK_PROGESTERONE_DISTRIBUTIONS = {
    'perimenstruation': {  # Days 1-5
        'min': 32.9,
        'max': 312.53,
        'mean': 119.96,
        'sd': 66.45
    },
    'mid_follicular': {    # Days 6-11
        'min': 32.9,
        'max': 312.53,
        'mean': 119.96,
        'sd': 66.45
    },
    'periovulation': {     # Days 12-14
        'min': 21.96,
        'max': 351.74,
        'mean': 149.78,
        'sd': 85.98
    },
    'early_luteal': {      # Days 15-21
        'min': 63.35,
        'max': 651.50,
        'mean': 283.01,
        'sd': 129.34
    },
    'mid_late_luteal': {   # Days 22-28
        'min': 15.76,
        'max': 386.08,
        'mean': 153.19,
        'sd': 88.5
    }
}

FALLBACK_TESTOSTERONE_DISTRIBUTIONS = {
    'perimenstruation': {  # Days 1-5
        'min': 126.7,
        'max': 146.5,
        'mean': 136.57,
        'sd': 78.0
    },
    'mid_follicular': {    # Days 6-11
        'min': 126.7,
        'max': 146.5,
        'mean': 136.57,
        'sd': 78.0
    },
    'periovulation': {     # Days 12-14
        'min': 126.7,
        'max': 146.5,
        'mean': 136.57,
        'sd': 78.0
    },
    'early_luteal': {      # Days 15-21
        'min': 126.7,
        'max': 146.5,
        'mean': 136.57,
        'sd': 78.0
    },
    'mid_late_luteal': {   # Days 22-28
        'min': 126.7,
        'max': 146.5,
        'mean': 136.57,
        'sd': 78.0
    }
}

def get_phase_from_cycle_day(cycle_day):
    """
    Determine the menstrual cycle phase based on cycle day.
    
    Args:
        cycle_day (int): Day of the menstrual cycle (1-28)
    
    Returns:
        str: Phase name
    """
    if 1 <= cycle_day <= 5:
        return 'perimenstruation'
    elif 6 <= cycle_day <= 11:
        return 'mid_follicular'
    elif 12 <= cycle_day <= 14:
        return 'periovulation'
    elif 15 <= cycle_day <= 21:
        return 'early_luteal'
    else:  # 22-28
        return 'mid_late_luteal'

def get_estradiol_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    """
    Get the estradiol distribution parameters for the specified phase or cycle day.
    
    Args:
        phase (str): Phase name (optional)
        cycle_day (int): Day of the menstrual cycle (optional)
        config_path (str): Path to config file
    
    Returns:
        dict: Distribution parameters including min, max, mean, and standard deviation
    """
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day)
    
    try:
        hormone_distributions = load_hormone_distributions_from_config(config_path)
        return hormone_distributions.get('estradiol', {}).get(phase, FALLBACK_ESTRADIOL_DISTRIBUTIONS[phase])
    except (FileNotFoundError, KeyError):
        return FALLBACK_ESTRADIOL_DISTRIBUTIONS[phase]

def get_progesterone_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    """
    Get the progesterone distribution parameters for the specified phase or cycle day.
    
    Args:
        phase (str): Phase name (optional)
        cycle_day (int): Day of the menstrual cycle (optional)
        config_path (str): Path to config file
    
    Returns:
        dict: Distribution parameters including min, max, mean, and standard deviation
    """
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day)
    
    try:
        hormone_distributions = load_hormone_distributions_from_config(config_path)
        return hormone_distributions.get('progesterone', {}).get(phase, FALLBACK_PROGESTERONE_DISTRIBUTIONS[phase])
    except (FileNotFoundError, KeyError):
        return FALLBACK_PROGESTERONE_DISTRIBUTIONS[phase]

def get_testosterone_distribution(phase=None, cycle_day=None, config_path='config/simulation_config.yaml'):
    """
    Get the testosterone distribution parameters for the specified phase or cycle day.
    
    Args:
        phase (str): Phase name (optional)
        cycle_day (int): Day of the menstrual cycle (optional)
        config_path (str): Path to config file
    
    Returns:
        dict: Distribution parameters including min, max, mean, and standard deviation
    """
    if cycle_day is not None:
        phase = get_phase_from_cycle_day(cycle_day)
    
    try:
        hormone_distributions = load_hormone_distributions_from_config(config_path)
        return hormone_distributions.get('testosterone', {}).get(phase, FALLBACK_TESTOSTERONE_DISTRIBUTIONS[phase])
    except (FileNotFoundError, KeyError):
        return FALLBACK_TESTOSTERONE_DISTRIBUTIONS[phase]

def sample_lognormal_from_distribution(dist_params):
    """
    Sample from a lognormal distribution with the given parameters.
    
    Args:
        dist_params (dict): Dictionary containing min, max, mean, and sd
    
    Returns:
        float: Sampled value
    """
    # Convert mean and sd to lognormal parameters
    mu = np.log(dist_params['mean']**2 / np.sqrt(dist_params['mean']**2 + dist_params['sd']**2))
    sigma = np.sqrt(np.log(1 + dist_params['sd']**2 / dist_params['mean']**2))
    
    # Sample from lognormal
    value = np.random.lognormal(mu, sigma)
    
    # Ensure value is within min/max bounds
    value = max(dist_params['min'], min(dist_params['max'], value))
    
    return value

def sample_baseline_values(config_path='config/simulation_config.yaml'):
    """
    Sample baseline values for each hormone from the perimenstruation phase.
    
    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Dictionary containing baseline values for estradiol, progesterone, and testosterone
    """
    try:
        hormone_distributions = load_hormone_distributions_from_config(config_path)
        return {
            'estradiol': sample_lognormal_from_distribution(hormone_distributions.get('estradiol', {}).get('perimenstruation', FALLBACK_ESTRADIOL_DISTRIBUTIONS['perimenstruation'])),
            'progesterone': sample_lognormal_from_distribution(hormone_distributions.get('progesterone', {}).get('perimenstruation', FALLBACK_PROGESTERONE_DISTRIBUTIONS['perimenstruation'])),
            'testosterone': sample_lognormal_from_distribution(hormone_distributions.get('testosterone', {}).get('perimenstruation', FALLBACK_TESTOSTERONE_DISTRIBUTIONS['perimenstruation']))
        }
    except (FileNotFoundError, KeyError):
        return {
            'estradiol': sample_lognormal_from_distribution(FALLBACK_ESTRADIOL_DISTRIBUTIONS['perimenstruation']),
            'progesterone': sample_lognormal_from_distribution(FALLBACK_PROGESTERONE_DISTRIBUTIONS['perimenstruation']),
            'testosterone': sample_lognormal_from_distribution(FALLBACK_TESTOSTERONE_DISTRIBUTIONS['perimenstruation'])
        }

def generate_progesterone_value(cycle_day, baseline):
    """
    Generate a progesterone value for the given cycle day, using the baseline as a reference.
    Custom pattern:
    - Days 1-9: fluctuate within 0.5 SD of baseline
    - Days 10-20: rising steadily, peak at day 20 (4x baseline)
    - Days 21-28: falling steadily, back to within 0.5 SD of baseline by day 28
    """
    if 1 <= cycle_day <= 9:
        # Fluctuate within 0.5 SD of baseline
        value = baseline + np.random.normal(0, 0.5 * baseline * 0.1)
    elif 10 <= cycle_day <= 20:
        # Rising steadily to peak at day 20 (5.5x baseline)
        # Linear interpolation from baseline at day 10 to 5.5x baseline at day 20
        peak = 4 * baseline
        value = baseline + (peak - baseline) * ((cycle_day - 10) / (20 - 10))
        value += np.random.normal(0, 0.1 * baseline)
    elif 21 <= cycle_day <= 28:
        # Falling steadily to baseline by day 28
        # Linear interpolation from peak at day 21 to baseline at day 28
        peak = 5 * baseline
        value = peak - (peak - baseline) * ((cycle_day - 21) / (28 - 21))
        value += np.random.normal(0, 0.1 * baseline)
    else:
        # Fallback: baseline
        value = baseline
    return max(0, value)

def generate_estradiol_value(cycle_day, baseline):
    """
    Generate an estradiol value for the given cycle day, using the baseline as a reference.
    Custom pattern:
    - Days 1-5: fluctuate within 0.5 SD of baseline
    - Days 6-14: increasing, peak at day 14 (5x baseline)
    - Days 15-17: decreasing (inflection at day 18)
    - Days 18-20: increasing, secondary peak at day 20 (2.5x baseline)
    - Days 21-28: decreasing, back to within 0.5 SD of baseline by day 28
    """
    if 1 <= cycle_day <= 5:
        # Fluctuate within 0.5 SD of baseline
        value = baseline + np.random.normal(0, 0.5 * baseline * 0.1)
    elif 6 <= cycle_day <= 14:
        # Increasing to peak at day 14 (5x baseline)
        peak = 5.0 * baseline
        value = baseline + (peak - baseline) * ((cycle_day - 6) / (14 - 6))
        value += np.random.normal(0, 0.1 * baseline)
    elif 15 <= cycle_day <= 17:
        # Decreasing from peak at day 14 to inflection at day 18
        peak = 5.0 * baseline
        inflection = baseline
        value = peak - (peak - inflection) * ((cycle_day - 14) / (18 - 14))
        value += np.random.normal(0, 0.1 * baseline)
    elif 18 <= cycle_day <= 20:
        # Increasing to secondary peak at day 20 (2.5x baseline)
        trough = baseline
        secondary_peak = 2.5 * baseline
        value = trough + (secondary_peak - trough) * ((cycle_day - 18) / (20 - 18))
        value += np.random.normal(0, 0.1 * baseline)
    elif 21 <= cycle_day <= 28:
        # Decreasing from secondary peak at day 20 to baseline at day 28
        secondary_peak = 2.5 * baseline
        value = secondary_peak - (secondary_peak - baseline) * ((cycle_day - 21) / (28 - 21))
        value += np.random.normal(0, 0.1 * baseline)
    else:
        # Fallback: baseline
        value = baseline
    return max(0, value) 