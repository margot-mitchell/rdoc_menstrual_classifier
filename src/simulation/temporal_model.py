import numpy as np
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TemporalModel:
    """
    A temporal model for generating hormone data based on menstrual cycle phases.
    """
    
    def __init__(self, config_path='config/simulation_config.yaml'):
        """
        Initialize the temporal model.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.phase_duration_sd_multiplier = self.config.get('phase_duration_sd_multiplier', 1.0)
        
        # Phase durations for each subject (will be generated per subject)
        self.subject_phase_durations = {}
        
        # Load hormone distributions
        self.hormone_distributions = self.config.get('hormones', {})
        
        # Define the phases in order
        self.phases = [
            'perimenstruation',
            'mid_follicular', 
            'periovulation',
            'early_luteal',
            'mid_late_luteal'
        ]
        
        # Generate phase durations for all subjects
        self._generate_subject_phase_durations()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_subject_phase_durations(self):
        """Generate phase durations for all subjects."""
        from src.simulation.utils import generate_phase_durations
        
        # Generate base phase durations
        phase_durations = generate_phase_durations(sd_multiplier=self.phase_duration_sd_multiplier)
        
        # Store the base phase durations (will be scaled per subject)
        self.base_phase_durations = phase_durations.copy()
        
        # Calculate the total duration of the base cycle
        original_total = sum(phase_durations.values())
        
        # Scale phase durations to match the target cycle length (28 days)
        target_cycle_length = 28
        scale_factor = target_cycle_length / original_total
        
        # Scale phase durations proportionally
        adjusted_phase_durations = {}
        for phase, duration in phase_durations.items():
            adjusted_phase_durations[phase] = int(round(duration * scale_factor))
        
        # Ensure the total matches exactly by adjusting the largest phase
        total_adjusted = sum(adjusted_phase_durations.values())
        if total_adjusted != target_cycle_length:
            # Find the phase with the largest duration and adjust it
            largest_phase = max(adjusted_phase_durations.items(), key=lambda x: x[1])[0]
            adjusted_phase_durations[largest_phase] += (target_cycle_length - total_adjusted)
        
        # Store the adjusted phase durations
        self.phase_durations = adjusted_phase_durations
        
        logger.info(f"Generated phase durations: {self.phase_durations}")
        logger.info(f"Total cycle length: {sum(self.phase_durations.values())}")
    
    def get_phase_from_cycle_day(self, cycle_day, phase_durations):
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
    
    def get_hormone_value(self, hormone_name: str, phase: str, cycle_day: int) -> float:
        """
        Get hormone value for a specific phase and cycle day.
        
        Args:
            hormone_name (str): Name of the hormone ('estradiol', 'progesterone', 'testosterone')
            phase (str): Current menstrual cycle phase
            cycle_day (int): Day of the menstrual cycle
            
        Returns:
            float: Hormone value
        """
        if hormone_name not in self.hormone_distributions:
            raise ValueError(f"Unknown hormone: {hormone_name}")
        
        if phase not in self.hormone_distributions[hormone_name]:
            raise ValueError(f"Unknown phase: {phase}")
        
        # Get distribution parameters for this hormone and phase
        params = self.hormone_distributions[hormone_name][phase]
        
        # Sample from the distribution
        mean = params['mean']
        sd = params['sd']
        
        # Generate value
        value = np.random.normal(mean, sd)
        
        # Ensure value is within bounds
        min_val = params.get('min', 0)
        max_val = params.get('max', float('inf'))
        value = np.clip(value, min_val, max_val)
        
        return value
    
    def generate_hormone_series(self, subject_id: int, n_days: int) -> pd.DataFrame:
        """
        Generate hormone data series for a subject.
        
        Args:
            subject_id (int): Subject ID
            n_days (int): Number of days to generate
            
        Returns:
            pd.DataFrame: DataFrame with hormone data
        """
        # Get phase durations for this subject
        phase_durations = self.phase_durations
        
        data = []
        cycle_day = 1
        
        for day in range(n_days):
            # Determine current phase
            phase = self.get_phase_from_cycle_day(cycle_day, phase_durations)
            
            # Generate hormone values
            estradiol = self.get_hormone_value('estradiol', phase, cycle_day)
            progesterone = self.get_hormone_value('progesterone', phase, cycle_day)
            testosterone = self.get_hormone_value('testosterone', phase, cycle_day)
            
            data.append({
                'subject_id': subject_id,
                'day': day + 1,
                'cycle_day': cycle_day,
                'phase': phase,
                'estradiol': estradiol,
                'progesterone': progesterone,
                'testosterone': testosterone
            })
            
            cycle_day += 1
            if cycle_day > sum(phase_durations.values()):
                cycle_day = 1  # Reset to beginning of cycle
        
        return pd.DataFrame(data)
    
    def generate_all_subjects_data(self, n_subjects: int, n_days: int) -> pd.DataFrame:
        """
        Generate hormone data for all subjects.
        
        Args:
            n_subjects (int): Number of subjects
            n_days (int): Number of days per subject
            
        Returns:
            pd.DataFrame: Combined hormone data for all subjects
        """
        all_data = []
        
        for subject_id in range(n_subjects):
            subject_data = self.generate_hormone_series(subject_id, n_days)
            all_data.append(subject_data)
        
        return pd.concat(all_data, ignore_index=True) 