import numpy as np
import pandas as pd
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class TemporalModel:
    """Temporal model for organizing menstrual cycle data."""
    def __init__(self):
        # Store data organized by subject_id
        self.subject_data = {}
        # Store survey responses
        self.survey_data = {}
        # Store period data
        self.period_data = {}
        # Store sample dates
        self.sample_dates = {}
        # Store phase durations for each subject (using realistic distributions)
        self.subject_phase_durations = {}
    
    def organize_data(self, hormone_df, period_df, survey_df):
        """Organize data from period_sleep_data and survey_responses."""
        # Convert dates to datetime
        period_df['date'] = pd.to_datetime(period_df['date'])
        hormone_df['date'] = pd.to_datetime(hormone_df['date'])
        
        # Organize survey data
        for _, row in survey_df.iterrows():
            subject_id = row['subject_id']
            self.survey_data[subject_id] = {
                'cycle_length': row['cycle_length'],
                'menstrual_pattern': row['menstrual_pattern'],
                'date_of_response': pd.to_datetime(row['date_of_response']),
                'date_of_last_period': pd.to_datetime(row['date_of_last_period'])
            }
        
        # Organize period data - only store date and period status
        for _, row in period_df.iterrows():
            subject_id = row['subject_id']
            if subject_id not in self.period_data:
                self.period_data[subject_id] = []
            
            self.period_data[subject_id].append({
                'date': row['date'],
                'period': row['period']
            })
        
        # Organize sample dates
        for _, row in hormone_df.iterrows():
            subject_id = row['subject_id']
            if subject_id not in self.sample_dates:
                self.sample_dates[subject_id] = []
            
            self.sample_dates[subject_id].append(row['date'])
        
        # Sort dates for each subject
        for subject_id in self.period_data:
            self.period_data[subject_id].sort(key=lambda x: x['date'])
        for subject_id in self.sample_dates:
            self.sample_dates[subject_id].sort()
        
        # Generate realistic phase durations for each subject
        self._generate_subject_phase_durations()
        
        # Log data organization
        logger.info(f"Organized data for {len(self.survey_data)} subjects")
        logger.info(f"Survey data keys: {list(self.survey_data.keys())}")
        logger.info(f"Period data keys: {list(self.period_data.keys())}")
        logger.info(f"Sample dates keys: {list(self.sample_dates.keys())}")
    
    def _generate_subject_phase_durations(self):
        """Generate realistic phase durations for each subject using the same distributions as simulation."""
        # Import the phase configuration from simulation
        from src.config.phase_config import generate_phase_durations
        from src.simulation.subject import get_phase_duration_multiplier
        
        for subject_id in self.survey_data.keys():
            # Get menstrual pattern to determine variability
            pattern = self.survey_data[subject_id]['menstrual_pattern']
            phase_duration_sd_multiplier = get_phase_duration_multiplier(pattern)
            
            # Generate realistic phase durations
            phase_durations = generate_phase_durations(sd_multiplier=phase_duration_sd_multiplier)
            
            # Scale to match the subject's cycle length
            cycle_length = self.survey_data[subject_id]['cycle_length']
            original_total = sum(phase_durations.values())
            scale_factor = cycle_length / original_total
            
            # Scale phase durations proportionally
            adjusted_phase_durations = {}
            for phase, duration in phase_durations.items():
                adjusted_phase_durations[phase] = int(round(duration * scale_factor))
            
            # Ensure the total matches exactly by adjusting the largest phase
            total_adjusted = sum(adjusted_phase_durations.values())
            if total_adjusted != cycle_length:
                largest_phase = max(adjusted_phase_durations.items(), key=lambda x: x[1])[0]
                adjusted_phase_durations[largest_phase] += (cycle_length - total_adjusted)
            
            self.subject_phase_durations[subject_id] = adjusted_phase_durations
            
            logger.debug(f"Subject {subject_id} phase durations: {adjusted_phase_durations}")
    
    def train(self, hormone_df, period_df, survey_df):
        """
        Organize the data for the rule-based temporal model.
        
        Args:
            hormone_df (pd.DataFrame): DataFrame containing hormone data
            period_df (pd.DataFrame): DataFrame containing period data
            survey_df (pd.DataFrame): DataFrame containing survey responses
        """
        logger.info("Organizing data for temporal model...")
        self.organize_data(hormone_df, period_df, survey_df)
        logger.info("Temporal model data organization complete")
    
    def get_subject_data(self, subject_id):
        """Get all data for a specific subject."""
        return {
            'survey': self.survey_data.get(subject_id, {}),
            'periods': self.period_data.get(subject_id, []),
            'sample_dates': self.sample_dates.get(subject_id, []),
            'phase_durations': self.subject_phase_durations.get(subject_id, {})
        }
    
    def get_all_subjects(self):
        """Get list of all subject IDs."""
        return list(self.survey_data.keys())
    
    def get_phase_from_cycle_day(self, cycle_day, phase_durations):
        """
        Map cycle day to detailed phase using realistic phase durations.
        
        Args:
            cycle_day (int): Day of the menstrual cycle
            phase_durations (dict): Dictionary of phase durations for this subject
            
        Returns:
            str: Phase name (perimenstruation, mid_follicular, periovulation, early_luteal, mid_late_luteal)
        """
        current_day = 0
        for phase, duration in phase_durations.items():
            current_day += duration
            if cycle_day <= current_day:
                return phase
        return 'mid_late_luteal'  # Default to last phase if something goes wrong
    
    def predict_cycle_position(self, hormone_df):
        """
        Predict cycle position for each sample using menstrual cycle length, date_of_last_period, 
        actual period data, and realistic phase distributions.
        
        Args:
            hormone_df (pd.DataFrame): DataFrame containing hormone data with columns:
                - subject_id
                - date
                - estradiol
                - progesterone
                - testosterone
        
        Returns:
            np.ndarray: Array of predicted phases for each sample
        """
        # Convert dates to datetime if they aren't already
        hormone_df = hormone_df.copy()
        hormone_df['date'] = pd.to_datetime(hormone_df['date'])
        
        # Load period data if not already available
        if not hasattr(self, 'period_df') or self.period_df is None:
            self.period_df = pd.read_csv('output/period_sleep_data.csv')
            self.period_df['date'] = pd.to_datetime(self.period_df['date'])
        
        # Initialize array for predictions
        predictions = []
        
        # Process each subject's data
        for subject_id in hormone_df['subject_id'].unique():
            subject_data = hormone_df[hormone_df['subject_id'] == subject_id].copy()
            subject_data = subject_data.sort_values('date')
            
            # Get subject's survey data and phase durations
            survey_info = self.survey_data.get(subject_id, {})
            phase_durations = self.subject_phase_durations.get(subject_id, {})
            
            if not survey_info or not phase_durations:
                logger.warning(f"No survey data or phase durations found for subject {subject_id}")
                predictions.extend(['mid_follicular'] * len(subject_data))  # Default to mid_follicular
                continue
            
            cycle_length = survey_info['cycle_length']
            date_of_last_period = survey_info['date_of_last_period']
            
            # Get period data for this subject
            subject_periods = self.period_df[self.period_df['subject_id'] == subject_id].copy()
            subject_periods = subject_periods.sort_values('date')
            
            # Find actual period days (where period == 'Yes')
            period_dates = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
            
            # Calculate cycle day for each sample
            for _, row in subject_data.iterrows():
                sample_date = row['date']
                
                # Check if this sample date is a period day - assign perimenstruation
                if sample_date in period_dates:
                    predictions.append('perimenstruation')
                    continue
                
                # Calculate days since last period
                days_since_period = (sample_date - date_of_last_period).days
                
                # Calculate cycle day (1 to cycle_length)
                cycle_day = (days_since_period % cycle_length) + 1
                
                # Use the realistic phase durations to determine the phase
                # This uses the same logic as the simulation data generation
                phase = self.get_phase_from_cycle_day(cycle_day, phase_durations)
                predictions.append(phase)
        
        return np.array(predictions) 