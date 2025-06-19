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
        
        # Log data organization
        logger.info(f"Organized data for {len(self.survey_data)} subjects")
        logger.info(f"Survey data keys: {list(self.survey_data.keys())}")
        logger.info(f"Period data keys: {list(self.period_data.keys())}")
        logger.info(f"Sample dates keys: {list(self.sample_dates.keys())}")
    
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
            'sample_dates': self.sample_dates.get(subject_id, [])
        }
    
    def get_all_subjects(self):
        """Get list of all subject IDs."""
        return list(self.survey_data.keys())
    
    def get_phase_from_cycle_day(self, cycle_day):
        """
        Map cycle day to detailed phase.
        
        Args:
            cycle_day (int): Day of the menstrual cycle (1-28)
            
        Returns:
            str: Phase name (perimenstruation, mid_follicular, periovulation, early_luteal, mid_late_luteal)
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
    
    def predict_cycle_position(self, hormone_df):
        """
        Predict cycle position for each sample using menstrual cycle length, date_of_last_period, and actual period data.
        
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
            
            # Get subject's survey data
            survey_info = self.survey_data.get(subject_id, {})
            if not survey_info:
                logger.warning(f"No survey data found for subject {subject_id}")
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
                
                # Check if this sample date is a period day
                if sample_date in period_dates:
                    predictions.append('perimenstruation')
                    continue
                
                # Calculate days since last period
                days_since_period = (sample_date - date_of_last_period).days
                
                # Calculate cycle day (1-28)
                cycle_day = (days_since_period % cycle_length) + 1
                
                # Count how many period days are in this cycle
                cycle_start = date_of_last_period
                cycle_end = cycle_start + pd.Timedelta(days=cycle_length)
                cycle_period_days = [d for d in period_dates if cycle_start <= d < cycle_end]
                num_period_days = len(cycle_period_days)
                
                # Calculate remaining days for other phases
                remaining_days = cycle_length - num_period_days
                
                # Adjust phase boundaries to fit remaining days
                # Original proportions: mid_follicular (6 days), periovulation (3 days), early_luteal (7 days), mid_late_luteal (7 days)
                # Total original: 23 days, so we scale by remaining_days/23
                scale_factor = remaining_days / 23.0
                
                mid_follicular_days = int(round(6 * scale_factor))
                periovulation_days = int(round(3 * scale_factor))
                early_luteal_days = int(round(7 * scale_factor))
                mid_late_luteal_days = remaining_days - mid_follicular_days - periovulation_days - early_luteal_days
                
                # Adjust cycle day to account for period days
                # Skip period days when calculating phase
                adjusted_cycle_day = cycle_day
                for period_date in cycle_period_days:
                    if period_date < sample_date:
                        adjusted_cycle_day -= 1
                
                # Map adjusted cycle day to phase
                if adjusted_cycle_day <= mid_follicular_days:
                    phase = 'mid_follicular'
                elif adjusted_cycle_day <= mid_follicular_days + periovulation_days:
                    phase = 'periovulation'
                elif adjusted_cycle_day <= mid_follicular_days + periovulation_days + early_luteal_days:
                    phase = 'early_luteal'
                else:
                    phase = 'mid_late_luteal'
                
                predictions.append(phase)
        
        return np.array(predictions) 