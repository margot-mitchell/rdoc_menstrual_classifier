"""
Rule-based prior model for menstrual cycle phase prediction.
Uses survey responses and backcounting/forwardcounting logic.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RuleBasedPrior:
    """
    Rule-based prior model that predicts menstrual phases using survey responses
    and backcounting/forwardcounting logic without requiring training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the rule-based prior model.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.survey_df = None
        self.period_df = None
        self.sequence_length = config.get('models', {}).get('temporal', {}).get('sequence_length', 70)
        
    def load_data(self) -> None:
        """Load survey and period data."""
        data_config = self.config['data']
        
        # Load survey data
        survey_path = data_config['survey_data_path']
        if os.path.exists(survey_path):
            self.survey_df = pd.read_csv(survey_path)
            self.survey_df['date_of_last_period'] = pd.to_datetime(self.survey_df['date_of_last_period'])
            self.survey_df['date_of_response'] = pd.to_datetime(self.survey_df['date_of_response'])
            logger.info(f"Loaded survey data: {len(self.survey_df)} responses")
        else:
            logger.warning(f"Survey data not found at {survey_path}")
            
        # Load period data
        period_path = data_config['period_data_path']
        if os.path.exists(period_path):
            self.period_df = pd.read_csv(period_path)
            self.period_df['date'] = pd.to_datetime(self.period_df['date'])
            logger.info(f"Loaded period data: {len(self.period_df)} records")
        else:
            logger.warning(f"Period data not found at {period_path}")
    
    def predict_phase_from_survey(self, subject_id: int, target_date: datetime) -> str:
        """
        Predict phase using survey response and backcounting/forwardcounting.
        
        Args:
            subject_id: Subject ID
            target_date: Date to predict phase for
            
        Returns:
            str: Predicted phase ('perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal')
        """
        if self.survey_df is None:
            return 'unknown'
            
        # Get survey response for this subject
        subject_survey = self.survey_df[self.survey_df['subject_id'] == subject_id]
        if subject_survey.empty:
            return 'unknown'
            
        survey_row = subject_survey.iloc[0]
        last_period_date = survey_row['date_of_last_period']
        cycle_length = survey_row['cycle_length']
        
        # Calculate days since last period
        days_since_period = (target_date - last_period_date).days
        
        # Handle negative days (before reported last period)
        if days_since_period < 0:
            # Backcount: assume previous cycle had same length
            days_since_period = days_since_period + cycle_length
            
        # Calculate cycle day (1-28)
        cycle_day = (days_since_period % cycle_length) + 1
        
        # Map cycle day to phase
        return self._map_cycle_day_to_phase(cycle_day)
    
    def predict_phase_from_period_data(self, subject_id: int, target_date: datetime) -> str:
        """
        Predict phase using actual period data and forwardcounting.
        
        Args:
            subject_id: Subject ID
            target_date: Date to predict phase for
            
        Returns:
            str: Predicted phase
        """
        if self.period_df is None:
            return 'unknown'
            
        # Get period data for this subject
        subject_periods = self.period_df[
            (self.period_df['subject_id'] == subject_id) & 
            (self.period_df['period'] == 'Yes')
        ].copy()
        
        if subject_periods.empty:
            return 'unknown'
            
        # Sort by date
        subject_periods = subject_periods.sort_values('date')
        
        # Find the most recent period before or on target date
        recent_periods = subject_periods[subject_periods['date'] <= target_date]
        
        if recent_periods.empty:
            # If no periods before target date, use the first period and backcount
            first_period = subject_periods.iloc[0]['date']
            days_before_first = (first_period - target_date).days
            
            # Estimate cycle length from available periods
            if len(subject_periods) >= 2:
                cycle_length = self._estimate_cycle_length(subject_periods)
            else:
                cycle_length = 28  # Default
                
            # Backcount cycles
            estimated_cycle_day = (cycle_length - (days_before_first % cycle_length)) % cycle_length
            if estimated_cycle_day == 0:
                estimated_cycle_day = cycle_length
                
            return self._map_cycle_day_to_phase(estimated_cycle_day)
        else:
            # Use most recent period and forwardcount
            last_period = recent_periods.iloc[-1]['date']
            days_since_period = (target_date - last_period).days
            
            # Estimate cycle length
            if len(subject_periods) >= 2:
                cycle_length = self._estimate_cycle_length(subject_periods)
            else:
                cycle_length = 28  # Default
                
            # Calculate cycle day
            cycle_day = (days_since_period % cycle_length) + 1
            
            return self._map_cycle_day_to_phase(cycle_day)
    
    def _estimate_cycle_length(self, period_data: pd.DataFrame) -> float:
        """
        Estimate cycle length from period data.
        
        Args:
            period_data: DataFrame with period dates
            
        Returns:
            float: Estimated cycle length in days
        """
        if len(period_data) < 2:
            return 28.0
            
        # Calculate intervals between consecutive periods
        dates = period_data['date'].sort_values().values
        intervals = []
        
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1])
            # Handle both timedelta and timedelta64 objects
            if hasattr(interval, 'days'):
                intervals.append(interval.days)
            else:
                # For numpy timedelta64, convert to days
                intervals.append(interval.astype('timedelta64[D]').astype(int))
            
        return np.mean(intervals)
    
    def _map_cycle_day_to_phase(self, cycle_day: int) -> str:
        """
        Map cycle day to menstrual phase.
        
        Args:
            cycle_day: Day of menstrual cycle (1-28)
            
        Returns:
            str: Phase name
        """
        if cycle_day <= 5:
            return 'perimenstruation'
        elif 6 <= cycle_day <= 13:
            return 'mid_follicular'
        elif 14 <= cycle_day <= 16:
            return 'periovulation'
        elif 17 <= cycle_day <= 22:
            return 'early_luteal'
        else:  # 23-28
            return 'mid_late_luteal'
    
    def predict_phases(self, hormone_data: pd.DataFrame) -> np.ndarray:
        """
        Predict phases for hormone data using rule-based logic.
        
        Args:
            hormone_data: DataFrame with hormone measurements
            
        Returns:
            np.ndarray: Array of predicted phases
        """
        if self.survey_df is None and self.period_df is None:
            self.load_data()
            
        predictions = []
        
        for _, row in hormone_data.iterrows():
            subject_id = row['subject_id']
            
            # Try to get date from the data
            if 'date' in hormone_data.columns:
                target_date = pd.to_datetime(row['date'])
            else:
                # If no date column, use a default date
                target_date = pd.to_datetime('2024-01-01')
            
            # Try period data first (more accurate), then fall back to survey
            if self.period_df is not None:
                phase = self.predict_phase_from_period_data(subject_id, target_date)
            elif self.survey_df is not None:
                phase = self.predict_phase_from_survey(subject_id, target_date)
            else:
                phase = 'unknown'
                
            predictions.append(phase)
            
        return np.array(predictions)
    
    def get_prior_probabilities(self, hormone_data: pd.DataFrame) -> np.ndarray:
        """
        Get prior probabilities for each phase based on rule-based predictions.
        
        Args:
            hormone_data: DataFrame with hormone measurements
            
        Returns:
            np.ndarray: Array of prior probabilities (n_samples, n_phases)
        """
        phase_predictions = self.predict_phases(hormone_data)
        
        # Define phase order
        phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
        
        # Create probability matrix
        n_samples = len(hormone_data)
        n_phases = len(phases)
        probabilities = np.zeros((n_samples, n_phases))
        
        for i, predicted_phase in enumerate(phase_predictions):
            if predicted_phase in phases:
                phase_idx = phases.index(predicted_phase)
                # High confidence for rule-based prediction (0.8)
                probabilities[i, phase_idx] = 0.8
                # Distribute remaining probability evenly among other phases
                remaining_prob = 0.2 / (n_phases - 1)
                for j in range(n_phases):
                    if j != phase_idx:
                        probabilities[i, j] = remaining_prob
            else:
                # If prediction is unknown, use uniform distribution
                probabilities[i, :] = 1.0 / n_phases
                
        return probabilities
    
    def save_predictions(self, hormone_data: pd.DataFrame, output_dir: str) -> None:
        """
        Save rule-based predictions to file.
        
        Args:
            hormone_data: DataFrame with hormone measurements
            output_dir: Directory to save predictions
        """
        predictions = self.predict_phases(hormone_data)
        
        # Create results DataFrame
        results_df = hormone_data.copy()
        results_df['rule_based_phase'] = predictions
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'rule_based_predictions.csv')
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Rule-based predictions saved to {output_path}")
        
        # Print summary
        phase_counts = pd.Series(predictions).value_counts()
        logger.info("Rule-based prediction summary:")
        for phase, count in phase_counts.items():
            logger.info(f"  {phase}: {count}") 