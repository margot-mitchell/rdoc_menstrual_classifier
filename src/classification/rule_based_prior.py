"""
Rule-based prior model for menstrual cycle phase prediction.
Uses survey responses and backcounting/forwardcounting logic.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import yaml

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
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
        
        self.phases = [
            'perimenstruation',
            'mid_follicular',
            'periovulation',
            'early_luteal',
            'mid_late_luteal'
        ]
        # Load phase durations from simulation config
        self.phase_durations_config = self._load_phase_durations()
        self.phase_durations_by_subject = {}  # subject_id -> phase_durations dict
        
        # Calculate cumulative phase boundaries for each subject
        self.phase_boundaries = {}
        
    def _load_phase_durations(self) -> Dict[str, Dict[str, float]]:
        """
        Load phase durations from simulation config.
        
        Returns:
            Dict containing phase durations with mean and sd
        """
        try:
            # Try to load from simulation config
            sim_config_path = os.path.join(project_root, 'config', 'simulation_config.yaml')
            with open(sim_config_path, 'r') as f:
                sim_config = yaml.safe_load(f)
            
            phase_durations = sim_config.get('phase_durations', {})
            logger.info(f"Loaded phase durations from simulation config: {phase_durations}")
            return phase_durations
            
        except Exception as e:
            logger.warning(f"Could not load phase durations from config: {e}")
            # Fallback to default durations (28-day cycle)
            default_durations = {
                'perimenstruation': {'mean': 3.0, 'sd': 1.0},
                'mid_follicular': {'mean': 8.4, 'sd': 1.8},
                'periovulation': {'mean': 3.5, 'sd': 1.0},
                'early_luteal': {'mean': 4.8, 'sd': 2.0},
                'mid_late_luteal': {'mean': 8.3, 'sd': 3.0}
            }
            logger.info(f"Using default phase durations: {default_durations}")
            return default_durations
    
    def get_subject_phase_durations(self, subject_id: int) -> Dict[str, int]:
        """
        Deterministically sample phase durations for a subject using (subject_id + 2000) as the seed.
        This ensures the prior gets different phase durations than the simulation (which uses subject_id as seed).
        Also applies menstrual pattern variability if available.
        """
        if subject_id not in self.phase_durations_by_subject:
            # Use different seed than simulation to avoid data leakage
            np.random.seed(subject_id + 2000)
            
            # Get menstrual pattern if available
            sd_multiplier = 1.0  # default
            if self.survey_df is not None:
                subject_survey = self.survey_df[self.survey_df['subject_id'] == subject_id]
                if not subject_survey.empty:
                    pattern = subject_survey.iloc[0].get('menstrual_pattern', 'Regular (within 5-7 days)')
                    sd_multiplier = self._get_phase_duration_multiplier(pattern)
            
            phase_durations = {}
            for phase, params in self.phase_durations_config.items():
                # Apply the sd_multiplier to adjust variability based on menstrual pattern
                duration = max(1, int(np.random.normal(params['mean'], params['sd'] * sd_multiplier)))
                phase_durations[phase] = duration
            self.phase_durations_by_subject[subject_id] = phase_durations
            np.random.seed()  # Reset seed
        return self.phase_durations_by_subject[subject_id]

    def _get_phase_duration_multiplier(self, pattern: str) -> float:
        """
        Get phase duration variability multiplier based on menstrual pattern.
        Same logic as in simulation.
        
        Args:
            pattern (str): Menstrual pattern
            
        Returns:
            float: Phase duration multiplier
        """
        if pattern == 'Extremely regular (no more than 1-2 days before or after expected)':
            return 0.3
        elif pattern == 'Very regular (within 3-4 days)':
            return 0.6
        else:  # 'Regular (within 5-7 days)' or any other pattern
            return 1.0

    def _calculate_phase_boundaries(self, cycle_length: float, subject_id: int = None) -> Dict[str, tuple]:
        """
        Calculate phase boundaries for a given cycle length using deterministic subject phase durations.
        """
        if subject_id is not None:
            phase_durations = self.get_subject_phase_durations(subject_id)
        else:
            # fallback: sample once
            np.random.seed(0)
            phase_durations = {}
            for phase, params in self.phase_durations_config.items():
                duration = max(1, int(np.random.normal(params['mean'], params['sd'])))
                phase_durations[phase] = duration
            np.random.seed()
        # Scale to match cycle_length
        total = sum(phase_durations.values())
        scale_factor = cycle_length / total
        scaled = {phase: max(1, int(round(d * scale_factor))) for phase, d in phase_durations.items()}
        # Adjust largest phase to match total
        total_scaled = sum(scaled.values())
        if total_scaled != cycle_length:
            largest_phase = max(scaled.items(), key=lambda x: x[1])[0]
            scaled[largest_phase] += (int(cycle_length) - total_scaled)
        # Build boundaries
        boundaries = {}
        current_day = 1
        for phase in self.phases:
            duration = scaled[phase]
            boundaries[phase] = (current_day, current_day + duration - 1)
            current_day += duration
        return boundaries
    
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
        
        # Map cycle day to phase using realistic durations
        return self._map_cycle_day_to_phase(cycle_day, cycle_length, subject_id)
    
    def predict_phase_from_period_data(self, subject_id: int, target_date: datetime) -> str:
        """
        Predict phase using actual period data and survey anchor logic.
        Handles cases where survey date_of_last_period is before, within, or after period data range.
        """
        if self.period_df is None or self.survey_df is None:
            return 'unknown'

        # Get period data for this subject
        subject_periods = self.period_df[self.period_df['subject_id'] == subject_id].copy()
        if subject_periods.empty:
            return 'unknown'
        subject_periods['date'] = pd.to_datetime(subject_periods['date'])
        subject_periods = subject_periods.sort_values('date')
        period_dates = subject_periods[subject_periods['period'] == 'Yes']['date']
        if period_dates.empty:
            return 'unknown'
        first_period_date = period_dates.min()
        last_period_date = period_dates.max()

        # Get survey date_of_last_period
        survey_row = self.survey_df[self.survey_df['subject_id'] == subject_id]
        if survey_row.empty:
            return 'unknown'
        survey_last_period = pd.to_datetime(survey_row.iloc[0]['date_of_last_period'])
        cycle_length = survey_row.iloc[0]['cycle_length']

        if survey_last_period < first_period_date:
            # Survey anchor is before period data range: forward count to find first anchor in data
            anchor = survey_last_period
            anchors = [anchor]
            while anchor < first_period_date:
                anchor += pd.Timedelta(days=cycle_length)
                anchors.append(anchor)
            # Now, for any target_date, find the most recent anchor <= target_date
            recent_anchor = max([a for a in anchors if a <= target_date], default=anchors[0])
            days_since_anchor = (target_date - recent_anchor).days
            cycle_day = (days_since_anchor % cycle_length) + 1
        elif survey_last_period > last_period_date:
            # Survey date is after period data: error
            raise ValueError(f"date_of_last_period for subject {subject_id} is after last period data date. Check survey data.")
        else:
            # Survey anchor is within period data: use first 'Yes' period value as anchor
            anchor = first_period_date
            days_since_anchor = (target_date - anchor).days
            cycle_day = (days_since_anchor % cycle_length) + 1

        # Map cycle day to phase (never assign perimenstruation except on period days)
        if target_date in period_dates.values:
            return 'perimenstruation'
        else:
            return self._map_cycle_day_to_nonperi_phase(cycle_day, cycle_length, subject_id)
    
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
    
    def _map_cycle_day_to_phase(self, cycle_day: int, cycle_length: float = 28.0, subject_id: int = None) -> str:
        """
        Map cycle day to menstrual phase using realistic phase durations.
        
        Args:
            cycle_day: Day of menstrual cycle (1 to cycle_length)
            cycle_length: Length of the menstrual cycle in days
            subject_id: Subject ID for individualization (optional)
            
        Returns:
            str: Phase name
        """
        # Calculate phase boundaries for this cycle length and subject
        boundaries = self._calculate_phase_boundaries(cycle_length, subject_id)
        
        # Find which phase this cycle day belongs to
        for phase, (start_day, end_day) in boundaries.items():
            if start_day <= cycle_day <= end_day:
                return phase
        
        # Fallback: if cycle day is outside calculated boundaries, use proportional mapping
        return self._fallback_phase_mapping(cycle_day, cycle_length)
    
    def _fallback_phase_mapping(self, cycle_day: int, cycle_length: float) -> str:
        """
        Fallback phase mapping using proportional day ranges.
        
        Args:
            cycle_day: Day of menstrual cycle
            cycle_length: Length of the menstrual cycle in days
            
        Returns:
            str: Phase name
        """
        # Calculate proportional boundaries based on mean durations
        total_mean_duration = sum(self.phase_durations_config[phase]['mean'] for phase in self.phases)
        
        current_day = 1
        for phase in self.phases:
            if phase in self.phase_durations_config:
                # Calculate proportional duration for this cycle length
                mean_duration = self.phase_durations_config[phase]['mean']
                proportional_duration = int((mean_duration / total_mean_duration) * cycle_length)
                
                start_day = current_day
                end_day = current_day + proportional_duration - 1
                
                if start_day <= cycle_day <= end_day:
                    return phase
                
                current_day += proportional_duration
        
        # If we get here, return the last phase
        return self.phases[-1]
    
    def _map_cycle_day_to_nonperi_phase(self, cycle_day: int, cycle_length: float = 28.0, subject_id: int = None) -> str:
        """
        Map cycle day to menstrual phase using realistic phase durations, but never assign perimenstruation.
        """
        # Calculate phase boundaries for this cycle length and subject
        boundaries = self._calculate_phase_boundaries(cycle_length, subject_id)
        # Remove perimenstruation from boundaries
        nonperi_boundaries = {phase: bounds for phase, bounds in boundaries.items() if phase != 'perimenstruation'}
        # Find which phase this cycle day belongs to
        for phase, (start_day, end_day) in nonperi_boundaries.items():
            if start_day <= cycle_day <= end_day:
                return phase
        # Fallback: if cycle day is outside calculated boundaries, use proportional mapping (excluding perimenstruation)
        return self._fallback_nonperi_phase_mapping(cycle_day, cycle_length)

    def _fallback_nonperi_phase_mapping(self, cycle_day: int, cycle_length: float) -> str:
        """
        Fallback phase mapping using proportional day ranges, but never assign perimenstruation.
        """
        nonperi_phases = [phase for phase in self.phases if phase != 'perimenstruation']
        total_mean_duration = sum(self.phase_durations_config[phase]['mean'] for phase in nonperi_phases)
        current_day = 1
        for phase in nonperi_phases:
            mean_duration = self.phase_durations_config[phase]['mean']
            proportional_duration = int((mean_duration / total_mean_duration) * cycle_length)
            start_day = current_day
            end_day = current_day + proportional_duration - 1
            if start_day <= cycle_day <= end_day:
                return phase
            current_day += proportional_duration
        return nonperi_phases[-1]
    
    def predict_phase_from_hormones(self, estradiol: float, progesterone: float, testosterone: float) -> str:
        """
        Predict phase using hormone-based rules.
        All hormone rules are commented out; always returns None.
        """
        # # Rule 1: Clear estradiol peak (periovulation)
        # if estradiol > 2.0:
        #     return 'periovulation'
        # 
        # # Rule 2: High progesterone indicates luteal phases
        # if progesterone > 150:
        #     if progesterone > 250:
        #         return 'mid_late_luteal'
        #     else:
        #         return 'early_luteal'
        # 
        # # Rule 3: Low estradiol and low progesterone = mid-follicular
        # if estradiol < 1.5 and progesterone < 120:
        #     return 'mid_follicular'
        # 
        # # Rule 4: Moderate estradiol but low progesterone = late follicular
        # if 1.5 <= estradiol <= 2.0 and progesterone < 150:
        #     return 'periovulation'
        # 
        # # Rule 5: High progesterone but moderate estradiol = early luteal
        # if 120 <= progesterone <= 150 and estradiol < 2.0:
        #     return 'early_luteal'
        return None

    def predict_phases(self, hormone_data: pd.DataFrame) -> np.ndarray:
        """
        Predict phases for hormone data using rule-based logic.
        Prioritizes period data for perimenstruation detection.
        
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
            
            # First, check if this is a period day using period data
            if 'date' in hormone_data.columns and self.period_df is not None:
                target_date = pd.to_datetime(row['date'])
                period_phase = self.predict_phase_from_period_data(subject_id, target_date)
                
                # If period data indicates perimenstruation, use it
                if period_phase == 'perimenstruation':
                    predictions.append('perimenstruation')
                    continue
            
            # For non-period days, try hormone-based rules first
            hormone_phase = self.predict_phase_from_hormones(
                row['estradiol'], row['progesterone'], row['testosterone']
            )
            
            if hormone_phase:
                predictions.append(hormone_phase)
            else:
                # Fall back to temporal rules if hormone rules don't match
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
    
    def generate_hormone_rule_features(self, hormone_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate hormone rule-based features for ML models.
        These features encode the hormone patterns that indicate different phases.
        
        Args:
            hormone_data: DataFrame with hormone measurements
            
        Returns:
            pd.DataFrame: DataFrame with hormone rule features
        """
        features = pd.DataFrame(index=hormone_data.index)
        
        # Feature 1: Estradiol peak indicator (periovulation)
        features['estradiol_peak'] = (hormone_data['estradiol'] > 4.0).astype(int)
        
        # Feature 2: Progesterone peak indicator (mid_late_luteal)
        features['progesterone_peak'] = (hormone_data['progesterone'] > 300).astype(int)
        
        # Feature 3: E2/P4 ratio features (with safe division)
        ratio = hormone_data['estradiol'] / (hormone_data['progesterone'] + 1e-8)
        features['e2_p4_high_ratio'] = (ratio > 0.02).astype(int)  # periovulation
        features['e2_p4_low_ratio'] = (ratio < 0.005).astype(int)  # mid_late_luteal
        
        # Feature 4: Hormone level ranges
        features['estradiol_low'] = (hormone_data['estradiol'] < 2.0).astype(int)
        features['estradiol_moderate'] = ((hormone_data['estradiol'] >= 2.0) & (hormone_data['estradiol'] <= 4.0)).astype(int)
        features['estradiol_high'] = (hormone_data['estradiol'] > 4.0).astype(int)
        
        features['progesterone_low'] = (hormone_data['progesterone'] < 150).astype(int)
        features['progesterone_moderate'] = ((hormone_data['progesterone'] >= 150) & (hormone_data['progesterone'] <= 300)).astype(int)
        features['progesterone_high'] = (hormone_data['progesterone'] > 300).astype(int)
        
        # Feature 5: Combined hormone patterns
        features['mid_follicular_pattern'] = (
            (hormone_data['estradiol'] >= 2.0) & 
            (hormone_data['estradiol'] <= 4.0) & 
            (hormone_data['progesterone'] < 200)
        ).astype(int)
        
        features['early_luteal_pattern'] = (
            (hormone_data['progesterone'] >= 150) & 
            (hormone_data['progesterone'] <= 300) & 
            (hormone_data['estradiol'] >= 1.5) & 
            (hormone_data['estradiol'] <= 3.0)
        ).astype(int)
        
        # Feature 6: Raw hormone values (safely normalized)
        estradiol_max = hormone_data['estradiol'].max()
        progesterone_max = hormone_data['progesterone'].max()
        testosterone_max = hormone_data['testosterone'].max()
        
        features['estradiol_norm'] = hormone_data['estradiol'] / (estradiol_max + 1e-8)
        features['progesterone_norm'] = hormone_data['progesterone'] / (progesterone_max + 1e-8)
        features['testosterone_norm'] = hormone_data['testosterone'] / (testosterone_max + 1e-8)
        
        # Feature 7: Hormone ratios (with bounds)
        features['e2_p4_ratio'] = np.clip(ratio, 0, 1.0)  # Clip to prevent extreme values
        features['e2_t_ratio'] = np.clip(hormone_data['estradiol'] / (hormone_data['testosterone'] + 1e-8), 0, 1.0)
        features['p4_t_ratio'] = np.clip(hormone_data['progesterone'] / (hormone_data['testosterone'] + 1e-8), 0, 1.0)
        
        # Feature 8: Hormone interactions (with bounds)
        features['e2_p4_interaction'] = np.clip(hormone_data['estradiol'] * hormone_data['progesterone'] / 1000, 0, 10)
        features['e2_t_interaction'] = np.clip(hormone_data['estradiol'] * hormone_data['testosterone'] / 1000, 0, 10)
        features['p4_t_interaction'] = np.clip(hormone_data['progesterone'] * hormone_data['testosterone'] / 1000, 0, 10)
        
        # Feature 9: Period indicators (if available)
        if 'date' in hormone_data.columns and self.period_df is not None:
            period_indicators = []
            for _, row in hormone_data.iterrows():
                subject_id = row['subject_id']
                target_date = pd.to_datetime(row['date'])
                period_phase = self.predict_phase_from_period_data(subject_id, target_date)
                period_indicators.append(1 if period_phase == 'perimenstruation' else 0)
            features['period_indicator'] = period_indicators
        else:
            features['period_indicator'] = 0
        
        # Feature 10: Confidence scores for hormone rules
        confidence_scores = []
        for i, row in hormone_data.iterrows():
            confidence = 0.0
            
            # High confidence for clear hormone patterns
            if row['estradiol'] > 4.0:
                confidence += 0.8
            elif row['progesterone'] > 300:
                confidence += 0.8
            elif ratio.iloc[i] > 0.02:
                confidence += 0.6
            elif ratio.iloc[i] < 0.005:
                confidence += 0.6
            elif features.loc[i, 'mid_follicular_pattern']:
                confidence += 0.4
            elif features.loc[i, 'early_luteal_pattern']:
                confidence += 0.4
            else:
                confidence += 0.1  # Low confidence for unclear patterns
                
            confidence_scores.append(min(confidence, 1.0))
        
        features['hormone_rule_confidence'] = confidence_scores
        
        return features

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
                
                if predicted_phase == 'perimenstruation':
                    # Hard probabilities for perimenstruation (1.0 for predicted, 0.0 for others)
                    probabilities[i, phase_idx] = 1.0
                    # All other phases get 0.0 probability
                    for j in range(n_phases):
                        if j != phase_idx:
                            probabilities[i, j] = 0.0
                else:
                    # Soft probabilities for other phases (0.8 for predicted, 0.2 distributed)
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