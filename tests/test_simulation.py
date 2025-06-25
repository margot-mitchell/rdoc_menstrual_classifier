"""
Tests for simulation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main.simulation import simulate_hormone_and_period_data, generate_survey_responses
from src.utils.data_loader import load_config
from src.utils.evaluator import check_survey_accuracy


class TestSimulation(unittest.TestCase):
    """Test cases for simulation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal test configuration
        self.test_config = {
            'simulation': {
                'n_subjects': 10,
                'n_hormone_samples': 20,
                'n_period_days': 50,
                'random_seed': 42
            },
            'cycle_length': {
                'mean': 28.9,
                'sd': 2.5,
                'min': 22,
                'max': 36
            }
        }
    
    def test_simulate_hormone_and_period_data(self):
        """Test hormone and period data simulation."""
        # Run simulation
        hormone_df, period_df, pattern_df, survey_df = simulate_hormone_and_period_data(self.test_config)
        
        # Check data types
        self.assertIsInstance(hormone_df, pd.DataFrame)
        self.assertIsInstance(period_df, pd.DataFrame)
        self.assertIsInstance(pattern_df, pd.DataFrame)
        self.assertIsInstance(survey_df, pd.DataFrame)
        
        # Check expected number of subjects
        self.assertEqual(len(pattern_df), self.test_config['simulation']['n_subjects'])
        self.assertEqual(len(survey_df), self.test_config['simulation']['n_subjects'])
        
        # Check hormone data columns
        expected_hormone_cols = ['subject_id', 'estradiol', 'progesterone', 'testosterone']
        for col in expected_hormone_cols:
            self.assertIn(col, hormone_df.columns)
        
        # Check period data columns
        expected_period_cols = ['subject_id', 'date', 'period']
        for col in expected_period_cols:
            self.assertIn(col, period_df.columns)
        
        # Check pattern data columns
        self.assertIn('subject_id', pattern_df.columns)
        self.assertIn('menstrual_pattern', pattern_df.columns)
        
        # Check survey data columns
        self.assertIn('subject_id', survey_df.columns)
        self.assertIn('cycle_length', survey_df.columns)
    
    def test_generate_survey_responses(self):
        """Test survey response generation."""
        # Create test data
        hormone_df, period_df, pattern_df, survey_df = simulate_hormone_and_period_data(self.test_config)
        
        # Generate survey responses
        survey_responses_df = generate_survey_responses(period_df, survey_df)
        
        # Check data type
        self.assertIsInstance(survey_responses_df, pd.DataFrame)
        
        # Check expected columns
        expected_cols = ['subject_id', 'menstrual_pattern', 'cycle_length', 
                        'date_of_response', 'date_of_last_period']
        for col in expected_cols:
            self.assertIn(col, survey_responses_df.columns)
        
        # Check number of responses
        self.assertEqual(len(survey_responses_df), len(survey_df))
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
simulation:
  n_subjects: 5
  n_hormone_samples: 10
  n_period_days: 30
  random_seed: 123
cycle_length:
  mean: 28.0
  sd: 2.0
  min: 24
  max: 32
            """)
            config_path = f.name
        
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Check configuration structure
            self.assertIn('simulation', config)
            self.assertIn('cycle_length', config)
            self.assertEqual(config['simulation']['n_subjects'], 5)
            self.assertEqual(config['cycle_length']['mean'], 28.0)
        
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_hormone_data_ranges(self):
        """Test that hormone data is within expected ranges."""
        hormone_df, _, _, _ = simulate_hormone_and_period_data(self.test_config)
        
        # Check hormone value ranges (basic sanity checks)
        for hormone in ['estradiol', 'progesterone', 'testosterone']:
            values = hormone_df[hormone]
            self.assertTrue(all(values >= 0), f"{hormone} values should be non-negative")
            self.assertTrue(len(values) > 0, f"{hormone} should have data")
    
    def test_cycle_length_distribution(self):
        """Test that cycle lengths follow expected distribution."""
        _, _, _, survey_df = simulate_hormone_and_period_data(self.test_config)
        
        cycle_lengths = survey_df['cycle_length']
        
        # Check that cycle lengths are within expected range
        config_cycle = self.test_config['cycle_length']
        self.assertTrue(all(cycle_lengths >= config_cycle['min']))
        self.assertTrue(all(cycle_lengths <= config_cycle['max']))
        
        # Check that mean is reasonable (within 2 SD of expected)
        expected_mean = config_cycle['mean']
        actual_mean = cycle_lengths.mean()
        self.assertLess(abs(actual_mean - expected_mean), 2 * config_cycle['sd'])

    def test_survey_accuracy_analysis(self):
        """Test survey accuracy analysis on simulated data."""
        # Load simulation config
        config = load_config('config/simulation_config.yaml')
        output_config = config['output']
        data_dir = output_config['data_dir']
        reports_dir = output_config['reports_dir']

        survey_file = os.path.join(data_dir, 'survey_responses.csv')
        period_file = os.path.join(data_dir, 'period_sleep_data.csv')

        # Check if files exist (skip test if not)
        if not (os.path.exists(survey_file) and os.path.exists(period_file)):
            self.skipTest("Survey or period data file not found. Run simulation first.")

        # Run the survey accuracy check
        results_df = check_survey_accuracy(
            survey_file=survey_file,
            period_file=period_file,
            output_dir=reports_dir
        )

        # Basic checks
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn('subject_id', results_df.columns)
        self.assertIn('reported_last_period', results_df.columns)
        self.assertTrue(os.path.exists(os.path.join(reports_dir, 'survey_accuracy_analysis.csv')))


if __name__ == '__main__':
    unittest.main() 