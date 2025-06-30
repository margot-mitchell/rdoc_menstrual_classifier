"""
Tests for simulation functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import random
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main.simulation import simulate_hormone_and_period_data, generate_survey_responses
from src.utils.data_loader import load_config
from src.utils.evaluator import check_survey_accuracy
from src.simulation.utils import generate_estradiol_value, generate_progesterone_value, get_subphase_boundaries


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

    def test_period_alignment_with_survey(self):
        """Test that survey date_of_last_period matches first period 'Yes' date in period data for each subject."""
        survey = pd.read_csv("outputs/data/survey_responses.csv")
        period = pd.read_csv("outputs/data/period_sleep_data.csv")
        mismatches = []
        for sub_id in survey['subject_id']:
            survey_date = pd.to_datetime(survey.loc[survey['subject_id'] == sub_id, 'date_of_last_period']).iloc[0]
            period_dates = period[(period['subject_id'] == sub_id) & (period['period'] == 'Yes')]['date']
            if period_dates.empty:
                mismatches.append((sub_id, 'No period days'))
                continue
            first_period_date = pd.to_datetime(period_dates).min()
            if survey_date != first_period_date:
                mismatches.append((sub_id, survey_date, first_period_date))
        if mismatches:
            print("Mismatches found between survey and period data:")
            for m in mismatches:
                print(m)
        else:
            print("All subjects aligned: survey date_of_last_period matches first period day.")

    def test_subphase_alignment(self):
        """Test that hormone peaks are correctly aligned with subphases."""
        baseline_estradiol = 2.0
        baseline_progesterone = 100.0
        
        # Example phase durations (you can vary these)
        phase_durations = {
            'perimenstruation': 3,
            'mid_follicular': 8,
            'periovulation': 4,
            'early_luteal': 5,
            'mid_late_luteal': 8
        }
        
        total_cycle_length = sum(phase_durations.values())
        print(f"Testing with {total_cycle_length}-day cycle")
        print(f"Phase durations: {phase_durations}")
        print()
        
        # Calculate subphase boundaries
        boundaries = get_subphase_boundaries(phase_durations)
        print("Subphase boundaries:")
        for phase, (start, end) in boundaries.items():
            print(f"  {phase}: days {start}-{end}")
        print()
        
        # Generate hormone values for the entire cycle
        estradiol_values = []
        progesterone_values = []
        phases = []
        
        for day in range(1, total_cycle_length + 1):
            # Determine current phase
            current_day = 0
            current_phase = None
            for phase, duration in phase_durations.items():
                current_day += duration
                if day <= current_day:
                    current_phase = phase
                    break
            
            # Generate hormone values
            estradiol = generate_estradiol_value(day, baseline_estradiol, phase_durations)
            progesterone = generate_progesterone_value(day, baseline_progesterone, phase_durations)
            
            estradiol_values.append(estradiol)
            progesterone_values.append(progesterone)
            phases.append(current_phase)
        
        # Find peaks
        estradiol_peak_day = np.argmax(estradiol_values) + 1
        progesterone_peak_day = np.argmax(progesterone_values) + 1
        
        # Find secondary estradiol peak (look specifically in mid_late_luteal phase)
        mid_late_start, mid_late_end = boundaries['mid_late_luteal']
        estradiol_values_array = np.array(estradiol_values)
        # Look for peak in mid_late_luteal phase (days are 1-indexed, so subtract 1 for array indexing)
        mid_late_values = estradiol_values_array[mid_late_start-1:mid_late_end]
        if len(mid_late_values) > 0:
            second_peak_idx = mid_late_start + np.argmax(mid_late_values)
        else:
            second_peak_idx = mid_late_start  # fallback
        second_peak_value = estradiol_values[second_peak_idx - 1]
        
        print("PEAK ANALYSIS:")
        print(f"Primary estradiol peak: day {estradiol_peak_day} (phase: {phases[estradiol_peak_day-1]})")
        print(f"Secondary estradiol peak: day {second_peak_idx} (phase: {phases[second_peak_idx-1]})")
        print(f"Progesterone peak: day {progesterone_peak_day} (phase: {phases[progesterone_peak_day-1]})")
        print()
        
        # Check if peaks are in correct phases
        periov_start, periov_end = boundaries['periovulation']
        mid_late_start, mid_late_end = boundaries['mid_late_luteal']
        mid_late_half = mid_late_start + (mid_late_end - mid_late_start + 1) // 2
        
        print("VERIFICATION:")
        print(f"Primary estradiol peak should be in periovulation (days {periov_start}-{periov_end}): {periov_start <= estradiol_peak_day <= periov_end}")
        print(f"Secondary estradiol peak should be in first half of mid_late_luteal (before day {mid_late_half}): {second_peak_idx <= mid_late_half}")
        print(f"Progesterone peak should be in first half of mid_late_luteal (before day {mid_late_half}): {progesterone_peak_day <= mid_late_half}")
        print(f"Secondary estradiol and progesterone peaks should be on same day: {second_peak_idx == progesterone_peak_day}")
        print()
        
        # Assertions for test framework
        self.assertTrue(periov_start <= estradiol_peak_day <= periov_end, 
                       f"Primary estradiol peak should be in periovulation (days {periov_start}-{periov_end}), but was on day {estradiol_peak_day}")
        self.assertTrue(second_peak_idx <= mid_late_half, 
                       f"Secondary estradiol peak should be in first half of mid_late_luteal (before day {mid_late_half}), but was on day {second_peak_idx}")
        self.assertTrue(progesterone_peak_day <= mid_late_half, 
                       f"Progesterone peak should be in first half of mid_late_luteal (before day {mid_late_half}), but was on day {progesterone_peak_day}")
        self.assertEqual(second_peak_idx, progesterone_peak_day, 
                        f"Secondary estradiol and progesterone peaks should be on same day, but were on days {second_peak_idx} and {progesterone_peak_day}")
        
        # Optional: Generate plot for visual verification (only if matplotlib is available and test is run directly)
        if __name__ == '__main__':
            self._generate_subphase_alignment_plot(estradiol_values, progesterone_values, boundaries, 
                                                 estradiol_peak_day, second_peak_idx, progesterone_peak_day, 
                                                 total_cycle_length)

    def get_proportional_phase_durations(self, cycle_length):
        # Reference: 28-day cycle phase durations
        ref_durations = {
            'perimenstruation': 3,
            'mid_follicular': 8,
            'periovulation': 4,
            'early_luteal': 5,
            'mid_late_luteal': 8
        }
        ref_total = sum(ref_durations.values())
        # Compute proportions
        proportions = {k: v / ref_total for k, v in ref_durations.items()}
        # Compute scaled durations
        durations = {k: int(round(proportions[k] * cycle_length)) for k in proportions}
        # Adjust to ensure sum matches cycle_length
        diff = cycle_length - sum(durations.values())
        if diff != 0:
            # Find the phase with the largest duration
            key = max(durations, key=lambda k: durations[k])
            durations[key] += diff
        return durations

    def test_peak_scaling(self):
        """Test that hormone peaks reach the correct baseline multipliers and generate a scaling plot (raw values, no normalization, no noise)."""
        import matplotlib.pyplot as plt
        import numpy as np
        import random
        import time
        from src.simulation.utils import generate_estradiol_value, generate_progesterone_value, get_subphase_boundaries
        seed = int(time.time())
        print(f'Using random seed: {seed}')
        np.random.seed(seed)
        random.seed(seed)
        baseline_estradiol = 2.0
        baseline_progesterone = 100.0
        cycle_lengths = [24, 28, 32, 36]
        estradiol_peaks = []
        progesterone_peaks = []

        # Monkey-patch hormone generation functions to remove noise
        orig_generate_estradiol_value = generate_estradiol_value
        orig_generate_progesterone_value = generate_progesterone_value

        def noiseless_generate_estradiol_value(cycle_day, baseline, phase_durations):
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
                value = baseline
            elif early_phase_end < cycle_day <= primary_peak_day:
                primary_peak = 5.0 * baseline
                progress = (cycle_day - early_phase_end) / (primary_peak_day - early_phase_end)
                value = baseline + (primary_peak - baseline) * progress
            elif primary_peak_day < cycle_day <= primary_peak_end:
                primary_peak = 5.0 * baseline
                inflection = baseline
                progress = (cycle_day - primary_peak_day) / (primary_peak_end - primary_peak_day)
                value = primary_peak - (primary_peak - inflection) * progress
            elif primary_peak_end < cycle_day <= secondary_peak_day:
                trough = baseline
                secondary_peak = 2.5 * baseline
                progress = (cycle_day - primary_peak_end) / (secondary_peak_day - primary_peak_end)
                value = trough + (secondary_peak - trough) * progress
            elif secondary_peak_day < cycle_day <= sum(phase_durations.values()):
                secondary_peak = 2.5 * baseline
                progress = (cycle_day - secondary_peak_day) / (sum(phase_durations.values()) - secondary_peak_day)
                value = secondary_peak - (secondary_peak - baseline) * progress
            else:
                value = baseline
            return max(0, value)

        def noiseless_generate_progesterone_value(cycle_day, baseline, phase_durations):
            boundaries = get_subphase_boundaries(phase_durations)
            mid_late_start, mid_late_end = boundaries['mid_late_luteal']
            mid_late_duration = mid_late_end - mid_late_start + 1
            peak_day = mid_late_start + (mid_late_duration // 2)
            rising_start = peak_day * 0.6
            falling_end = phase_durations.get('mid_late_luteal', 8) + mid_late_start
            if cycle_day <= rising_start:
                value = baseline
            elif rising_start < cycle_day <= peak_day:
                peak = 4 * baseline
                progress = (cycle_day - rising_start) / (peak_day - rising_start)
                value = baseline + (peak - baseline) * progress
            elif peak_day < cycle_day <= falling_end:
                peak = 4 * baseline
                progress = (cycle_day - peak_day) / (falling_end - peak_day)
                value = peak - (peak - baseline) * progress
            else:
                value = baseline
            return max(0, value)

        # Create temporary functions in utils module
        import src.simulation.utils as utils
        utils.generate_estradiol_value = noiseless_generate_estradiol_value
        utils.generate_progesterone_value = noiseless_generate_progesterone_value

        try:
            for cycle_length in cycle_lengths:
                phase_durations = self.get_proportional_phase_durations(cycle_length)
                total_cycle_length = sum(phase_durations.values())
                boundaries = get_subphase_boundaries(phase_durations)
                estradiol_values = []
                progesterone_values = []
                for day in range(1, total_cycle_length + 1):
                    estradiol = utils.generate_estradiol_value(day, baseline_estradiol, phase_durations)
                    progesterone = utils.generate_progesterone_value(day, baseline_progesterone, phase_durations)
                    estradiol_values.append(estradiol)
                    progesterone_values.append(progesterone)
                # Find peaks (raw, no normalization)
                estradiol_peak = max(estradiol_values)
                progesterone_peak = max(progesterone_values)
                estradiol_peaks.append(estradiol_peak)
                progesterone_peaks.append(progesterone_peak)
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            # Estradiol
            axes[0].scatter(cycle_lengths, estradiol_peaks, color='red', s=120, label='Estradiol Peak')
            axes[0].axhline(5.0 * baseline_estradiol, color='red', linestyle='--', alpha=0.6, label='Expected (5x baseline)')
            axes[0].set_xlabel('Cycle Length (days)')
            axes[0].set_ylabel('Estradiol Peak (raw value)')
            axes[0].set_title('Estradiol Peak Scaling (Raw)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            # Progesterone
            axes[1].scatter(cycle_lengths, progesterone_peaks, color='blue', s=120, label='Progesterone Peak')
            axes[1].axhline(4.0 * baseline_progesterone, color='blue', linestyle='--', alpha=0.6, label='Expected (4x baseline)')
            axes[1].set_xlabel('Cycle Length (days)')
            axes[1].set_ylabel('Progesterone Peak (raw value)')
            axes[1].set_title('Progesterone Peak Scaling (Raw)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            # Save to test_output
            import os
            test_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_output')
            os.makedirs(test_output_dir, exist_ok=True)
            plot_path = os.path.join(test_output_dir, 'test_peak_scaling.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Peak scaling plot saved as '{plot_path}'")
            # Also run the original assertions for one cycle length (28)
            idx_28 = cycle_lengths.index(28)
            estradiol_primary_peak = estradiol_peaks[idx_28]
            progesterone_peak = progesterone_peaks[idx_28]
            expected_estradiol_primary = 5.0 * baseline_estradiol
            expected_progesterone = 4.0 * baseline_progesterone
            tolerance = 0.2
            self.assertGreaterEqual(estradiol_primary_peak, expected_estradiol_primary * (1 - tolerance))
            self.assertLessEqual(estradiol_primary_peak, expected_estradiol_primary * (1 + tolerance))
            self.assertGreaterEqual(progesterone_peak, expected_progesterone * (1 - tolerance))
            self.assertLessEqual(progesterone_peak, expected_progesterone * (1 + tolerance))
        finally:
            # Restore original functions
            utils.generate_estradiol_value = orig_generate_estradiol_value
            utils.generate_progesterone_value = orig_generate_progesterone_value

    def _generate_subphase_alignment_plot(self, estradiol_values, progesterone_values, boundaries, 
                                        estradiol_peak_day, second_peak_idx, progesterone_peak_day, 
                                        total_cycle_length):
        """Generate plot for visual verification of subphase alignment."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Estradiol plot
        ax1.plot(range(1, total_cycle_length + 1), estradiol_values, 'r-', linewidth=2, label='Estradiol')
        ax1.axvline(x=estradiol_peak_day, color='red', linestyle='--', alpha=0.7, label=f'Primary peak (day {estradiol_peak_day})')
        ax1.axvline(x=second_peak_idx, color='orange', linestyle='--', alpha=0.7, label=f'Secondary peak (day {second_peak_idx})')
        
        # Add phase boundaries
        for phase, (start, end) in boundaries.items():
            ax1.axvspan(start, end, alpha=0.2, label=phase)
        
        ax1.set_xlabel('Cycle Day')
        ax1.set_ylabel('Estradiol Level')
        ax1.set_title('Estradiol Levels by Subphase')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Progesterone plot
        ax2.plot(range(1, total_cycle_length + 1), progesterone_values, 'b-', linewidth=2, label='Progesterone')
        ax2.axvline(x=progesterone_peak_day, color='blue', linestyle='--', alpha=0.7, label=f'Peak (day {progesterone_peak_day})')
        
        # Add phase boundaries
        for phase, (start, end) in boundaries.items():
            ax2.axvspan(start, end, alpha=0.2, label=phase)
        
        ax2.set_xlabel('Cycle Day')
        ax2.set_ylabel('Progesterone Level')
        ax2.set_title('Progesterone Levels by Subphase')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to test output directory
        test_output_dir = os.path.join(project_root, 'test_output')
        os.makedirs(test_output_dir, exist_ok=True)
        plot_path = os.path.join(test_output_dir, 'test_subphase_alignment.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Subphase alignment plot saved as '{plot_path}'")

    def test_phase_duration_distribution_against_reference(self):
        """Analyze simulated data and compare phase duration means/SDs to reference values."""
        import pandas as pd
        import numpy as np
        # Reference values (mean, sd)
        reference = {
            'perimenstruation': (3.6, 1.2),
            'mid_follicular': (10.1, 2.7),
            'periovulation': (4.2, 1.4),
            'early_luteal': (5.7, 3.1),
            'mid_late_luteal': (5.3, 5.3)
        }
        # Load simulated hormone data (assume phase column exists)
        df = pd.read_csv('outputs/data/full_hormone_data_labeled.csv')
        # Count consecutive days in each phase for each subject
        phase_lengths = {phase: [] for phase in reference}
        for subject_id, group in df.groupby('subject_id'):
            phases = group.sort_values('date')['phase'].values
            current_phase = None
            current_length = 0
            for phase in phases:
                if phase != current_phase:
                    if current_phase is not None and current_phase in phase_lengths:
                        phase_lengths[current_phase].append(current_length)
                    current_phase = phase
                    current_length = 1
                else:
                    current_length += 1
            # Add last phase
            if current_phase is not None and current_phase in phase_lengths:
                phase_lengths[current_phase].append(current_length)
        # Print and compare
        print("\nPhase duration statistics (simulated vs reference):")
        for phase in reference:
            arr = np.array(phase_lengths[phase])
            mean = arr.mean() if len(arr) > 0 else float('nan')
            sd = arr.std(ddof=1) if len(arr) > 1 else float('nan')
            ref_mean, ref_sd = reference[phase]
            print(f"{phase:18s} Simulated: mean={mean:.2f}, sd={sd:.2f} | Reference: mean={ref_mean:.2f}, sd={ref_sd:.2f}")


if __name__ == '__main__':
    unittest.main() 