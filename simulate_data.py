import pandas as pd
from datetime import datetime, timedelta
from src.config.hormone_config import (
    get_estradiol_distribution, 
    get_progesterone_distribution,
    get_testosterone_distribution,
    sample_lognormal_from_distribution,
    sample_baseline_values,
    generate_progesterone_value,
    generate_estradiol_value
)
from src.config.phase_config import (
    generate_phase_durations,
    get_phase_from_cycle_day
)
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import os
import json

def simulate_hormone_and_period_data(n_subjects=100, n_hormone_samples=70, n_period_days=150):
    """
    Simulate hormone and period data for multiple subjects.
    
    Args:
        n_subjects (int): Number of subjects to simulate
        n_hormone_samples (int): Number of days of hormone data to generate
        n_period_days (int): Number of days of period data to generate (default: 150)
    
    Returns:
        tuple: (hormone_data, period_data, menstrual_patterns) DataFrames containing simulated data and patterns
    """
    # Define possible menstrual patterns
    pattern_options = [
        'Extremely regular (no more than 1-2 days before or after expected)',
        'Very regular (within 3-4 days)',
        'Regular (within 5-7 days)',
        # 'Usually irregular',  # Commented out as we don't want to assign this value
        # 'No periods'  # Commented out as we don't want to assign this value
    ]
    
    # Assign patterns to subjects
    subject_patterns = np.random.choice(pattern_options, size=n_subjects)
    
    # Generate random start dates for each subject between 2025/1/1 and 2026/12/31
    start_dates = pd.date_range(start='2025-01-01', end='2026-12-31', periods=n_subjects).date.tolist()
    np.random.shuffle(start_dates)
    
    # Generate hormone data
    hormone_data = []
    period_data = []
    
    for subject_id in range(n_subjects):
        # Get this subject's menstrual pattern
        pattern = subject_patterns[subject_id]
        
        # Adjust phase duration variability based on pattern
        if pattern == 'Extremely regular (no more than 1-2 days before or after expected)':
            # Very low variability in phase durations
            phase_duration_sd_multiplier = 0.3
        elif pattern == 'Very regular (within 3-4 days)':
            # Low variability in phase durations
            phase_duration_sd_multiplier = 0.6
        else:  # 'Regular (within 5-7 days)'
            # Moderate variability in phase durations
            phase_duration_sd_multiplier = 1.0
        
        # Sample baseline values for this subject
        baselines = sample_baseline_values()
        
        # Generate phase durations for this subject with adjusted variability
        phase_durations = generate_phase_durations(sd_multiplier=phase_duration_sd_multiplier)
        total_cycle_length = sum(phase_durations.values())
        
        # Randomly select a starting point in the cycle
        start_idx = np.random.randint(0, total_cycle_length)
        
        # Generate hormone data for n_hormone_samples days
        for day in range(n_hormone_samples):
            # Calculate cycle day (1 to total_cycle_length)
            cycle_day = ((day + start_idx) % total_cycle_length) + 1
            
            # Get current phase
            current_phase = get_phase_from_cycle_day(cycle_day, phase_durations)
            
            # Generate hormone values
            estradiol = generate_estradiol_value(cycle_day, baselines['estradiol'])
            progesterone = generate_progesterone_value(cycle_day, baselines['progesterone'])
            testosterone = sample_lognormal_from_distribution(get_testosterone_distribution(cycle_day=cycle_day))
            
            hormone_data.append({
                'subject_id': subject_id,
                'date': (start_dates[subject_id] + timedelta(days=day)).strftime('%Y-%m-%d'),
                'cycle_day': cycle_day,
                'estradiol': estradiol,
                'progesterone': progesterone,
                'testosterone': testosterone,
                'phase': current_phase
            })
        
        # Generate period data using the same cycle alignment
        for day in range(n_period_days):
            cycle_day = ((day + start_idx) % total_cycle_length) + 1
            current_phase = get_phase_from_cycle_day(cycle_day, phase_durations)
            
            # Determine if it's a period day (perimenstruation phase)
            is_period = current_phase == 'perimenstruation'
            
            # Generate flow value (Heavy/Light/Medium/N/A) during period days
            flow = 'N/A'
            if is_period:
                # Flow is heaviest on first day, then decreases
                phase_day = cycle_day - sum(d for p, d in phase_durations.items() if p != 'perimenstruation')
                if phase_day == 1:
                    flow = 'Heavy'
                else:
                    flow = np.random.choice(['Light', 'Medium'])
            
            # Generate spotting value (Yes/No)
            # Higher chance of spotting around ovulation (periovulation phase) and before period
            spotting = 'No'
            if not is_period:
                if current_phase == 'periovulation':
                    spotting = 'Yes' if np.random.random() < 0.3 else 'No'  # 30% chance
                elif current_phase == 'mid_late_luteal' and cycle_day == total_cycle_length:
                    spotting = 'Yes' if np.random.random() < 0.2 else 'No'  # 20% chance
            
            # Generate sleep data - completely random between 4-11 hours
            sleep_hours = np.random.uniform(4, 11)
            
            period_data.append({
                'subject_id': subject_id,
                'date': (start_dates[subject_id] + timedelta(days=day)).strftime('%Y-%m-%d'),
                'cycle_day': cycle_day,
                'period': 'Yes' if is_period else 'No',
                'flow': flow,
                'spotting': spotting,
                'sleep_hours': sleep_hours,
                'phase': current_phase
            })
    
    hormone_df = pd.DataFrame(hormone_data)
    period_df = pd.DataFrame(period_data)
    
    # Create DataFrame of subject patterns
    pattern_df = pd.DataFrame({
        'subject_id': range(n_subjects),
        'menstrual_pattern': subject_patterns
    })
    
    return hormone_df, period_df, pattern_df

def plot_hormone_cycles(df, subject_id=0, output_path='output/hormone_cycles.png'):
    """
    Plot hormone levels over time for a single subject, showing the full 70 days of data
    but labeling x-axis with the 10 sample points used in unlabeled data.
    Uses color scheme to show phases and includes a phase legend.
    
    Args:
        df (pd.DataFrame): DataFrame containing hormone data
        subject_id (int): ID of the subject to plot (default: 0)
        output_path (str): Path to save the plot
    """
    # Define phase colors
    phase_colors = {
        'perimenstruation': '#FFB6C1',  # Light pink
        'mid_follicular': '#98FB98',    # Light green
        'periovulation': '#87CEEB',     # Sky blue
        'early_luteal': '#DDA0DD',      # Plum
        'mid_late_luteal': '#F0E68C'    # Khaki
    }
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 12))
    ax2 = ax1.twinx()
    
    # Get data for specified subject and sort by date
    subject_data = df[df['subject_id'] == subject_id].sort_values('date')
    
    # Create sample numbers (1 to 70)
    sample_numbers = range(1, len(subject_data) + 1)
    
    # Plot E2 and P4 on first subplot
    ax1.plot(sample_numbers, subject_data['estradiol'], linestyle='--', color='red', label='Estradiol')
    ax2.plot(sample_numbers, subject_data['progesterone'], linestyle='-', color='blue', label='Progesterone')
    
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Estradiol (pg/ml)', color='red')
    ax2.set_ylabel('Progesterone (pg/ml)', color='blue')
    ax1.set_ylim(0, 25)
    ax2.set_ylim(0, 1200)
    
    # Add phase background colors
    # Create a list of phases and their corresponding sample numbers
    phases = []
    current_phase = None
    phase_start = 1
    
    for i, (_, row) in enumerate(subject_data.iterrows(), 1):
        if row['phase'] != current_phase:
            if current_phase is not None:
                phases.append((current_phase, phase_start, i))
            current_phase = row['phase']
            phase_start = i
    
    # Add the last phase
    if current_phase is not None:
        phases.append((current_phase, phase_start, len(subject_data)))
    
    # Create phase patches for legend
    phase_patches = []
    for phase, color in phase_colors.items():
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=phase.replace('_', ' ').title())
        phase_patches.append(patch)
    
    # Add colored backgrounds for each phase
    for phase, start, end in phases:
        ax1.axvspan(start, end, color=phase_colors[phase], alpha=0.3)
        ax3.axvspan(start, end, color=phase_colors[phase], alpha=0.3)
    
    # Add hormone legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    hormone_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add phase legend
    ax1.add_artist(hormone_legend)  # Add back hormone legend
    ax1.legend(handles=phase_patches, loc='upper left', title='Menstrual Phases')
    
    ax1.set_title(f'Estradiol and Progesterone Levels - Subject {subject_id}')
    
    # Plot Testosterone on second subplot
    ax3.plot(sample_numbers, subject_data['testosterone'], linestyle='-', color='green', label='Testosterone')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('Testosterone (pg/ml)', color='green')
    ax3.set_ylim(0, 700)
    
    # Add phase legend to testosterone plot
    ax3.legend(handles=phase_patches, loc='upper left', title='Menstrual Phases')
    ax3.legend(loc='upper right')
    ax3.set_title(f'Testosterone Levels - Subject {subject_id}')
    
    # Set x-axis ticks to show only the 10 sample points
    sample_points = range(1, len(subject_data) + 1, 7)[:10]  # Get positions of 10 samples
    sample_labels = [f'sample_{i+1}' for i in range(10)]  # Create labels sample_1 to sample_10
    
    for ax in [ax1, ax3]:
        ax.set_xticks(sample_points)
        ax.set_xticklabels(sample_labels)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f'Hormone cycle plot for subject {subject_id} saved to {output_path}')

def get_date_of_last_period(period_df, subject_id):
    """Get the date of the last period for a subject."""
    subject_data = period_df[period_df['subject_id'] == subject_id]
    period_dates = subject_data[subject_data['period'] == 'Yes']['date'].tolist()
    return period_dates[0] if period_dates else None

def get_date_of_response(period_df, subject_id):
    """Get a random date for the survey response."""
    subject_data = period_df[period_df['subject_id'] == subject_id]
    return np.random.choice(subject_data['date'].tolist())

def calculate_actual_cycle_lengths(labeled_df):
    """
    Calculate actual cycle lengths for each subject from labeled data.
    
    Args:
        labeled_df (pd.DataFrame): DataFrame containing labeled hormone data
    
    Returns:
        dict: Dictionary mapping subject_id to mean cycle length
    """
    cycle_lengths = {}
    
    for subject_id in labeled_df['subject_id'].unique():
        subject_data = labeled_df[labeled_df['subject_id'] == subject_id].copy()
        subject_data['date'] = pd.to_datetime(subject_data['date'])
        subject_data = subject_data.sort_values('date')
        
        # Find cycle starts (first day of follicular phase)
        cycle_starts = subject_data[
            (subject_data['phase'] == 'mid_follicular') & 
            (subject_data['phase'].shift(1) != 'mid_follicular')
        ]['date'].tolist()
        
        if len(cycle_starts) > 1:
            # Calculate cycle lengths
            cycle_diffs = [(cycle_starts[i] - cycle_starts[i-1]).days 
                          for i in range(1, len(cycle_starts))]
            mean_length = int(round(np.mean(cycle_diffs)))
            cycle_lengths[subject_id] = mean_length
    
    return cycle_lengths

def update_survey_responses(survey_df, cycle_lengths):
    """
    Update survey responses to match actual cycle lengths.
    
    Args:
        survey_df (pd.DataFrame): DataFrame containing survey responses
        cycle_lengths (dict): Dictionary mapping subject_id to mean cycle length
    
    Returns:
        pd.DataFrame: Updated survey responses
    """
    updated_survey = survey_df.copy()
    
    for subject_id, actual_length in cycle_lengths.items():
        if subject_id in updated_survey['subject_id'].values:
            mask = updated_survey['subject_id'] == subject_id
            updated_survey.loc[mask, 'length_of_typical_cycle'] = actual_length
    
    return updated_survey

def calculate_metrics(hormone_df):
    """
    Calculate and save metrics about the hormone data.
    
    Args:
        hormone_df (pd.DataFrame): DataFrame containing hormone data
    """
    # Calculate mean hormone levels by phase
    phase_means = hormone_df.groupby('phase')[['estradiol', 'progesterone', 'testosterone']].mean()
    
    # Calculate cycle length statistics
    cycle_lengths = []
    for subject_id in hormone_df['subject_id'].unique():
        subject_data = hormone_df[hormone_df['subject_id'] == subject_id]
        cycle_length = subject_data['cycle_day'].max()
        cycle_lengths.append(cycle_length)
    
    # Calculate phase duration statistics
    phase_durations = hormone_df.groupby(['subject_id', 'phase']).size().reset_index(name='duration')
    phase_stats = phase_durations.groupby('phase')['duration'].agg(['mean', 'min', 'max'])
    
    # Create structured metrics dictionary
    metrics = {
        'phase_means': phase_means.to_dict(),
        'cycle_lengths': {
            'mean': float(np.mean(cycle_lengths)),
            'min': float(min(cycle_lengths)),
            'max': float(max(cycle_lengths))
        },
        'phase_durations': phase_stats.to_dict()
    }
    
    # Save basic metrics to JSON file
    with open('output/basic_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Define phase day ranges
    phase_ranges = {
        'perimenstruation': 'Cycle Days: 1-5 (according to Gloe)',
        'mid_follicular': 'Cycle Days: 6-11 (according to Gloe)',
        'periovulation': 'Cycle Days: 12-16 (according to Gloe)',
        'early_luteal': 'Cycle Days: 17-22 (according to Gloe)',
        'mid_late_luteal': 'Cycle Days: 23-28 (according to Gloe)'
    }
    
    # Calculate detailed metrics for each phase
    detailed_metrics = {
        'total_cycle_length': {
            'mean': float(np.mean(cycle_lengths)),
            'std': float(np.std(cycle_lengths))
        },
        'phases': {}
    }
    
    # Calculate detailed metrics for each phase
    for phase in hormone_df['phase'].unique():
        phase_data = hormone_df[hormone_df['phase'] == phase]
        
        # Initialize phase metrics
        detailed_metrics['phases'][phase] = {
            'duration': {
                'mean': float(phase_stats.loc[phase, 'mean']),
                'std': float(phase_data.groupby('subject_id').size().std())
            },
            'hormones': {
                'estradiol': {},
                'progesterone': {},
                'testosterone': {}
            }
        }
        
        # Calculate hormone statistics
        for hormone in ['estradiol', 'progesterone', 'testosterone']:
            hormone_data = phase_data[hormone]
            detailed_metrics['phases'][phase]['hormones'][hormone] = {
                'mean': float(hormone_data.mean()),
                'std': float(hormone_data.std()),
                'min': float(hormone_data.min()),
                'max': float(hormone_data.max())
            }
        
        # Add phase day range
        detailed_metrics['phases'][phase]['day_range'] = phase_ranges.get(phase, 'Unknown')
    
    # Save detailed metrics to text file
    with open('output/simulated_metrics.txt', 'w') as f:
        # Write total cycle length
        f.write("Total Cycle Length:\n")
        f.write(f"Mean ± SD: {detailed_metrics['total_cycle_length']['mean']:.1f} ± {detailed_metrics['total_cycle_length']['std']:.1f} days\n\n")
        
        # Write phase metrics
        for phase, phase_metrics in detailed_metrics['phases'].items():
            f.write(f"{phase.title()} Phase ({phase_metrics['day_range']}):\n")
            f.write(f"Duration: {phase_metrics['duration']['mean']:.1f} ± {phase_metrics['duration']['std']:.1f} days\n")
            
            for hormone, hormone_metrics in phase_metrics['hormones'].items():
                f.write(f"{hormone.title()}:\n")
                f.write(f"  Mean ± SD: {hormone_metrics['mean']:.1f} ± {hormone_metrics['std']:.1f}\n")
                f.write(f"  Range: {hormone_metrics['min']:.1f} - {hormone_metrics['max']:.1f}\n")
            f.write("\n")
    
    # Print basic metrics for immediate feedback
    print("\nMean hormone levels by phase:")
    print(phase_means)
    print("\nCycle length statistics:")
    print(f"Mean cycle length: {metrics['cycle_lengths']['mean']:.1f} days")
    print(f"Min cycle length: {metrics['cycle_lengths']['min']} days")
    print(f"Max cycle length: {metrics['cycle_lengths']['max']} days")
    print("\nPhase duration statistics (days):")
    print(phase_stats)
    
    return metrics

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate data
    hormone_df, period_df, pattern_df = simulate_hormone_and_period_data()
    
    # Create survey responses
    survey_df = pd.DataFrame({
        'subject_id': hormone_df['subject_id'].unique(),
        'date_of_last_period': [get_date_of_last_period(period_df, subject_id) for subject_id in hormone_df['subject_id'].unique()],
        'length_of_typical_cycle': [str(period_df[period_df['subject_id'] == subject_id]['cycle_day'].max()) for subject_id in hormone_df['subject_id'].unique()],
        'date_of_response': [get_date_of_response(period_df, subject_id) for subject_id in hormone_df['subject_id'].unique()]
    })
    
    # Merge in the menstrual patterns
    survey_df = survey_df.merge(pattern_df, on='subject_id')
    
    # Calculate actual cycle lengths and update survey responses
    cycle_lengths = calculate_actual_cycle_lengths(hormone_df)
    survey_df = update_survey_responses(survey_df, cycle_lengths)
    
    # Save full hormone data with labels
    hormone_df.to_csv('output/full_hormone_data_labeled.csv', index=False)
    print("Full hormone data with labels saved to output/full_hormone_data_labeled.csv")
    
    # Create unlabeled version of hormone data with 10 samples per subject, spaced 7 days apart
    hormone_df_unlabeled = []
    for subject_id in hormone_df['subject_id'].unique():
        subject_data = hormone_df[hormone_df['subject_id'] == subject_id].sort_values('date')
        # Take every 7th sample, starting from the first sample
        samples = subject_data.iloc[::7][:10]  # Take first 10 samples
        # Add sample number
        samples['sample_number'] = [f'sample_{i+1}' for i in range(len(samples))]
        hormone_df_unlabeled.append(samples)
    
    hormone_df_unlabeled = pd.concat(hormone_df_unlabeled)
    # Drop cycle_day and phase columns, keep only necessary columns
    hormone_df_unlabeled = hormone_df_unlabeled[['subject_id', 'date', 'estradiol', 'progesterone', 'testosterone', 'sample_number']]
    hormone_df_unlabeled.to_csv('output/hormone_data_unlabeled.csv', index=False)
    print("Unlabeled hormone data saved to output/hormone_data_unlabeled.csv")
    
    # Save period and sleep data
    period_df.to_csv('output/period_sleep_data.csv', index=False)
    print("Period and sleep data saved to output/period_sleep_data.csv")
    
    # Save survey responses
    survey_df.to_csv('output/survey_responses.csv', index=False)
    print("Survey responses saved to output/survey_responses.csv")
    
    # Plot hormone cycles for first 5 subjects
    for subject_id in range(5):
        plot_hormone_cycles(hormone_df, subject_id=subject_id, 
                          output_path=f'output/hormone_cycles_subject_{subject_id}.png')
    
    # Calculate and save metrics
    calculate_metrics(hormone_df)

if __name__ == "__main__":
    main() 