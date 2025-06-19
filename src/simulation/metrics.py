import numpy as np
import pandas as pd
import json

def calculate_metrics(hormone_df):
    """
    Calculate and save metrics about the hormone data.
    
    Args:
        hormone_df (pd.DataFrame): DataFrame containing hormone data
        
    Returns:
        str: Formatted metrics text
    """
    # Calculate mean hormone levels by phase
    phase_means = hormone_df.groupby('phase')[['estradiol', 'progesterone', 'testosterone']].mean()
    
    # Calculate cycle length statistics
    cycle_lengths = []
    for subject_id in hormone_df['subject_id'].unique():
        subject_data = hormone_df[hormone_df['subject_id'] == subject_id]
        cycle_length = subject_data['cycle_day'].max()
        cycle_lengths.append(cycle_length)
    
    # Calculate phase duration statistics by counting continuous occurrences
    phase_durations = []
    for subject_id in hormone_df['subject_id'].unique():
        subject_data = hormone_df[hormone_df['subject_id'] == subject_id].sort_values('date')
        
        # Group by phase and count continuous occurrences
        current_phase = None
        current_duration = 0
        for _, row in subject_data.iterrows():
            if row['phase'] == current_phase:
                current_duration += 1
            else:
                if current_phase is not None:
                    phase_durations.append({
                        'subject_id': subject_id,
                        'phase': current_phase,
                        'duration': current_duration
                    })
                current_phase = row['phase']
                current_duration = 1
        
        # Add the last phase duration
        if current_phase is not None:
            phase_durations.append({
                'subject_id': subject_id,
                'phase': current_phase,
                'duration': current_duration
            })
    
    # Convert to DataFrame and calculate statistics
    phase_durations_df = pd.DataFrame(phase_durations)
    phase_stats = phase_durations_df.groupby('phase')['duration'].agg(['mean', 'min', 'max'])
    
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
                'std': float(phase_durations_df[phase_durations_df['phase'] == phase]['duration'].std())
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
    
    # Format metrics text
    metrics_text = []
    
    # Write total cycle length
    metrics_text.append("Total Cycle Length:")
    metrics_text.append(f"Mean ± SD: {detailed_metrics['total_cycle_length']['mean']:.1f} ± {detailed_metrics['total_cycle_length']['std']:.1f} days\n")
    
    # Write phase metrics
    for phase, phase_metrics in detailed_metrics['phases'].items():
        metrics_text.append(f"{phase.title()} Phase ({phase_metrics['day_range']}):")
        metrics_text.append(f"Duration: {phase_metrics['duration']['mean']:.1f} ± {phase_metrics['duration']['std']:.1f} days")
        
        for hormone, hormone_metrics in phase_metrics['hormones'].items():
            metrics_text.append(f"{hormone.title()}:")
            metrics_text.append(f"  Mean ± SD: {hormone_metrics['mean']:.1f} ± {hormone_metrics['std']:.1f}")
            metrics_text.append(f"  Range: {hormone_metrics['min']:.1f} - {hormone_metrics['max']:.1f}")
        metrics_text.append("")
    
    # Print basic metrics for immediate feedback
    print("\nMean hormone levels by phase:")
    print(phase_means)
    print("\nCycle length statistics:")
    print(f"Mean cycle length: {metrics['cycle_lengths']['mean']:.1f} days")
    print(f"Min cycle length: {metrics['cycle_lengths']['min']} days")
    print(f"Max cycle length: {metrics['cycle_lengths']['max']} days")
    print("\nPhase duration statistics (days):")
    print(phase_stats)
    
    return "\n".join(metrics_text) 