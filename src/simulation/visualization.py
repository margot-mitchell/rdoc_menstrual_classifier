import matplotlib.pyplot as plt
import numpy as np

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