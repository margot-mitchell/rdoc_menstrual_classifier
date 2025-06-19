import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from src.simulation.temporal_model import TemporalModel
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load all required data files."""
    try:
        hormone_df = pd.read_csv('output/hormone_data_unlabeled.csv')
        period_df = pd.read_csv('output/period_sleep_data.csv')
        survey_df = pd.read_csv('output/survey_responses.csv')
        labeled_df = pd.read_csv('output/full_hormone_data_labeled.csv')
        
        logger.info("Successfully loaded all data files:")
        logger.info(f"- Unlabeled hormone data: {len(hormone_df)} rows")
        logger.info(f"- Period/sleep data: {len(period_df)} rows")
        logger.info(f"- Survey responses: {len(survey_df)} rows")
        logger.info(f"- Labeled hormone data: {len(labeled_df)} rows")
        
        return hormone_df, period_df, survey_df, labeled_df
    
    except FileNotFoundError as e:
        logger.error(f"Error: Could not find one or more data files: {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None

def get_phase_from_cycle_day(cycle_day):
    """Convert cycle day to phase."""
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

def plot_confusion_matrix(y_true, y_pred, output_path='output/temporal/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Follicular', 'Luteal'])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix: Phase Classification (Temporal Model)')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")

def plot_cycle_prediction(actual_cycle, predicted_cycle, subject_id, output_path):
    """Plot actual vs predicted cycle days."""
    plt.figure(figsize=(12, 6))
    
    # Handle both scalar and array cases for predicted_cycle
    if np.isscalar(predicted_cycle):
        predicted_cycle = [predicted_cycle]
    
    # Plot actual cycle days
    plt.plot(actual_cycle, label='Actual Cycle Day', marker='o')
    # Plot predicted cycle days
    plt.plot(predicted_cycle, label='Predicted Cycle Day', marker='x', linestyle='--')
    
    plt.title(f'Cycle Day Prediction - Subject {subject_id}')
    plt.xlabel('Sample Number')
    plt.ylabel('Cycle Day')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    # Load data
    hormone_df, period_df, survey_df, labeled_df = load_data()
    if hormone_df is None or period_df is None or survey_df is None or labeled_df is None:
        return
    
    # Create output directories
    os.makedirs('output/temporal/predictions', exist_ok=True)
    
    # Initialize temporal model
    model = TemporalModel()
    
    # Train model on full sequences
    logger.info("Training temporal model...")
    model.train(labeled_df, period_df, survey_df)
    logger.info("Temporal model training complete")
    
    # Make predictions for each subject
    all_predictions = []
    all_true_phases = []
    all_subject_ids = []
    all_dates = []
    all_detailed_phases = []  # Store detailed phase predictions
    
    for subject_id in hormone_df['subject_id'].unique():
        # Get subject's unlabeled samples
        subject_samples = hormone_df[hormone_df['subject_id'] == subject_id].sort_values('date')
        
        # Get corresponding labeled data for evaluation
        subject_labeled = labeled_df[labeled_df['subject_id'] == subject_id].sort_values('date')
        
        # Create a mapping of dates to phases for this subject
        date_to_phase = dict(zip(subject_labeled['date'], subject_labeled['phase']))
        
        # Predict cycle positions
        predicted_cycle = model.predict_cycle_position(subject_samples)
        
        # Log raw cycle day predictions for debugging
        logger.info(f"Subject {subject_id} - Raw cycle day predictions: {predicted_cycle[0]}")
        
        # Convert cycle days to phases
        # Handle both scalar and array cases
        pred = predicted_cycle[0]
        if np.isscalar(pred):
            predicted_phases = [get_phase_from_cycle_day(pred)]
        else:
            predicted_phases = [get_phase_from_cycle_day(day) for day in pred]
        
        # Get true phases for the same dates
        true_phases = []
        valid_indices = []
        
        for idx, row in subject_samples.iterrows():
            if row['date'] in date_to_phase:
                true_phases.append(date_to_phase[row['date']])
                valid_indices.append(idx)
        
        # Only keep predictions where we have true values
        if valid_indices:
            predicted_phases = [predicted_phases[i] if i < len(predicted_phases) else predicted_phases[-1] 
                              for i in range(len(valid_indices))]
            
            # Convert to follicular/luteal
            predicted_main_phases = ['luteal' if p in ['early_luteal', 'mid_late_luteal'] else 'follicular' 
                                   for p in predicted_phases]
            true_main_phases = ['luteal' if p in ['early_luteal', 'mid_late_luteal'] else 'follicular' 
                               for p in true_phases]
            
            # Store predictions and true values
            all_predictions.extend(predicted_main_phases)
            all_true_phases.extend(true_main_phases)
            all_subject_ids.extend([subject_id] * len(predicted_main_phases))
            all_dates.extend(subject_samples.loc[valid_indices, 'date'].tolist())
            all_detailed_phases.extend(predicted_phases)
        
        # Plot cycle prediction for first 5 subjects
        if subject_id < 5 and valid_indices:
            plot_cycle_prediction(
                subject_labeled['cycle_day'].values[:len(predicted_cycle[0]) if not np.isscalar(predicted_cycle[0]) else 1],
                predicted_cycle[0],
                subject_id,
                f'output/temporal/predictions/cycle_prediction_subject_{subject_id}.png'
            )
    
    # Log phase distribution
    phase_counts = Counter(all_detailed_phases)
    logger.info("Detailed phase distribution:")
    for phase, count in phase_counts.items():
        logger.info(f"- {phase}: {count} predictions")
    
    # Log main phase distribution
    main_phase_counts = Counter(all_predictions)
    logger.info("Main phase distribution:")
    for phase, count in main_phase_counts.items():
        logger.info(f"- {phase}: {count} predictions")
    
    # Calculate and print accuracy
    accuracy = accuracy_score(all_true_phases, all_predictions)
    logger.info(f"Overall accuracy: {accuracy:.3f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_true_phases, all_predictions)
    
    # Save predictions
    results_df = pd.DataFrame({
        'subject_id': all_subject_ids,
        'date': all_dates,
        'true_phase': all_true_phases,
        'predicted_phase': all_predictions
    })
    results_df.to_csv('output/temporal/predictions.csv', index=False)
    logger.info("Saved predictions to output/temporal/predictions.csv")

if __name__ == "__main__":
    main() 