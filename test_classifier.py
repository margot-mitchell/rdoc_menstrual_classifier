import pandas as pd
import numpy as np
from classify import MenstrualClassifier, create_target_variable, load_data
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, output_path='output/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Follicular', 'Luteal'])
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix: Phase Classification')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")

def plot_feature_importance(classifier, feature_names, output_path='output/feature_importance.png'):
    """Plot and save feature importance."""
    # Get feature importance
    importance = classifier.model.feature_importances_
    
    # Create DataFrame for easier plotting
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Feature importance plot saved to {output_path}")

def test_classifier():
    """Evaluate classifier predictions against labeled data."""
    try:
        # Load the data
        logger.info("Loading data...")
        hormone_df = pd.read_csv('output/hormone_data_with_predictions.csv')
        labeled_df = pd.read_csv('output/full_hormone_data_labeled.csv')
        
        if hormone_df is None or labeled_df is None:
            raise ValueError("Failed to load data files")
        
        logger.info(f"Predicted data shape: {hormone_df.shape}")
        logger.info(f"Labeled data shape: {labeled_df.shape}")
        
        # Get true phases from labeled data
        logger.info("Getting true phases from labeled data...")
        true_phases = []
        predicted_phases = []
        valid_indices = []
        
        for idx, row in hormone_df.iterrows():
            # Find matching date in labeled data
            matching = labeled_df[
                (labeled_df['subject_id'] == row['subject_id']) & 
                (labeled_df['date'] == row['date'])
            ]
            if not matching.empty:
                phase = matching.iloc[0]['phase']
                # Convert subphase to main phase
                if phase in ['early_luteal', 'mid_late_luteal']:
                    true_phases.append('luteal')
                else:  # periovulation, mid_follicular, perimenstruation
                    true_phases.append('follicular')
                # Get the predicted phase
                predicted_phases.append(row['phase'])
                valid_indices.append(idx)
            else:
                logger.warning(f"No matching phase found for subject {row['subject_id']} on {row['date']}")
        
        # Calculate accuracy
        accuracy = accuracy_score(true_phases, predicted_phases)
        logger.info(f"Accuracy: {accuracy:.2f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(true_phases, predicted_phases)
        
        # Save misclassified samples
        misclassified_mask = np.array(true_phases) != np.array(predicted_phases)
        if np.any(misclassified_mask):
            misclassified_df = hormone_df.iloc[valid_indices][misclassified_mask].copy()
            misclassified_df['true_phase'] = np.array(true_phases)[misclassified_mask]
            misclassified_df['predicted_phase'] = np.array(predicted_phases)[misclassified_mask]
            misclassified_df.to_csv('output/misclassified_samples.csv', index=False)
            logger.info(f"Saved {len(misclassified_df)} misclassified samples to output/misclassified_samples.csv")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Error in test_classifier: {str(e)}")
        raise

if __name__ == "__main__":
    test_classifier() 