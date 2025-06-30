"""
Model evaluator utility module for model evaluation and metrics calculation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_survey_accuracy(survey_file: str, period_file: str, output_dir: str) -> pd.DataFrame:
    """
    Check the accuracy of date_of_last_period and cycle_length from survey data.
    
    Args:
        survey_file (str): Path to survey responses CSV file
        period_file (str): Path to period data CSV file
        output_dir (str): Output directory for saving results
        
    Returns:
        pd.DataFrame: DataFrame with accuracy analysis results
    """
    # Load data
    survey_df = pd.read_csv(survey_file)
    period_df = pd.read_csv(period_file)
    
    # Convert dates to datetime
    survey_df['date_of_last_period'] = pd.to_datetime(survey_df['date_of_last_period'])
    survey_df['date_of_response'] = pd.to_datetime(survey_df['date_of_response'])
    period_df['date'] = pd.to_datetime(period_df['date'])
    
    logger.info(f"Loaded {len(survey_df)} survey responses")
    logger.info(f"Loaded {len(period_df)} period records")
    
    # Get the date range of period data
    min_period_date = period_df['date'].min()
    max_period_date = period_df['date'].max()
    logger.info(f"Period data date range: {min_period_date} to {max_period_date}")
    
    # Analyze each subject
    accuracy_results = []
    
    for _, survey_row in survey_df.iterrows():
        subject_id = survey_row['subject_id']
        reported_last_period = survey_row['date_of_last_period']
        reported_cycle_length = survey_row['cycle_length']
        response_date = survey_row['date_of_response']
        
        # Get actual period data for this subject
        subject_periods = period_df[period_df['subject_id'] == subject_id].copy()
        subject_periods = subject_periods.sort_values('date')
        
        # Find actual periods (where period == 'Yes')
        actual_periods = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
        
        if not actual_periods:
            logger.warning(f"Subject {subject_id}: No actual periods found in period data")
            continue
        
        # Check if reported last period is within THIS SUBJECT'S period data range
        subject_min_date = subject_periods['date'].min()
        subject_max_date = subject_periods['date'].max()
        reported_in_range = subject_min_date <= reported_last_period <= subject_max_date
        
        # Check if reported last period matches any actual period (only if in range)
        period_match = False
        closest_period = None
        days_diff = None
        
        if reported_in_range:
            period_match = reported_last_period in actual_periods
            
            # Find the closest actual period to the reported date
            if actual_periods:
                closest_period = min(actual_periods, key=lambda x: abs((x - reported_last_period).days))
                days_diff = abs((closest_period - reported_last_period).days)
        
        # Calculate actual cycle length from period data
        if len(actual_periods) >= 2:
            # Find consecutive period starts (first day of each period)
            # Group consecutive period days and take the first day of each period
            period_starts = []
            current_period_start = None
            
            for period_date in actual_periods:
                if current_period_start is None:
                    current_period_start = period_date
                elif (period_date - current_period_start).days > 7:  # Gap of more than 7 days indicates new period
                    period_starts.append(current_period_start)
                    current_period_start = period_date
            
            # Add the last period start
            if current_period_start is not None:
                period_starts.append(current_period_start)
            
            # Calculate intervals between consecutive period starts
            if len(period_starts) >= 2:
                intervals = []
                for i in range(1, len(period_starts)):
                    interval = (period_starts[i] - period_starts[i-1]).days
                    intervals.append(interval)
                
                actual_cycle_length = np.mean(intervals)
                cycle_length_diff = abs(reported_cycle_length - actual_cycle_length)
            else:
                actual_cycle_length = None
                cycle_length_diff = None
        else:
            actual_cycle_length = None
            cycle_length_diff = None
        
        # Store results
        result = {
            'subject_id': subject_id,
            'reported_last_period': reported_last_period,
            'reported_cycle_length': reported_cycle_length,
            'response_date': response_date,
            'reported_in_range': reported_in_range,
            'period_match': period_match,
            'closest_period': closest_period,
            'days_diff': days_diff,
            'actual_cycle_length': actual_cycle_length,
            'cycle_length_diff': cycle_length_diff,
            'num_actual_periods': len(actual_periods)
        }
        accuracy_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(accuracy_results)
    
    # Print summary statistics
    logger.info("\n=== SURVEY DATA ACCURACY ANALYSIS ===")
    
    # Count how many reported dates are outside the range
    out_of_range = results_df['reported_in_range'].sum()
    total_subjects = len(results_df)
    logger.info(f"Reported dates within period data range: {out_of_range}/{total_subjects} ({out_of_range/total_subjects*100:.1f}%)")
    
    # Period date accuracy (only for dates within range)
    in_range_results = results_df[results_df['reported_in_range']]
    if len(in_range_results) > 0:
        period_matches = in_range_results['period_match'].sum()
        logger.info(f"Reported dates that match actual period days: {period_matches}/{len(in_range_results)} ({period_matches/len(in_range_results)*100:.1f}%)")
        
        # Days difference statistics (only for dates within range)
        valid_days_diff = in_range_results['days_diff'].dropna()
        if len(valid_days_diff) > 0:
            logger.info(f"Average days difference from closest period: {valid_days_diff.mean():.1f} days")
            logger.info(f"Median days difference: {valid_days_diff.median():.1f} days")
            logger.info(f"Max days difference: {valid_days_diff.max():.0f} days")
    
    # Cycle length accuracy (all subjects)
    valid_cycle_diff = results_df['cycle_length_diff'].dropna()
    if len(valid_cycle_diff) > 0:
        logger.info(f"Average cycle length difference: {valid_cycle_diff.mean():.1f} days")
        logger.info(f"Median cycle length difference: {valid_cycle_diff.median():.1f} days")
        logger.info(f"Max cycle length difference: {valid_cycle_diff.max():.0f} days")
    
    # Show problematic cases
    logger.info("\n=== PROBLEMATIC CASES ===")
    
    # Cases where reported period doesn't match any actual period (only within range)
    if len(in_range_results) > 0:
        mismatched = in_range_results[~in_range_results['period_match']].copy()
        # Only show cases with large differences (>5 days) as these are truly problematic
        large_mismatches = mismatched[mismatched['days_diff'] > 5].copy()
        # Exclude cases where reported_last_period is exactly cycle_length days before the first actual period
        filtered_mismatches = []
        for _, row in large_mismatches.iterrows():
            subject_id = row['subject_id']
            reported_last_period = row['reported_last_period']
            cycle_length = row['reported_cycle_length']
            # Get actual period data for this subject
            subject_periods = period_df[period_df['subject_id'] == subject_id].copy()
            subject_periods = subject_periods.sort_values('date')
            actual_periods = subject_periods[subject_periods['period'] == 'Yes']['date'].tolist()
            if actual_periods:
                first_period = min(actual_periods)
                synthetic_date = first_period - pd.Timedelta(days=cycle_length)
                # If reported_last_period is not the synthetic date, keep as problematic
                if not pd.Timestamp(reported_last_period) == pd.Timestamp(synthetic_date):
                    filtered_mismatches.append(row)
            else:
                filtered_mismatches.append(row)
        if len(filtered_mismatches) > 0:
            logger.info(f"Subjects with large period date mismatches (within range, not synthetic) ({len(filtered_mismatches)}):")
            for row in filtered_mismatches[:10]:
                logger.info(f"  Subject {row['subject_id']}: reported {row['reported_last_period']}, closest actual {row['closest_period']} (diff: {row['days_diff']} days)")
        else:
            logger.info("No subjects with large period date mismatches found (excluding synthetic cases).")
    
    # Cases with large cycle length differences
    large_cycle_diff = results_df[results_df['cycle_length_diff'] > 5]
    if len(large_cycle_diff) > 0:
        logger.info(f"\nSubjects with large cycle length differences ({len(large_cycle_diff)}):")
        for _, row in large_cycle_diff.head(10).iterrows():
            logger.info(f"  Subject {row['subject_id']}: reported {row['reported_cycle_length']}, actual {row['actual_cycle_length']:.1f} (diff: {row['cycle_length_diff']:.1f} days)")
    else:
        logger.info("No subjects with large cycle length differences found.")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'survey_accuracy_analysis.csv'), index=False)
    logger.info(f"\nDetailed results saved to {output_dir}/survey_accuracy_analysis.csv")
    
    return results_df


class ModelEvaluator:
    """Model evaluator for calculating metrics and generating evaluation plots."""
    
    def __init__(self, output_dir: str):
        """
        Initialize model evaluator.
        
        Args:
            output_dir (str): Output directory for saving results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_model_dir(self, model_name: str) -> str:
        """
        Get model-specific subdirectory path.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Path to model-specific subdirectory
        """
        # Clean model name for directory name
        clean_name = model_name.lower().replace(" ", "_").replace("-", "_")
        model_dir = os.path.join(self.output_dir, clean_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                model_name: str, X: Optional[pd.DataFrame] = None,
                subject_ids: Optional[np.ndarray] = None,
                dates: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            X (pd.DataFrame): Feature matrix (optional)
            subject_ids (np.ndarray): Subject IDs (optional)
            dates (np.ndarray): Dates (optional)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Calculate ROC AUC if binary classification
        if len(np.unique(y_true)) == 2:
            try:
                # For binary classification, we need probability predictions
                # This is a simplified version - you may need to adjust based on your model
                metrics['roc_auc'] = 0.5  # Placeholder - implement based on your model's predict_proba
            except:
                metrics['roc_auc'] = np.nan
        else:
            metrics['roc_auc'] = np.nan
        
        # Get model-specific directory
        model_dir = self._get_model_dir(model_name)
        
        # Save detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_path = os.path.join(model_dir, 'classification_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n")
            f.write("=" * 50 + "\n")
            f.write(classification_report(y_true, y_pred))
        
        # Save misclassified samples if feature matrix is provided
        if X is not None and subject_ids is not None:
            self._save_misclassified_samples(X, y_true, y_pred, subject_ids, dates, model_name)
        
        return metrics
    
    def _save_misclassified_samples(self, X: pd.DataFrame, y_true: np.ndarray, 
                                  y_pred: np.ndarray, subject_ids: np.ndarray,
                                  dates: Optional[np.ndarray], model_name: str) -> None:
        """Save misclassified samples for analysis."""
        misclassified_mask = y_true != y_pred
        
        if misclassified_mask.sum() > 0:
            misclassified_data = {
                'subject_id': subject_ids[misclassified_mask],
                'true_label': y_true[misclassified_mask],
                'predicted_label': y_pred[misclassified_mask]
            }
            
            if dates is not None:
                misclassified_data['date'] = dates[misclassified_mask]
            
            # Add feature values
            for col in X.columns:
                misclassified_data[col] = X.iloc[misclassified_mask][col].values
            
            misclassified_df = pd.DataFrame(misclassified_data)
            model_dir = self._get_model_dir(model_name)
            output_path = os.path.join(model_dir, 'misclassified_samples.csv')
            misclassified_df.to_csv(output_path, index=False)
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], model_name: str) -> None:
        """Plot feature importance."""
        if not feature_importance:
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance for {model_name}')
        plt.gca().invert_yaxis()
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        output_path = os.path.join(model_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), 
                   yticklabels=np.unique(y_true))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model_name}')
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        output_path = os.path.join(model_dir, 'confusion_matrix_test_set.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> None:
        """Plot ROC curve."""
        if len(np.unique(y_true)) != 2:
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        output_path = os.path.join(model_dir, 'roc_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> None:
        """Plot precision-recall curve."""
        if len(np.unique(y_true)) != 2:
            return
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        output_path = os.path.join(model_dir, 'precision_recall_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_evaluation_results(self, results: Dict[str, Dict[str, float]], output_path: str) -> None:
        """Save evaluation results to CSV."""
        results_df = pd.DataFrame(results).T
        results_df.to_csv(output_path)
    
    def generate_evaluation_report(self, results: Dict[str, Dict[str, float]], output_path: str) -> None:
        """Generate a comprehensive evaluation report."""
        with open(output_path, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
                f.write("\n")
            
            # Find best model
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            f.write(f"Best Model (by accuracy): {best_model}\n")
            f.write(f"Best Accuracy: {results[best_model]['accuracy']:.4f}\n") 