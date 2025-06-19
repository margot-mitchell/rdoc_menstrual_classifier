import os
import sys
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from simulation.temporal_model import TemporalModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass

class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier implementation."""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier as RF
        self.model = RF(n_estimators=n_estimators, random_state=random_state)
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Random Forest model."""
        self.feature_names = X.columns
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, path: str) -> None:
        """Save the trained model and feature names."""
        import joblib
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model and feature names."""
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']

class TemporalClassifier(BaseClassifier):
    """Temporal model classifier implementation."""
    
    def __init__(self, sequence_length: int = 70):
        from simulation.temporal_model import TemporalModel
        self.model = TemporalModel(sequence_length=sequence_length)
        self.feature_names = None
        self.output_dir = 'output/temporal'
        os.makedirs(self.output_dir, exist_ok=True)
        self.period_df = None
        self.survey_df = None
    
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Train the temporal model."""
        # For temporal model, X should be the labeled_df
        if y is not None:
            logger.warning("Temporal model ignores target variable y, using labeled data directly")
        
        # Load period and survey data
        self.period_df = pd.read_csv('output/period_sleep_data.csv')
        self.survey_df = pd.read_csv('output/survey_responses.csv')
        
        # Convert dates to datetime
        X['date'] = pd.to_datetime(X['date'])
        self.period_df['date'] = pd.to_datetime(self.period_df['date'])
        
        # Train model on full sequences with cycle structure
        logger.info("Training temporal model...")
        self.model.train(X, self.period_df, self.survey_df)
        
        # Save trained model
        self.model.save_model(os.path.join(self.output_dir, 'model.pt'))
        logger.info("Saved trained model to output/temporal/model.pt")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the temporal model."""
        # Create a DataFrame with the required format for temporal model
        predict_df = pd.DataFrame({
            'subject_id': X.index.get_level_values('subject_id') if isinstance(X.index, pd.MultiIndex) else range(len(X)),
            'date': X.index.get_level_values('date') if isinstance(X.index, pd.MultiIndex) else pd.date_range(start='2024-01-01', periods=len(X)),
            'estradiol': X['estradiol'] if 'estradiol' in X.columns else X.iloc[:, 0],
            'progesterone': X['progesterone'] if 'progesterone' in X.columns else X.iloc[:, 1],
            'testosterone': X['testosterone'] if 'testosterone' in X.columns else X.iloc[:, 2]
        })
        
        # Convert dates to datetime
        predict_df['date'] = pd.to_datetime(predict_df['date'])
        
        # Load period and survey data if not already loaded
        if self.period_df is None:
            self.period_df = pd.read_csv('output/period_sleep_data.csv')
            self.period_df['date'] = pd.to_datetime(self.period_df['date'])
        if self.survey_df is None:
            self.survey_df = pd.read_csv('output/survey_responses.csv')
        
        # Add period information to predict_df
        for subject_id in predict_df['subject_id'].unique():
            subject_periods = self.period_df[
                (self.period_df['subject_id'] == subject_id) & 
                (self.period_df['period'] == 1)
            ]
            if not subject_periods.empty:
                # Add period column to predict_df for this subject
                subject_mask = predict_df['subject_id'] == subject_id
                predict_df.loc[subject_mask, 'period'] = 0  # Default to 0
                for _, period_row in subject_periods.iterrows():
                    period_date = period_row['date']
                    predict_df.loc[
                        (subject_mask) & (predict_df['date'] == period_date),
                        'period'
                    ] = 1
        
        # Log the input data
        logger.info("\nInput data for temporal model:")
        logger.info(f"Number of samples: {len(predict_df)}")
        logger.info(f"Number of subjects: {predict_df['subject_id'].nunique()}")
        logger.info("\nHormone value ranges:")
        for hormone in ['estradiol', 'progesterone', 'testosterone']:
            logger.info(f"{hormone}: min={predict_df[hormone].min():.2f}, max={predict_df[hormone].max():.2f}, mean={predict_df[hormone].mean():.2f}")
        
        # Ensure we have the required columns
        required_columns = ['subject_id', 'date', 'estradiol', 'progesterone', 'testosterone']
        missing_columns = [col for col in required_columns if col not in predict_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by subject_id and date
        predict_df = predict_df.sort_values(['subject_id', 'date'])
        
        # Get predictions for each subject
        predictions = []
        all_cycle_positions = []
        
        for subject_id in predict_df['subject_id'].unique():
            subject_data = predict_df[predict_df['subject_id'] == subject_id].copy()
            
            # Log subject data
            logger.info(f"\nSubject {subject_id} data:")
            logger.info(f"Number of samples: {len(subject_data)}")
            logger.info(f"Date range: {subject_data['date'].min()} to {subject_data['date'].max()}")
            
            # Get predictions for the sequence
            cycle_positions = self.model.predict_cycle_position(subject_data)
            
            # Log predictions
            logger.info(f"Raw cycle position predictions: {cycle_positions}")
            
            # Handle both scalar and array predictions
            if np.isscalar(cycle_positions):
                cycle_positions = [cycle_positions]
            
            all_cycle_positions.extend(cycle_positions)
            
            # Convert cycle positions to phases
            for pos in cycle_positions:
                # Map cycle day to phase based on hormone_config.py ranges
                if 1 <= pos <= 14:  # follicular phase (days 1-14)
                    predictions.append('follicular')
                else:  # luteal phase (days 15-28)
                    predictions.append('luteal')
            
            # Plot cycle prediction
            plt.figure(figsize=(10, 6))
            # If cycle_position is a single value, repeat it for all dates
            if np.isscalar(cycle_positions) or len(cycle_positions) == 1:
                cycle_positions = np.full(len(subject_data), cycle_positions[0] if isinstance(cycle_positions, np.ndarray) else cycle_positions)
            plt.plot(subject_data['date'], cycle_positions, 'b-', label='Predicted Cycle Day')
            plt.axhline(y=14, color='r', linestyle='--', label='Phase Boundary')
            plt.title(f'Cycle Prediction for Subject {subject_id}')
            plt.xlabel('Date')
            plt.ylabel('Cycle Day')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'output/temporal/predictions/cycle_prediction_subject_{subject_id}.png')
            plt.close()
        
        # Log prediction distribution
        prediction_counts = pd.Series(predictions).value_counts()
        logger.info("\nPrediction distribution:")
        for phase, count in prediction_counts.items():
            logger.info(f"{phase}: {count} predictions ({count/len(predictions)*100:.1f}%)")
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        # Temporal model doesn't provide feature importance
        return {}
    
    def _plot_cycle_prediction(self, true_cycle_days: np.ndarray, 
                             predicted_cycle_days: np.ndarray, 
                             subject_id: int) -> None:
        """Plot cycle prediction for a subject."""
        plt.figure(figsize=(10, 6))
        plt.plot(true_cycle_days, label='True Cycle Day', marker='o')
        plt.plot(predicted_cycle_days, label='Predicted Cycle Day', marker='x')
        plt.title(f'Cycle Day Prediction - Subject {subject_id}')
        plt.xlabel('Sample Number')
        plt.ylabel('Cycle Day')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'predictions/cycle_prediction_subject_{subject_id}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Cycle prediction plot saved to {output_path}")

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    
    def prepare_features(self, hormone_df: pd.DataFrame, period_df: pd.DataFrame, 
                        survey_df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features by combining hormone, period, and survey data."""
        from sklearn.preprocessing import StandardScaler
        
        # Create a mapping of subject_id to their survey response
        survey_dict = survey_df.set_index('subject_id').to_dict('index')
        
        # Convert dates to datetime at the start
        hormone_df = hormone_df.copy()
        hormone_df['date'] = pd.to_datetime(hormone_df['date'])
        period_df = period_df.copy()
        period_df['date'] = pd.to_datetime(period_df['date'])
        
        # Create expanded data
        expanded_data = []
        for _, row in hormone_df.iterrows():
            subject_id = row['subject_id']
            combined = {k: v for k, v in row.to_dict().items() 
                       if k not in ['date', 'subject_id']}
            
            # Add period data - only use non-phase related features
            subject_period = period_df[period_df['subject_id'] == subject_id].copy()
            if not subject_period.empty:
                hormone_date = row['date']
                # Calculate time differences in days
                time_diffs = (subject_period['date'] - hormone_date).dt.total_seconds() / (24 * 3600)
                closest_idx = time_diffs.abs().idxmin()
                period_row = subject_period.loc[closest_idx]
                # Exclude any features that might leak phase information
                excluded_cols = ['cycle_day', 'subject_id', 'date', 'phase', 'sleep_hours', 
                               'period', 'cycle_position']
                period_data = {k: v for k, v in period_row.to_dict().items() 
                             if k not in excluded_cols}
                combined.update(period_data)
            
            # Add survey data - only use non-phase related features
            if subject_id in survey_dict:
                survey_data = survey_dict[subject_id].copy()
                # Exclude any features that might leak phase information
                excluded_cols = ['subject_id', 'date_of_response', 'date_of_last_period', 
                               'cycle_length', 'menstrual_pattern']
                survey_data = {k: v for k, v in survey_data.items() 
                             if k not in excluded_cols}
                combined.update(survey_data)
            
            expanded_data.append(combined)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(expanded_data)
        
        # Remove target-related columns
        columns_to_exclude = ['target_phase', 'phase', 'cycle_day', 'cycle_position']
        features_df = features_df.drop(columns=[col for col in columns_to_exclude 
                                              if col in features_df.columns])
        
        # Convert categorical variables
        if 'menstrual_pattern' in features_df.columns:
            features_df['menstrual_pattern'] = features_df['menstrual_pattern'].astype('category')
        
        # One-hot encoding
        X = pd.get_dummies(features_df, drop_first=True)
        
        # Store feature columns during training
        if is_training:
            self.feature_columns = X.columns.tolist()
        else:
            # Ensure same columns as training
            if self.feature_columns is None:
                raise ValueError("Model must be trained before making predictions")
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                X_new = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
                for col in X.columns:
                    if col in self.feature_columns:
                        X_new[col] = X[col]
                X = X_new
        
        # Scale features
        if is_training:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Get target variable
        y = hormone_df['phase'] if 'phase' in hormone_df.columns else None
        
        return pd.DataFrame(X_scaled, columns=X.columns), y

class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                model_name: str, X: pd.DataFrame = None, 
                subject_ids: np.ndarray = None, dates: np.ndarray = None) -> Dict[str, float]:
        """Evaluate model performance."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Print classification report
        logger.info(f"\nClassification Report for {model_name}:")
        logger.info(classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, model_name)
        
        # Save misclassified samples if we have the data
        if X is not None and subject_ids is not None and dates is not None:
            self._save_misclassified_samples(
                X, y_true, y_pred, subject_ids, dates, model_name
            )
        
        return metrics
    
    def _save_misclassified_samples(self, X: pd.DataFrame, y_true: np.ndarray, 
                                  y_pred: np.ndarray, subject_ids: np.ndarray,
                                  dates: np.ndarray, model_name: str) -> None:
        """Save misclassified samples to a CSV file."""
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        
        if not any(misclassified_mask):
            logger.info(f"No misclassified samples found for {model_name}")
            return
        
        # Create DataFrame with misclassified samples
        misclassified_df = pd.DataFrame({
            'subject_id': subject_ids[misclassified_mask],
            'date': dates[misclassified_mask],
            'true_phase': y_true[misclassified_mask],
            'predicted_phase': y_pred[misclassified_mask]
        })
        
        # Add feature values
        for col in X.columns:
            misclassified_df[col] = X[col].values[misclassified_mask]
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, 'misclassified_samples.csv')
        misclassified_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(misclassified_df)} misclassified samples to {output_path}")
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              model_name: str) -> None:
        """Plot feature importance."""
        if not feature_importance:
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        features, importances = zip(*sorted_features[:20])
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Most Important Features - {model_name}')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {output_path}")

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Follicular', 'Luteal'],
                   yticklabels=['Follicular', 'Luteal'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {output_path}")

def run_classification():
    """Run classification using both Random Forest and Temporal models."""
    logger.info("Running Random Forest Classifier...")
    
    # Load data
    logger.info("Successfully loaded all data files:")
    unlabeled_hormone_data = pd.read_csv('output/hormone_data_unlabeled.csv')
    period_sleep_response = pd.read_csv('output/period_sleep_data.csv')
    survey_responses = pd.read_csv('output/survey_responses.csv')
    labeled_hormone_data = pd.read_csv('output/full_hormone_data_labeled.csv')
    
    logger.info(f"- Unlabeled hormone data: {len(unlabeled_hormone_data)} rows")
    logger.info(f"- Period/sleep data: {len(period_sleep_response)} rows")
    logger.info(f"- Survey responses: {len(survey_responses)} rows")
    logger.info(f"- Labeled hormone data: {len(labeled_hormone_data)} rows")
    
    # Convert date columns to datetime
    for df in [unlabeled_hormone_data, period_sleep_response, labeled_hormone_data]:
        df['date'] = pd.to_datetime(df['date'])
    
    # Remove samples without target phases
    valid_mask = labeled_hormone_data['phase'].notna()
    labeled_hormone_data = labeled_hormone_data[valid_mask]
    
    # Get unique subjects
    unique_subjects = labeled_hormone_data['subject_id'].unique()
    
    # Split subjects into train and test sets
    train_subjects, test_subjects = train_test_split(
        unique_subjects, 
        test_size=0.2, 
        random_state=42
    )
    
    # Split data based on subjects
    train_data = labeled_hormone_data[labeled_hormone_data['subject_id'].isin(train_subjects)]
    test_data = labeled_hormone_data[labeled_hormone_data['subject_id'].isin(test_subjects)]
    
    # Prepare features for training and testing
    data_processor = DataProcessor()
    X_train, y_train = data_processor.prepare_features(
        train_data, 
        period_sleep_response, 
        survey_responses,
        is_training=True
    )
    X_test, y_test = data_processor.prepare_features(
        test_data, 
        period_sleep_response, 
        survey_responses,
        is_training=False
    )
    
    # Train and evaluate Random Forest
    rf_classifier = RandomForestClassifier()
    rf_classifier.train(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    
    # Print classification report
    logger.info("\nClassification Report for RandomForest:")
    logger.info(classification_report(y_test, rf_predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('output/randomforest/confusion_matrix_RandomForest.png')
    plt.close()
    
    # Save misclassified samples
    misclassified = test_data[rf_predictions != y_test].copy()
    misclassified['predicted_phase'] = rf_predictions[rf_predictions != y_test]
    misclassified.to_csv('output/randomforest/misclassified_samples.csv', index=False)
    logger.info(f"Saved {len(misclassified)} misclassified samples to output/randomforest/misclassified_samples.csv")
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, rf_predictions)
    logger.info("\nModel metrics for RandomForest:")
    logger.info(f"accuracy: {accuracy:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importances = rf_classifier.model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance - Random Forest')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [rf_classifier.feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/randomforest/feature_importance_RandomForest.png')
    plt.close()
    
    # Save predictions
    predictions_df = test_data.copy()
    predictions_df['predicted_phase'] = rf_predictions
    predictions_df.to_csv('output/randomforest/predictions.csv', index=False)
    logger.info("Saved predictions to output/randomforest/predictions.csv")
    
    # Save model
    rf_classifier.save_model('output/randomforest/model.joblib')
    logger.info("Saved model to output/randomforest/model.joblib")
    
    # Temporal Model
    logger.info("\nRunning Temporal Model Classifier...")
    
    # Load data again for temporal model
    logger.info("Successfully loaded all data files:")
    logger.info(f"- Unlabeled hormone data: {len(unlabeled_hormone_data)} rows")
    logger.info(f"- Period/sleep data: {len(period_sleep_response)} rows")
    logger.info(f"- Survey responses: {len(survey_responses)} rows")
    logger.info(f"- Labeled hormone data: {len(labeled_hormone_data)} rows")
    
    # Train temporal model on labeled data
    logger.info("Training temporal model...")
    temporal_model = TemporalModel()
    temporal_model.train(labeled_hormone_data, period_sleep_response, survey_responses)
    
    # Use unlabeled data for predictions (10 samples per subject)
    predict_hormone_data = unlabeled_hormone_data.copy()
    
    # Print input data info
    logger.info("\nInput data for temporal model predictions:")
    logger.info(f"Number of samples: {len(predict_hormone_data)}")
    logger.info(f"Number of subjects: {len(predict_hormone_data['subject_id'].unique())}")
    
    # Print hormone value ranges
    logger.info("\nHormone value ranges:")
    for hormone in ['estradiol', 'progesterone', 'testosterone']:
        values = predict_hormone_data[hormone].values
        logger.info(f"{hormone}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
    
    # Make predictions for each subject
    all_predictions = []
    all_true_phases = []
    all_subject_ids = []
    all_dates = []
    
    for subject_id in predict_hormone_data['subject_id'].unique():
        subject_data = predict_hormone_data[predict_hormone_data['subject_id'] == subject_id]
        
        # Predict cycle position
        cycle_position = temporal_model.predict_cycle_position(subject_data)
        all_predictions.extend(cycle_position)
        
        # Get corresponding true phases from labeled data
        for _, row in subject_data.iterrows():
            # Find matching sample in labeled data
            matching = labeled_hormone_data[
                (labeled_hormone_data['subject_id'] == row['subject_id']) & 
                (labeled_hormone_data['date'] == row['date'])
            ]
            
            if not matching.empty:
                true_phase = matching.iloc[0]['phase']
                all_true_phases.append(true_phase)
                all_subject_ids.append(row['subject_id'])
                all_dates.append(row['date'])
            else:
                logger.warning(f"No matching labeled sample found for subject {row['subject_id']} on {row['date']}")
    
    # Print classification report
    logger.info("\nClassification Report for Temporal:")
    logger.info(classification_report(all_true_phases, all_predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_true_phases, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Temporal Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('output/temporal/confusion_matrix_Temporal.png')
    plt.close()
    
    # Save misclassified samples
    misclassified_mask = np.array(all_predictions) != np.array(all_true_phases)
    if np.any(misclassified_mask):
        misclassified_df = pd.DataFrame({
            'subject_id': np.array(all_subject_ids)[misclassified_mask],
            'date': np.array(all_dates)[misclassified_mask],
            'true_phase': np.array(all_true_phases)[misclassified_mask],
            'predicted_phase': np.array(all_predictions)[misclassified_mask]
        })
        misclassified_df.to_csv('output/temporal/misclassified_samples.csv', index=False)
        logger.info(f"Saved {len(misclassified_df)} misclassified samples to output/temporal/misclassified_samples.csv")
    
    # Calculate and print metrics
    accuracy = accuracy_score(all_true_phases, all_predictions)
    logger.info("\nModel metrics for Temporal:")
    logger.info(f"accuracy: {accuracy:.3f}")
    
    # Show prediction distribution
    prediction_counts = pd.Series(all_predictions).value_counts()
    logger.info("\nPrediction distribution for unlabeled samples:")
    for phase, count in prediction_counts.items():
        logger.info(f"{phase}: {count} predictions ({count/len(all_predictions)*100:.1f}%)")
    
    # Save predictions
    predictions_df = predict_hormone_data.copy()
    predictions_df['predicted_phase'] = all_predictions
    predictions_df.to_csv('output/temporal/predictions.csv', index=False)
    logger.info("Saved predictions to output/temporal/predictions.csv")

def main():
    """Main function to run classification with different models."""
    run_classification()

if __name__ == '__main__':
    main() 