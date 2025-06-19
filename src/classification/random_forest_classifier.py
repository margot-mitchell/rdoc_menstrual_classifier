import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the phases
PHASES = ['follicular', 'luteal']

def load_data():
    """
    Load all required data files including the labeled data for training.
    
    Returns:
        tuple: (hormone_df, period_df, survey_df, labeled_df) DataFrames containing the loaded data
    """
    try:
        hormone_df = pd.read_csv('output/hormone_data_unlabeled.csv')
        period_df = pd.read_csv('output/period_sleep_data.csv')
        survey_df = pd.read_csv('output/survey_responses.csv')
        labeled_df = pd.read_csv('output/full_hormone_data_labeled.csv')
        
        print("Successfully loaded all data files:")
        print(f"- Unlabeled hormone data: {len(hormone_df)} rows")
        print(f"- Period/sleep data: {len(period_df)} rows")
        print(f"- Survey responses: {len(survey_df)} rows")
        print(f"- Labeled hormone data: {len(labeled_df)} rows")
        
        return hormone_df, period_df, survey_df, labeled_df
    
    except FileNotFoundError as e:
        print(f"Error: Could not find one or more data files: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def classify_all_subjects(df):
    """
    Classify all subjects in the dataset into menstrual cycle phases.
    
    Args:
        df (pd.DataFrame): DataFrame containing hormone data with columns:
            - subject_id
            - date
            - estradiol
            - progesterone
            - testosterone
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'phase' column containing predictions
    """
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Initialize the classifier
    classifier = MenstrualClassifier()
    
    # Load the training data
    hormone_df, period_df, survey_df, labeled_df = load_data()
    if hormone_df is None or period_df is None or survey_df is None or labeled_df is None:
        raise ValueError("Failed to load training data")
    
    # Create target variable for training
    X, y = create_target_variable(hormone_df, labeled_df)
    
    # Train the classifier
    classifier.train(hormone_df, period_df, survey_df, y)
    
    # Make predictions for each subject
    predictions = []
    for subject_id in result_df['subject_id'].unique():
        # Get data for this subject
        subject_data = result_df[result_df['subject_id'] == subject_id].copy()
        
        # Get corresponding period and survey data
        subject_period = period_df[period_df['subject_id'] == subject_id]
        subject_survey = survey_df[survey_df['subject_id'] == subject_id]
        
        # Make predictions
        preds = classifier.predict(subject_data, subject_period, subject_survey)
        
        # Add predictions to the result
        subject_data.loc[:, 'phase'] = preds
        
        # Add target phase if available in labeled data
        subject_labeled = labeled_df[labeled_df['subject_id'] == subject_id]
        if not subject_labeled.empty:
            # Convert subphase to follicular/luteal
            subject_data['target_phase'] = subject_data['date'].map(
                lambda x: 'luteal' if subject_labeled[subject_labeled['date'] == x]['phase'].iloc[0] in ['early_luteal', 'mid_luteal'] 
                else 'follicular' if subject_labeled[subject_labeled['date'] == x]['phase'].iloc[0] in ['follicular', 'periovulation']
                else None
            )
        
        predictions.append(subject_data)
    
    # Combine all predictions
    result_df = pd.concat(predictions, ignore_index=True)
    
    return result_df

def create_target_variable(unlabeled_df, labeled_df):
    """
    Create target variable by matching unlabeled samples with their corresponding labeled data.
    Convert subphases to follicular/luteal.
    
    Args:
        unlabeled_df (pd.DataFrame): DataFrame containing unlabeled hormone data with sample labels
        labeled_df (pd.DataFrame): DataFrame containing labeled hormone data with phase information
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable
    """
    # Create a copy of unlabeled data to avoid modifying the original
    df = unlabeled_df.copy()
    
    # Initialize target variable
    df['target_phase'] = None
    
    # For each subject and sample, find the corresponding phase in labeled data
    for subject_id in df['subject_id'].unique():
        # Get labeled data for this subject
        subject_labeled = labeled_df[labeled_df['subject_id'] == subject_id]
        
        # Get unlabeled samples for this subject
        subject_unlabeled = df[df['subject_id'] == subject_id]
        
        # For each sample in the unlabeled data
        for _, row in subject_unlabeled.iterrows():
            # Find the matching date in labeled data
            matching_date = subject_labeled[subject_labeled['date'] == row['date']]
            if not matching_date.empty:
                # Convert subphase to follicular/luteal
                subphase = matching_date.iloc[0]['phase']
                if subphase in ['early_luteal', 'mid_late_luteal']:
                    df.loc[row.name, 'target_phase'] = 'luteal'
                else:  # periovulation, mid_follicular, perimenstruation
                    df.loc[row.name, 'target_phase'] = 'follicular'
    
    # Remove rows where we couldn't find a matching phase
    df = df.dropna(subset=['target_phase'])
    
    if len(df) == 0:
        raise ValueError("No matching phases found between labeled and unlabeled data")
    
    print(f"\nCreated target variable with {len(df)} samples")
    print("\nDistribution of phases:")
    print(df['target_phase'].value_counts())
    
    # Return the full DataFrame for feature preparation and the target variable
    return df, df['target_phase']

class MenstrualClassifier:
    def __init__(self):
        """Initialize the classifier with a single model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_training = True  # Flag to track if we're in training mode
        
    def prepare_features(self, hormone_df, period_df, survey_df):
        """Prepare features by combining hormone, period, and survey data."""
        print("\nPreparing features...")
        print(f"Input shapes - hormone_df: {hormone_df.shape}, period_df: {period_df.shape}, survey_df: {survey_df.shape}")
        
        try:
            # Create a mapping of subject_id to their survey response
            survey_dict = survey_df.set_index('subject_id').to_dict('index')
            print(f"Number of subjects with survey data: {len(survey_dict)}")
            
            # Create a new DataFrame with the same structure as hormone_df
            expanded_data = []
            
            # Ensure hormone_df has the required columns
            required_cols = ['subject_id', 'date', 'estradiol', 'progesterone', 'testosterone']
            if not all(col in hormone_df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in hormone_df. Required: {required_cols}, Found: {hormone_df.columns.tolist()}")
            
            # Process each row in hormone_df
            for _, row in hormone_df.iterrows():
                subject_id = row['subject_id']
                # Initialize with hormone data, excluding date and subject_id
                combined = {k: v for k, v in row.to_dict().items() if k not in ['date', 'subject_id']}
                
                # Add period data if available
                subject_period = period_df[period_df['subject_id'] == subject_id].copy()
                if not subject_period.empty:
                    # Convert dates to datetime for proper comparison
                    subject_period['date'] = pd.to_datetime(subject_period['date'])
                    hormone_date = pd.to_datetime(row['date'])
                    
                    # Find the closest date in period data
                    closest_idx = (subject_period['date'] - hormone_date).abs().idxmin()
                    period_row = subject_period.loc[closest_idx]
                    
                    # Use period data, excluding sleep_hours
                    period_data = {k: v for k, v in period_row.to_dict().items() 
                                 if k not in ['cycle_day', 'subject_id', 'date', 'phase', 'sleep_hours']}
                    combined.update(period_data)
                
                # Add survey data if available
                if subject_id in survey_dict:
                    survey_data = survey_dict[subject_id].copy()
                    
                    # Convert date_of_last_period to days since last period
                    if 'date_of_last_period' in survey_data:
                        last_period_date = pd.to_datetime(survey_data['date_of_last_period'])
                        hormone_date = pd.to_datetime(row['date'])
                        days_since_last_period = (hormone_date - last_period_date).days
                        survey_data['days_since_last_period'] = days_since_last_period
                        del survey_data['date_of_last_period']
                    
                    # Remove other non-feature columns
                    for col in ['subject_id', 'date_of_response']:
                        if col in survey_data:
                            del survey_data[col]
                    
                    combined.update(survey_data)
                
                expanded_data.append(combined)
            
            print(f"Number of expanded data entries: {len(expanded_data)}")
            
            # Convert to DataFrame
            features_df = pd.DataFrame(expanded_data)
            print(f"Features DataFrame shape: {features_df.shape}")
            print("Features DataFrame columns:", features_df.columns.tolist())
            
            # Remove target-related columns if they exist
            columns_to_exclude = ['target_phase', 'phase']
            features_df = features_df.drop(columns=[col for col in columns_to_exclude if col in features_df.columns])
            
            # Convert categorical variables to numeric
            # First, ensure menstrual_pattern is treated as categorical
            if 'menstrual_pattern' in features_df.columns:
                features_df['menstrual_pattern'] = features_df['menstrual_pattern'].astype('category')
            
            X = pd.get_dummies(features_df, drop_first=True)
            print(f"After one-hot encoding shape: {X.shape}")
            print("One-hot encoded columns:", X.columns.tolist())

            # During training, store the feature columns
            if self.is_training:
                self.feature_columns = X.columns.tolist()
                print(f"Stored {len(self.feature_columns)} feature columns during training")
            else:
                # During prediction, ensure we have the same columns as training
                if self.feature_columns is None:
                    raise ValueError("Model must be trained before making predictions")
                missing_cols = set(self.feature_columns) - set(X.columns)
                print(f"Missing columns during prediction: {len(missing_cols)}")
                if missing_cols:
                    print("Missing columns:", list(missing_cols))
                    # Create a new DataFrame with all required columns
                    X_new = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
                    # Copy existing columns
                    for col in X.columns:
                        if col in self.feature_columns:
                            X_new[col] = X[col]
                    X = X_new
                print(f"Final feature matrix shape: {X.shape}")
            
            # Scale the features
            if self.is_training:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            print(f"Scaled feature matrix shape: {X_scaled.shape}")
            
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise
    
    def train(self, hormone_df, period_df, survey_df, y):
        """
        Train the model.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            period_df (pd.DataFrame): Period data
            survey_df (pd.DataFrame): Survey data
            y (array-like): Target labels (follicular/luteal)
        """
        print("\nTraining model...")
        print(f"Target variable shape: {len(y)}")
        print(f"Target variable type: {type(y)}")
        print(f"Target variable values: {y.unique()}")
        
        try:
            self.is_training = True
            # Prepare features
            X = self.prepare_features(hormone_df, period_df, survey_df)
            print(f"Feature matrix shape: {X.shape}")
            
            # Train model
            self.model.fit(X, y)
            print("Model training completed")
            self.is_training = False
            
        except Exception as e:
            print(f"Error in train: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            raise
    
    def predict(self, hormone_df, period_df, survey_df):
        """
        Make predictions using the model.
        
        Returns:
            array: Predictions (follicular/luteal)
        """
        self.is_training = False
        # Prepare features
        X = self.prepare_features(hormone_df, period_df, survey_df)

    # Make predictions
        return self.model.predict(X)
    
    def evaluate(self, hormone_df, period_df, survey_df, y_true):
        """
        Evaluate the model's performance.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            period_df (pd.DataFrame): Period data
            survey_df (pd.DataFrame): Survey data
            y_true (array-like): True labels
        """
        y_pred = self.predict(hormone_df, period_df, survey_df)
        
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=PHASES))

def plot_feature_importance(classifier, feature_names, output_path='output/rf/feature_importance.png'):
    """
    Plot and save feature importance.
    
    Args:
        classifier: Trained RandomForestClassifier instance
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get feature importances
    importances = classifier.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")

def main():
    # Load the data
    hormone_df, period_df, survey_df, labeled_df = load_data()
    if hormone_df is None or period_df is None or survey_df is None or labeled_df is None:
        return
    
    # Create output directory
    os.makedirs('output/rf', exist_ok=True)
    
    # Create target variable for training
    X, y = create_target_variable(hormone_df, labeled_df)
    
    # Initialize and train the classifier
    classifier = MenstrualClassifier()
    classifier.train(hormone_df, period_df, survey_df, y)
    
    # Make predictions for all subjects
    predictions_df = classify_all_subjects(hormone_df)
    
    # Save predictions
    predictions_df.to_csv('output/rf/predictions.csv', index=False)
    print("\nSaved predictions to output/rf/predictions.csv")
    
    # Plot feature importance
    if hasattr(classifier.model, 'feature_importances_'):
        plot_feature_importance(classifier.model, classifier.feature_columns)
    
    # Evaluate the model only on samples where we have target phases
    if len(predictions_df) > 0 and 'target_phase' in predictions_df.columns:
        eval_df = predictions_df.dropna(subset=['target_phase'])
        if len(eval_df) > 0:
            accuracy = accuracy_score(eval_df['target_phase'], eval_df['phase'])
            print(f"\nOverall accuracy: {accuracy:.3f}")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(eval_df['target_phase'], eval_df['phase']))
            
            # Plot confusion matrix
            cm = confusion_matrix(eval_df['target_phase'], eval_df['phase'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Follicular', 'Luteal'],
                       yticklabels=['Follicular', 'Luteal'])
            plt.title('Confusion Matrix (Random Forest)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('output/rf/confusion_matrix.png')
            plt.close()
            print("Confusion matrix plot saved to output/rf/confusion_matrix.png")
        else:
            print("\nNo samples with target phases available for evaluation")
    else:
        print("\nNo target phases available for evaluation")

if __name__ == "__main__":
    main()
