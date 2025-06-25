"""
Data loader utility module for loading configuration and data files.
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def load_hormone_data(data_path: str) -> pd.DataFrame:
    """
    Load hormone data from CSV file.
    
    Args:
        data_path (str): Path to the hormone data CSV file
        
    Returns:
        pd.DataFrame: Hormone data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Hormone data file not found: {data_path}")
    
    return pd.read_csv(data_path)


def load_period_data(data_path: str) -> pd.DataFrame:
    """
    Load period data from CSV file.
    
    Args:
        data_path (str): Path to the period data CSV file
        
    Returns:
        pd.DataFrame: Period data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Period data file not found: {data_path}")
    
    return pd.read_csv(data_path)


def load_survey_data(data_path: str) -> pd.DataFrame:
    """
    Load survey data from CSV file.
    
    Args:
        data_path (str): Path to the survey data CSV file
        
    Returns:
        pd.DataFrame: Survey data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Survey data file not found: {data_path}")
    
    return pd.read_csv(data_path)


def load_pattern_data(data_path: str) -> pd.DataFrame:
    """
    Load menstrual pattern data from CSV file.
    
    Args:
        data_path (str): Path to the pattern data CSV file
        
    Returns:
        pd.DataFrame: Pattern data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Pattern data file not found: {data_path}")
    
    return pd.read_csv(data_path)


def save_data(data: pd.DataFrame, output_path: str) -> None:
    """
    Save data to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        output_path (str): Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)


def check_data_files(config: Dict[str, Any]) -> bool:
    """
    Check if all required data files exist.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    data_config = config.get('data', {})
    required_files = [
        data_config.get('hormone_data_path'),
        data_config.get('survey_data_path')
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path and not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        return False
    
    return True


def load_and_split_data(file_path: str, test_size: float = 0.2, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data from CSV file and split into train/test sets.
    
    Args:
        file_path (str): Path to the data CSV file
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: None)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) Train and test feature matrices and target variables
    """
    # Load data
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = pd.read_csv(file_path)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train-test split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data for classification.
    
    Args:
        data (pd.DataFrame): Raw data with target column
        
    Returns:
        tuple: (X, y) Feature matrix and target variable
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    # Assuming the target column is 'phase' or 'menstrual_pattern'
    target_columns = ['phase', 'menstrual_pattern', 'target']
    target_col = None
    
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")
    
    # Extract target variable
    y = df[target_col]
    
    # Remove target and non-feature columns
    feature_columns = [col for col in df.columns if col not in 
                      target_columns + ['subject_id', 'date', 'cycle_day', 'cycle_position']]
    
    X = df[feature_columns]
    
    # Convert categorical variables to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y


def preprocess_data_with_prior(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data for classification with rule-based prior as features.
    
    Args:
        data (pd.DataFrame): Raw data with target column
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        tuple: (X, y) Feature matrix with prior features and target variable
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    target_columns = ['phase', 'menstrual_pattern', 'target']
    target_col = None
    
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")
    
    # Extract target variable
    y = df[target_col]
    
    # Remove target and non-feature columns
    feature_columns = [col for col in df.columns if col not in 
                      target_columns + ['subject_id', 'date', 'cycle_day', 'cycle_position']]
    
    X = df[feature_columns]
    
    # Add rule-based prior as features if enabled
    if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
        try:
            from src.temporal_models.rule_based_prior import RuleBasedPrior
            
            # Initialize rule-based prior
            rule_prior = RuleBasedPrior(config)
            rule_prior.load_data()
            
            # Get prior predictions
            prior_predictions = rule_prior.predict_phases(df)
            
            # Add prior predictions as a feature
            X['prior_phase'] = prior_predictions
            
            # Convert prior phases to one-hot encoding
            phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
            prior_encoded = pd.get_dummies(X['prior_phase'], prefix='prior')
            
            # Ensure all phases are present (fill missing with 0)
            for phase in phases:
                col_name = f'prior_{phase}'
                if col_name not in prior_encoded.columns:
                    prior_encoded[col_name] = 0
            
            # Remove the original prior_phase column and add encoded features
            X = X.drop('prior_phase', axis=1)
            X = pd.concat([X, prior_encoded], axis=1)
            
            print(f"Added {len(prior_encoded.columns)} prior features to the dataset")
            
        except Exception as e:
            print(f"Warning: Could not add prior features: {str(e)}")
    
    # Convert categorical variables to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y


def preprocess_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess unlabeled data for prediction.
    
    Args:
        data (pd.DataFrame): Raw data without target column
        
    Returns:
        pd.DataFrame: Preprocessed feature matrix
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove non-feature columns
    non_feature_columns = ['subject_id', 'date', 'cycle_day', 'cycle_position', 
                          'phase', 'menstrual_pattern', 'target']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    
    X = df[feature_columns]
    
    # Convert categorical variables to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X


def preprocess_unlabeled_data_with_prior(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess unlabeled data for prediction with rule-based prior as features.
    
    Args:
        data (pd.DataFrame): Raw data without target column
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        pd.DataFrame: Preprocessed feature matrix with prior features
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Remove non-feature columns
    non_feature_columns = ['subject_id', 'date', 'cycle_day', 'cycle_position', 
                          'phase', 'menstrual_pattern', 'target']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    
    X = df[feature_columns]
    
    # Add rule-based prior as features if enabled
    if config.get('models', {}).get('temporal', {}).get('use_as_prior', False):
        try:
            from src.temporal_models.rule_based_prior import RuleBasedPrior
            
            # Initialize rule-based prior
            rule_prior = RuleBasedPrior(config)
            rule_prior.load_data()
            
            # Get prior predictions
            prior_predictions = rule_prior.predict_phases(df)
            
            # Add prior predictions as a feature
            X['prior_phase'] = prior_predictions
            
            # Convert prior phases to one-hot encoding
            phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
            prior_encoded = pd.get_dummies(X['prior_phase'], prefix='prior')
            
            # Ensure all phases are present (fill missing with 0)
            for phase in phases:
                col_name = f'prior_{phase}'
                if col_name not in prior_encoded.columns:
                    prior_encoded[col_name] = 0
            
            # Remove the original prior_phase column and add encoded features
            X = X.drop('prior_phase', axis=1)
            X = pd.concat([X, prior_encoded], axis=1)
            
            print(f"Added {len(prior_encoded.columns)} prior features to the dataset")
            
        except Exception as e:
            print(f"Warning: Could not add prior features: {str(e)}")
    
    # Convert categorical variables to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X 