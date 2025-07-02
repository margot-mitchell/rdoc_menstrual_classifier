"""
Data loader utility module for loading configuration and data files.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessor that handles consistent scaling and feature engineering between training and prediction.
    Persists the scaler to ensure the same scaling is applied during prediction.
    Handles hormone, interaction, ratio, prior, and categorical features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        self.categorical_encoders = {}
        self.use_prior = self.config.get('models', {}).get('temporal', {}).get('use_as_prior', False)
        self.prior_config = self.config if self.use_prior else None
        
    def process_features(self, df: pd.DataFrame, add_prior: bool = False) -> pd.DataFrame:
        """
        Unified feature engineering pipeline.
        Args:
            df: Input DataFrame
            add_prior: Whether to add prior features
        Returns:
            pd.DataFrame: Feature matrix
        """
        features_config = self.config.get('features', {})
        hormone_features = features_config.get('hormone_features', ['estradiol', 'progesterone', 'testosterone'])
        X = df[hormone_features].copy()
        
        # Add interaction features
        if features_config.get('add_interaction_features', False):
            for i, feat1 in enumerate(hormone_features):
                for feat2 in hormone_features[i+1:]:
                    interaction_name = f"{feat1}_{feat2}_interaction"
                    X[interaction_name] = df[feat1] * df[feat2]
        
        # Add ratio features
        if features_config.get('add_ratio_features', False):
            for i, feat1 in enumerate(hormone_features):
                for feat2 in hormone_features[i+1:]:
                    ratio_name = f"{feat1}_{feat2}_ratio"
                    X[ratio_name] = df[feat1] / (df[feat2] + 1e-8)
        
        # Add prior features if enabled
        if add_prior and self.use_prior:
            try:
                from src.classification.rule_based_prior import RuleBasedPrior
                rule_prior = RuleBasedPrior(self.prior_config)
                rule_prior.load_data()
                
                # Generate hormone rule features instead of just predictions
                hormone_rule_features = rule_prior.generate_hormone_rule_features(df)
                
                # Also get prior predictions for comparison
                prior_predictions = rule_prior.predict_phases(df)
                
                # Add hormone rule features
                X = pd.concat([X, hormone_rule_features], axis=1)
                
                # Add prior predictions as categorical features
                X['prior_phase'] = prior_predictions
                phases = ['perimenstruation', 'mid_follicular', 'periovulation', 'early_luteal', 'mid_late_luteal']
                prior_encoded = pd.get_dummies(X['prior_phase'], prefix='prior')
                for phase in phases:
                    col_name = f'prior_{phase}'
                    if col_name not in prior_encoded.columns:
                        prior_encoded[col_name] = 0
                X = X.drop('prior_phase', axis=1)
                X = pd.concat([X, prior_encoded], axis=1)
                
                logger.info(f"Added {len(hormone_rule_features.columns)} hormone rule features and {len(prior_encoded.columns)} prior features to the dataset")
            except Exception as e:
                logger.warning(f"Could not add prior features: {str(e)}")
        
        # Add any additional features (future-proofing)
        # ...
        return X
    
    def fit(self, df: pd.DataFrame, add_prior: bool = False) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            add_prior: Whether to add prior features
            
        Returns:
            self: Fitted preprocessor
        """
        X = self.process_features(df, add_prior=add_prior)
        self.feature_names = list(X.columns)
        X_processed = self._encode_categorical_variables(X, is_training=True)
        self.scaler = StandardScaler()
        self.scaler.fit(X_processed)
        self.is_fitted = True
        logger.info(f"Fitted preprocessor with {len(self.feature_names)} features")
        return self
    
    def transform(self, df: pd.DataFrame, add_prior: bool = False) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Feature matrix to transform
            add_prior: Whether to add prior features
            
        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = self.process_features(df, add_prior=add_prior)
        X_aligned = self._align_features(X)
        
        # Validate feature alignment
        self._validate_features(X_aligned)
        
        X_processed = self._encode_categorical_variables(X_aligned, is_training=False)
        X_scaled = self.scaler.transform(X_processed)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, df: pd.DataFrame, add_prior: bool = False) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Feature matrix to fit and transform
            add_prior: Whether to add prior features
            
        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        self.fit(df, add_prior=add_prior)
        return self.transform(df, add_prior=add_prior)
    
    def _encode_categorical_variables(self, X: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Encode categorical variables consistently.
        
        Args:
            X: Feature matrix
            is_training: Whether this is training data
            
        Returns:
            pd.DataFrame: Feature matrix with encoded categorical variables
        """
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                if is_training:
                    # During training, create encoder and fit
                    unique_values = X[col].unique()
                    self.categorical_encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
                    X_encoded[col] = X[col].map(self.categorical_encoders[col])
                else:
                    # During prediction, use existing encoder
                    if col in self.categorical_encoders:
                        X_encoded[col] = X[col].map(self.categorical_encoders[col])
                        # Fill unknown values with -1
                        X_encoded[col] = X_encoded[col].fillna(-1)
                    else:
                        # If encoder doesn't exist, use simple encoding
                        X_encoded[col] = pd.Categorical(X[col]).codes
        
        return X_encoded
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure X has the same features as training data.
        
        Args:
            X: Feature matrix to align
            
        Returns:
            pd.DataFrame: Aligned feature matrix
        """
        if self.feature_names is None:
            raise ValueError("Preprocessor must be fitted before aligning features")
        
        # Create DataFrame with same columns as training
        X_aligned = pd.DataFrame(0, index=X.index, columns=self.feature_names)
        
        # Copy existing columns
        for col in X.columns:
            if col in self.feature_names:
                X_aligned[col] = X[col]
        
        return X_aligned
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that the feature matrix has the expected features.
        
        Args:
            X: Feature matrix to validate
            
        Raises:
            ValueError: If feature validation fails
        """
        if self.feature_names is None:
            raise ValueError("Preprocessor must be fitted before validating features")
        
        # Check feature count
        if X.shape[1] != len(self.feature_names):
            from src.utils.model_utils import debug_feature_mismatch
            debug_msg = debug_feature_mismatch(self.feature_names, list(X.columns), "DataPreprocessor")
            raise ValueError(
                f"Feature count mismatch: expected {len(self.feature_names)} features, "
                f"got {X.shape[1]} features. {debug_msg}"
            )
        
        # Check feature names
        if list(X.columns) != self.feature_names:
            from src.utils.model_utils import debug_feature_mismatch
            debug_msg = debug_feature_mismatch(self.feature_names, list(X.columns), "DataPreprocessor")
            raise ValueError(f"Feature mismatch detected: {debug_msg}")
        
        logger.info(f"Feature validation passed: {len(self.feature_names)} features aligned")
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_encoders': self.categorical_encoders,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load(self, filepath: str) -> 'DataPreprocessor':
        """
        Load a fitted preprocessor.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            self: Loaded preprocessor
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        self.categorical_encoders = preprocessor_data.get('categorical_encoders', {})
        self.config = preprocessor_data.get('config', {})
        self.is_fitted = preprocessor_data.get('is_fitted', True)
        
        logger.info(f"Loaded preprocessor from {filepath}")
        return self


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
    
    # Preprocess data using the new preprocessor
    X, y = preprocess_data(data)
    
    # Train-test split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess_data(data: pd.DataFrame, preprocessor: Optional[DataPreprocessor] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data for classification using the persistent preprocessor.
    Args:
        data (pd.DataFrame): Raw data with target column
        preprocessor (DataPreprocessor, optional): Fitted preprocessor for prediction
    Returns:
        tuple: (X, y) Feature matrix and target variable
    """
    df = data.copy().dropna()
    target_columns = ['phase', 'menstrual_pattern', 'target']
    target_col = next((col for col in target_columns if col in df.columns), None)
    if target_col is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")
    y = df[target_col]
    if preprocessor is not None and preprocessor.is_fitted:
        X_processed = preprocessor.transform(df, add_prior=False)
    else:
        if preprocessor is None:
            preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(df, add_prior=False)
    return X_processed, y


def preprocess_data_with_prior(data: pd.DataFrame, config: Dict[str, Any], preprocessor: Optional[DataPreprocessor] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess data for classification with rule-based prior as features.
    Args:
        data (pd.DataFrame): Raw data with target column
        config (Dict[str, Any]): Configuration dictionary
        preprocessor (DataPreprocessor, optional): Fitted preprocessor for prediction
    Returns:
        tuple: (X, y) Feature matrix with prior features and target variable
    """
    df = data.copy().dropna()
    target_columns = ['phase', 'menstrual_pattern', 'target']
    target_col = next((col for col in target_columns if col in df.columns), None)
    if target_col is None:
        raise ValueError(f"No target column found. Expected one of: {target_columns}")
    y = df[target_col]
    if preprocessor is not None and preprocessor.is_fitted:
        X_processed = preprocessor.transform(df, add_prior=True)
    else:
        if preprocessor is None:
            preprocessor = DataPreprocessor(config)
        X_processed = preprocessor.fit_transform(df, add_prior=True)
    return X_processed, y


def preprocess_unlabeled_data(data: pd.DataFrame, preprocessor: DataPreprocessor) -> pd.DataFrame:
    """
    Preprocess unlabeled data for prediction using a fitted preprocessor.
    Args:
        data (pd.DataFrame): Raw data without target column
        preprocessor (DataPreprocessor): Fitted preprocessor
    Returns:
        pd.DataFrame: Preprocessed feature matrix
    """
    if not preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted before preprocessing unlabeled data")
    df = data.copy().dropna()
    X_processed = preprocessor.transform(df, add_prior=False)
    return X_processed


def preprocess_unlabeled_data_with_prior(data: pd.DataFrame, config: Dict[str, Any], preprocessor: DataPreprocessor) -> pd.DataFrame:
    """
    Preprocess unlabeled data for prediction with rule-based prior as features.
    Args:
        data (pd.DataFrame): Raw data without target column
        config (Dict[str, Any]): Configuration dictionary
        preprocessor (DataPreprocessor): Fitted preprocessor
    Returns:
        pd.DataFrame: Preprocessed feature matrix with prior features
    """
    if not preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted before preprocessing unlabeled data")
    df = data.copy().dropna()
    X_processed = preprocessor.transform(df, add_prior=True)
    return X_processed 