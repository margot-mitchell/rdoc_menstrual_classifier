"""
Data preprocessor utility module for data preprocessing and feature engineering.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.data_loader import (
    load_hormone_data,
    load_period_data,
    load_survey_data,
    load_pattern_data
)


class DataProcessor:
    """Data processor for preprocessing and feature engineering."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required data files.
        
        Returns:
            tuple: (hormone_df, period_df, survey_df, pattern_df)
        """
        data_config = self.config['data']
        
        # Load hormone data
        hormone_df = load_hormone_data(data_config['hormone_data_path'])
        
        # Load period data (if available)
        period_df = None
        if 'period_data_path' in data_config:
            try:
                period_df = load_period_data(data_config['period_data_path'])
            except FileNotFoundError:
                print("Period data not found, proceeding without it")
        
        # Load survey data
        survey_df = load_survey_data(data_config['survey_data_path'])
        
        # Load pattern data (if available)
        pattern_df = None
        if 'pattern_data_path' in data_config:
            try:
                pattern_df = load_pattern_data(data_config['pattern_data_path'])
            except FileNotFoundError:
                print("Pattern data not found, proceeding without it")
        
        return hormone_df, period_df, survey_df, pattern_df
    
    def prepare_features(self, hormone_df: pd.DataFrame, period_df: pd.DataFrame = None, 
                        survey_df: pd.DataFrame = None, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for classification.
        
        Args:
            hormone_df (pd.DataFrame): Hormone data
            period_df (pd.DataFrame): Period data (optional)
            survey_df (pd.DataFrame): Survey data (optional)
            is_training (bool): Whether this is for training (affects scaling)
            
        Returns:
            tuple: (X, y) Feature matrix and target variable
        """
        features_config = self.config['features']
        
        # Start with hormone features
        hormone_features = features_config['hormone_features']
        X = hormone_df[hormone_features].copy()
        
        # Add interaction features if configured
        if features_config.get('add_interaction_features', False):
            X = self._add_interaction_features(X)
        
        # Add ratio features if configured
        if features_config.get('add_ratio_features', False):
            X = self._add_ratio_features(X)
        
        # Normalize features if configured
        if features_config.get('normalize_features', False):
            X = self._normalize_features(X, is_training)
        
        # Create target variable from survey data
        y = self._create_target_variable(hormone_df, survey_df)
        
        # Store feature names
        if is_training:
            self.feature_names = X.columns.tolist()
        
        return X, y
    
    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between hormone levels."""
        hormone_features = self.config['features']['hormone_features']
        
        # Add pairwise interactions
        for i, feat1 in enumerate(hormone_features):
            for feat2 in hormone_features[i+1:]:
                interaction_name = f"{feat1}_{feat2}_interaction"
                X[interaction_name] = X[feat1] * X[feat2]
        
        return X
    
    def _add_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add ratio features between hormone levels."""
        hormone_features = self.config['features']['hormone_features']
        
        # Add ratios (avoid division by zero)
        for i, feat1 in enumerate(hormone_features):
            for feat2 in hormone_features[i+1:]:
                ratio_name = f"{feat1}_{feat2}_ratio"
                # Add small epsilon to avoid division by zero
                X[ratio_name] = X[feat1] / (X[feat2] + 1e-8)
        
        return X
    
    def _normalize_features(self, X: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Normalize features using StandardScaler."""
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def _create_target_variable(self, hormone_df: pd.DataFrame, survey_df: pd.DataFrame) -> pd.Series:
        """Create target variable from survey data."""
        # Merge hormone data with survey data based on subject_id
        merged_df = hormone_df.merge(survey_df, on='subject_id', how='left')
        
        # Create target variable (example: classify based on menstrual pattern)
        # This is a simplified example - you may need to adjust based on your specific needs
        if 'menstrual_pattern' in merged_df.columns:
            # Convert pattern to numeric for classification
            pattern_mapping = {
                'regular': 0,
                'irregular': 1,
                'anovulatory': 2
            }
            y = merged_df['menstrual_pattern'].map(pattern_mapping)
        else:
            # Fallback: use cycle length as target (binarize for classification)
            cycle_length = merged_df.get('cycle_length', 28)
            y = (cycle_length > 30).astype(int)  # Binary classification: long vs short cycles
        
        return y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        data_config = self.config['data']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_config['test_size'],
            random_state=data_config['random_state'],
            stratify=y if len(y.unique()) > 1 else None
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> list:
        """Get feature names."""
        return self.feature_names if self.feature_names else []
    
    def save_preprocessor(self, output_path: str) -> None:
        """Save preprocessor state."""
        import joblib
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(preprocessor_data, output_path)
    
    def load_preprocessor(self, input_path: str) -> None:
        """Load preprocessor state."""
        import joblib
        preprocessor_data = joblib.load(input_path)
        self.scaler = preprocessor_data['scaler']
        self.feature_names = preprocessor_data['feature_names']
        self.config = preprocessor_data['config'] 