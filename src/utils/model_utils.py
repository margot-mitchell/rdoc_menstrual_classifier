"""
Model utilities for saving and loading trained models.
"""

import os
import joblib
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def save_model(model, model_name: str, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save a trained model with metadata.
    
    Args:
        model: The trained model object
        model_name: Name of the model (e.g., 'random_forest', 'logistic_regression')
        output_dir: Directory to save the model
        metadata: Additional metadata to save with the model
        
    Returns:
        str: Path to the saved model file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model data dictionary
    model_data = {
        'model': model,
        'model_name': model_name,
        'metadata': metadata or {}
    }
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
    joblib.dump(model_data, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    return model_path


def load_model(model_path: str):
    """
    Load a trained model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        tuple: (model, model_name, metadata)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    model_name = model_data.get('model_name', 'unknown')
    metadata = model_data.get('metadata', {})
    
    logger.info(f"Model loaded from: {model_path}")
    return model, model_name, metadata


def list_available_models(models_dir: str) -> list:
    """
    List all available trained models.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        list: List of available model files
    """
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    return model_files


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Get information about a saved model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        dict: Model information
    """
    try:
        model, model_name, metadata = load_model(model_path)
        
        info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'metadata': metadata,
            'file_size': os.path.getsize(model_path)
        }
        
        # Add model-specific information
        if hasattr(model, 'feature_names_in_'):
            info['n_features'] = len(model.feature_names_in_)
            info['feature_names'] = list(model.feature_names_in_)
        
        if hasattr(model, 'classes_'):
            info['n_classes'] = len(model.classes_)
            info['classes'] = list(model.classes_)
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}


def predict_with_model(model_path: str, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a saved model.
    
    Args:
        model_path: Path to the saved model
        X: Feature matrix for prediction
        
    Returns:
        np.ndarray: Predictions
    """
    model, model_name, metadata = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(X)
    
    logger.info(f"Made predictions using {model_name} model")
    return predictions


def predict_proba_with_model(model_path: str, X: pd.DataFrame) -> np.ndarray:
    """
    Get prediction probabilities using a saved model.
    
    Args:
        model_path: Path to the saved model
        X: Feature matrix for prediction
        
    Returns:
        np.ndarray: Prediction probabilities
    """
    model, model_name, metadata = load_model(model_path)
    
    # Check if model supports probability predictions
    if not hasattr(model, 'predict_proba'):
        raise ValueError(f"Model {model_name} does not support probability predictions")
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X)
    
    logger.info(f"Got prediction probabilities using {model_name} model")
    return probabilities 