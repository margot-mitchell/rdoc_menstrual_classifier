"""
Model utilities for saving and loading trained models with comprehensive metadata.
"""

import os
import joblib
import logging
import json
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelBundle:
    """
    A comprehensive model bundle that contains all artifacts needed for prediction.
    This ensures consistency between training and prediction by bundling everything together.
    """
    
    def __init__(self, 
                 model: Any,
                 preprocessor: Any,
                 config: Dict[str, Any],
                 training_metrics: Dict[str, Any],
                 feature_names: list,
                 model_name: str,
                 label_encoder: Any = None):
        """
        Initialize a model bundle.
        
        Args:
            model: The trained model object
            preprocessor: The fitted data preprocessor
            config: Configuration used for training
            training_metrics: Training performance metrics
            feature_names: List of feature names used for training
            model_name: Name of the model
            label_encoder: Optional label encoder (for models like XGBoost)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.training_metrics = training_metrics
        self.feature_names = feature_names
        self.model_name = model_name
        self.label_encoder = label_encoder
        self.created_at = datetime.now().isoformat()
        
        # Validate the bundle
        self._validate()
    
    def _validate(self):
        """Validate the model bundle components."""
        if self.model is None:
            raise ValueError("Model cannot be None")
        if self.preprocessor is None:
            raise ValueError("Preprocessor cannot be None")
        if not self.feature_names:
            raise ValueError("Feature names cannot be empty")
        if not self.config:
            raise ValueError("Config cannot be empty")
    
    def save(self, filepath: str) -> str:
        """
        Save the complete model bundle to a single file.
        
        Args:
            filepath: Path where to save the bundle
            
        Returns:
            str: Path to the saved bundle
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare bundle data
        bundle_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'created_at': self.created_at,
            'bundle_version': '1.0',
            'label_encoder': self.label_encoder
        }
        
        # Save bundle
        joblib.dump(bundle_data, filepath)
        logger.info(f"Model bundle saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelBundle':
        """
        Load a model bundle from file.
        
        Args:
            filepath: Path to the saved bundle
            
        Returns:
            ModelBundle: Loaded model bundle
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model bundle not found: {filepath}")
        
        # Load bundle data
        bundle_data = joblib.load(filepath)
        
        # Create bundle instance
        bundle = cls(
            model=bundle_data['model'],
            preprocessor=bundle_data['preprocessor'],
            config=bundle_data['config'],
            training_metrics=bundle_data['training_metrics'],
            feature_names=bundle_data['feature_names'],
            model_name=bundle_data['model_name'],
            label_encoder=bundle_data.get('label_encoder', None)
        )
        
        # Restore creation time
        bundle.created_at = bundle_data.get('created_at', datetime.now().isoformat())
        
        logger.info(f"Model bundle loaded from: {filepath}")
        return bundle
    
    def validate_prediction_data(self, X: pd.DataFrame) -> bool:
        """
        Validate that prediction data is compatible with the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            bool: True if compatible, raises error if not
        """
        # Check feature count
        if X.shape[1] != len(self.feature_names):
            actual_features = list(X.columns) if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
            debug_msg = debug_feature_mismatch(self.feature_names, actual_features, self.model_name)
            raise ValueError(
                f"Feature count mismatch for model '{self.model_name}': "
                f"expected {len(self.feature_names)} features, got {X.shape[1]} features. "
                f"{debug_msg}"
            )
        
        # Check feature names and order (if available)
        if hasattr(X, 'columns'):
            # Check for missing features
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                debug_msg = debug_feature_mismatch(self.feature_names, list(X.columns), self.model_name)
                raise ValueError(
                    f"Missing required features for model '{self.model_name}'. "
                    f"{debug_msg}"
                )
            
            # Check for extra features
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                logger.warning(
                    f"Extra features found for model '{self.model_name}': {extra_features}. "
                    f"These will be ignored during prediction."
                )
            
            # Check feature order
            if list(X.columns) != self.feature_names:
                logger.warning(
                    f"Feature order mismatch for model '{self.model_name}'. "
                    f"Expected order: {self.feature_names}. "
                    f"Actual order: {list(X.columns)}. "
                    f"Reordering features to match expected order."
                )
                # Reorder features to match expected order
                X = X[self.feature_names]
        
        logger.info(f"Feature validation passed for model '{self.model_name}'")
        return True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the bundled model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        # Validate input data
        self.validate_prediction_data(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        logger.info(f"Made predictions using {self.model_name} model")
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using the bundled model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        # Validate input data
        self.validate_prediction_data(X)
        
        # Check if model supports probability predictions
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_name} does not support probability predictions")
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        logger.info(f"Got prediction probabilities using {self.model_name} model")
        return probabilities
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model bundle.
        
        Returns:
            dict: Model bundle information
        """
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'preprocessor_type': type(self.preprocessor).__name__,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'created_at': self.created_at,
            'config_summary': {
                'data_config': self.config.get('data', {}),
                'models_config': self.config.get('models', {}),
                'output_config': self.config.get('output', {})
            }
        }
        
        # Add model-specific information
        if hasattr(self.model, 'classes_'):
            info['n_classes'] = len(self.model.classes_)
            info['classes'] = list(self.model.classes_)
        
        # Add label encoder classes if present
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            info['label_encoder_classes'] = list(self.label_encoder.classes_)
        
        return info


def save_model_bundle(model: Any, 
                     preprocessor: Any,
                     config: Dict[str, Any],
                     training_metrics: Dict[str, Any],
                     feature_names: list,
                     model_name: str,
                     output_dir: str,
                     label_encoder: Any = None) -> str:
    """
    Save a complete model bundle.
    
    Args:
        model: The trained model
        preprocessor: The fitted preprocessor
        config: Training configuration
        training_metrics: Training performance metrics
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save the bundle
        label_encoder: Optional label encoder (for models like XGBoost)
        
    Returns:
        str: Path to the saved bundle
    """
    # Create model bundle
    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        config=config,
        training_metrics=training_metrics,
        feature_names=feature_names,
        model_name=model_name,
        label_encoder=label_encoder
    )
    
    # Save bundle
    filepath = os.path.join(output_dir, f'{model_name}_bundle.joblib')
    bundle.save(filepath)
    
    return filepath


def load_model_bundle(filepath: str) -> ModelBundle:
    """
    Load a model bundle.
    
    Args:
        filepath: Path to the saved bundle
        
    Returns:
        ModelBundle: Loaded model bundle
    """
    return ModelBundle.load(filepath)


def list_available_bundles(models_dir: str) -> list:
    """
    List all available model bundles.
    
    Args:
        models_dir: Directory containing saved bundles
        
    Returns:
        list: List of available bundle files
    """
    if not os.path.exists(models_dir):
        return []
    
    bundle_files = [f for f in os.listdir(models_dir) if f.endswith('_bundle.joblib')]
    return bundle_files


def get_bundle_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a saved model bundle.
    
    Args:
        filepath: Path to the bundle file
        
    Returns:
        dict: Bundle information
    """
    try:
        bundle = ModelBundle.load(filepath)
        return bundle.get_info()
    except Exception as e:
        logger.error(f"Error getting bundle info: {str(e)}")
        return {}


# Legacy functions for backward compatibility
def save_model(model, model_name: str, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Legacy function for backward compatibility.
    Use save_model_bundle instead for new code.
    """
    logger.warning("save_model is deprecated. Use save_model_bundle instead.")
    
    # Create a minimal bundle for backward compatibility
    if metadata is None:
        metadata = {}
    
    # Extract components from metadata
    preprocessor = metadata.get('preprocessor')
    config = metadata.get('config', {})
    training_metrics = metadata.get('training_metrics', {})
    feature_names = metadata.get('feature_names', [])
    
    return save_model_bundle(
        model=model,
        preprocessor=preprocessor,
        config=config,
        training_metrics=training_metrics,
        feature_names=feature_names,
        model_name=model_name,
        output_dir=output_dir
    )


def load_model(model_path: str):
    """
    Legacy function for backward compatibility.
    Use load_model_bundle instead for new code.
    """
    logger.warning("load_model is deprecated. Use load_model_bundle instead.")
    
    bundle = ModelBundle.load(model_path)
    return bundle.model, bundle.model_name, bundle.get_info()


def predict_with_model(model_path: str, X: pd.DataFrame) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Use ModelBundle.predict instead for new code.
    """
    logger.warning("predict_with_model is deprecated. Use ModelBundle.predict instead.")
    
    bundle = ModelBundle.load(model_path)
    return bundle.predict(X)


def predict_proba_with_model(model_path: str, X: pd.DataFrame) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Use ModelBundle.predict_proba instead for new code.
    """
    logger.warning("predict_proba_with_model is deprecated. Use ModelBundle.predict_proba instead.")
    
    bundle = ModelBundle.load(model_path)
    return bundle.predict_proba(X)


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Use get_bundle_info instead for new code.
    """
    logger.warning("get_model_info is deprecated. Use get_bundle_info instead.")
    
    return get_bundle_info(model_path)


def debug_feature_mismatch(expected_features: list, actual_features: list, model_name: str = "unknown") -> str:
    """
    Generate a detailed debug message for feature mismatches.
    
    Args:
        expected_features: List of expected feature names
        actual_features: List of actual feature names
        model_name: Name of the model for context
        
    Returns:
        str: Detailed debug message
    """
    missing_features = set(expected_features) - set(actual_features)
    extra_features = set(actual_features) - set(expected_features)
    
    debug_msg = f"\n=== FEATURE MISMATCH DEBUG for {model_name} ===\n"
    debug_msg += f"Expected {len(expected_features)} features: {expected_features}\n"
    debug_msg += f"Actual {len(actual_features)} features: {actual_features}\n"
    
    if missing_features:
        debug_msg += f"\n❌ MISSING FEATURES ({len(missing_features)}): {list(missing_features)}\n"
    
    if extra_features:
        debug_msg += f"\n⚠️  EXTRA FEATURES ({len(extra_features)}): {list(extra_features)}\n"
    
    if not missing_features and not extra_features:
        debug_msg += f"\n✅ All features present, but order may be different.\n"
        debug_msg += f"Expected order: {expected_features}\n"
        debug_msg += f"Actual order: {actual_features}\n"
    
    debug_msg += f"\n=== TROUBLESHOOTING ===\n"
    debug_msg += f"1. Check if you're using the correct preprocessor for this model\n"
    debug_msg += f"2. Verify that prior features are enabled/disabled consistently\n"
    debug_msg += f"3. Ensure the same feature engineering pipeline is used for training and prediction\n"
    debug_msg += f"4. Check if the model was trained with different configuration\n"
    
    return debug_msg 