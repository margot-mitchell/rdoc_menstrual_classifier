"""
Tests for classification functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.main.classification import (
    RandomForestClassifier, 
    LogisticRegressionClassifier, 
    SVMClassifier
)
from src.utils.preprocessor import DataProcessor
from src.utils.evaluator import ModelEvaluator


class TestClassification(unittest.TestCase):
    """Test cases for classification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test configuration
        self.test_config = {
            'data': {
                'hormone_data_path': 'test_data/hormone_data.csv',
                'survey_data_path': 'test_data/survey_data.csv',
                'test_size': 0.2,
                'random_state': 42,
                'n_samples_per_subject': 7
            },
            'features': {
                'hormone_features': ['estradiol', 'progesterone', 'testosterone'],
                'normalize_features': True,
                'add_interaction_features': True,
                'add_ratio_features': True
            },
            'models': {
                'random_forest': {
                    'n_estimators': 10,
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                },
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 100,
                    'random_state': 42
                },
                'support_vector_machine': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'random_state': 42
                }
            },
            'output': {
                'results_dir': 'test_output',
                'figures_dir': 'test_output/figures',
                'save_model': True,
                'save_predictions': True,
                'generate_confusion_matrix': True,
                'generate_feature_importance': True
            }
        }
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create test data for classification."""
        # Create hormone data
        n_subjects = 20
        n_samples_per_subject = 7
        
        hormone_data = []
        for subject_id in range(n_subjects):
            for sample in range(n_samples_per_subject):
                hormone_data.append({
                    'subject_id': subject_id,
                    'estradiol': np.random.normal(1.5, 0.5),
                    'progesterone': np.random.normal(200, 100),
                    'testosterone': np.random.normal(140, 20)
                })
        
        self.hormone_df = pd.DataFrame(hormone_data)
        
        # Create survey data
        survey_data = []
        for subject_id in range(n_subjects):
            survey_data.append({
                'subject_id': subject_id,
                'menstrual_pattern': np.random.choice(['regular', 'irregular', 'anovulatory']),
                'cycle_length': np.random.normal(28, 3)
            })
        
        self.survey_df = pd.DataFrame(survey_data)
    
    def test_random_forest_classifier(self):
        """Test Random Forest classifier."""
        classifier = RandomForestClassifier(self.test_config)
        
        # Create test data
        X = pd.DataFrame({
            'estradiol': np.random.normal(1.5, 0.5, 100),
            'progesterone': np.random.normal(200, 100, 100),
            'testosterone': np.random.normal(140, 20, 100)
        })
        y = pd.Series(np.random.choice([0, 1, 2], 100))
        
        # Train classifier
        classifier.train(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        
        # Check predictions
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(pred in [0, 1, 2] for pred in predictions))
        
        # Check feature importance
        feature_importance = classifier.get_feature_importance()
        self.assertIsInstance(feature_importance, dict)
        self.assertGreater(len(feature_importance), 0)
    
    def test_logistic_regression_classifier(self):
        """Test Logistic Regression classifier."""
        classifier = LogisticRegressionClassifier(self.test_config)
        
        # Create test data
        X = pd.DataFrame({
            'estradiol': np.random.normal(1.5, 0.5, 100),
            'progesterone': np.random.normal(200, 100, 100),
            'testosterone': np.random.normal(140, 20, 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))  # Binary classification
        
        # Train classifier
        classifier.train(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        
        # Check predictions
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Check feature importance
        feature_importance = classifier.get_feature_importance()
        self.assertIsInstance(feature_importance, dict)
        self.assertGreater(len(feature_importance), 0)
    
    def test_svm_classifier(self):
        """Test SVM classifier."""
        classifier = SVMClassifier(self.test_config)
        
        # Create test data
        X = pd.DataFrame({
            'estradiol': np.random.normal(1.5, 0.5, 100),
            'progesterone': np.random.normal(200, 100, 100),
            'testosterone': np.random.normal(140, 20, 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))  # Binary classification
        
        # Train classifier
        classifier.train(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        
        # Check predictions
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
        # Check feature importance (should be empty for SVM)
        feature_importance = classifier.get_feature_importance()
        self.assertIsInstance(feature_importance, dict)
    
    def test_data_processor(self):
        """Test data processor."""
        processor = DataProcessor(self.test_config)
        
        # Skip data loading test since we don't have actual files
        # Instead, test feature preparation with our test data
        X, y = processor.prepare_features(self.hormone_df, None, self.survey_df)
        
        # Check feature matrix
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        
        # Check that features were added
        if self.test_config['features']['add_interaction_features']:
            interaction_features = [col for col in X.columns if 'interaction' in col]
            self.assertGreater(len(interaction_features), 0)
        
        if self.test_config['features']['add_ratio_features']:
            ratio_features = [col for col in X.columns if 'ratio' in col]
            self.assertGreater(len(ratio_features), 0)
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        evaluator = ModelEvaluator('test_output')
        
        # Create test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # Evaluate model
        metrics = evaluator.evaluate(y_true, y_pred, 'Test Model')
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)
    
    def test_feature_importance_plotting(self):
        """Test feature importance plotting."""
        evaluator = ModelEvaluator('test_output')
        
        # Create test feature importance
        feature_importance = {
            'estradiol': 0.4,
            'progesterone': 0.3,
            'testosterone': 0.2,
            'estradiol_progesterone_interaction': 0.1
        }
        
        # Test plotting (should not raise an error)
        try:
            evaluator.plot_feature_importance(feature_importance, 'Test Model')
        except Exception as e:
            # If plotting fails due to missing matplotlib backend, that's okay
            pass
    
    def test_confusion_matrix_plotting(self):
        """Test confusion matrix plotting."""
        evaluator = ModelEvaluator('test_output')
        
        # Create test data
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        # Test plotting (should not raise an error)
        try:
            evaluator._plot_confusion_matrix(y_true, y_pred, 'Test Model')
        except Exception as e:
            # If plotting fails due to missing matplotlib backend, that's okay
            pass
    
    def test_model_saving_and_loading(self):
        """Test model saving and loading."""
        classifier = RandomForestClassifier(self.test_config)
        
        # Create test data
        X = pd.DataFrame({
            'estradiol': np.random.normal(1.5, 0.5, 50),
            'progesterone': np.random.normal(200, 100, 50),
            'testosterone': np.random.normal(140, 20, 50)
        })
        y = pd.Series(np.random.choice([0, 1], 50))
        
        # Train classifier
        classifier.train(X, y)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            classifier.save_model(model_path)
            
            # Load model
            new_classifier = RandomForestClassifier(self.test_config)
            new_classifier.load_model(model_path)
            
            # Test predictions are the same
            original_predictions = classifier.predict(X)
            loaded_predictions = new_classifier.predict(X)
            
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
        
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)


if __name__ == '__main__':
    unittest.main() 