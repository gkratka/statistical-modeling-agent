"""
Simple integration tests for Keras-Telegram workflow.

Tests core functionality without complex mocking.
"""

import pytest
from src.bot.workflow_handlers import is_keras_model
from src.engines.trainers.keras_templates import get_template


class TestIsKerasModelHelper:
    """Test is_keras_model() helper function."""

    def test_keras_models_detected(self):
        """Test Keras models are correctly identified."""
        assert is_keras_model('keras_binary_classification') == True
        assert is_keras_model('keras_multiclass_classification') == True
        assert is_keras_model('keras_regression') == True

    def test_sklearn_models_not_detected(self):
        """Test sklearn models are not identified as Keras."""
        assert is_keras_model('linear') == False
        assert is_keras_model('random_forest') == False
        assert is_keras_model('neural_network') == False
        assert is_keras_model('auto') == False


class TestKerasTemplates:
    """Test Keras template integration."""

    def test_binary_template_structure(self):
        """Test binary classification template has correct structure."""
        template = get_template('keras_binary_classification', n_features=14)

        assert 'layers' in template
        assert 'compile' in template
        assert len(template['layers']) == 2
        assert template['layers'][-1]['activation'] == 'sigmoid'
        assert template['compile']['loss'] == 'binary_crossentropy'

    def test_multiclass_template_structure(self):
        """Test multiclass template has correct structure."""
        template = get_template('keras_multiclass_classification', n_features=10, n_classes=5)

        assert template['layers'][-1]['units'] == 5
        assert template['layers'][-1]['activation'] == 'softmax'
        assert template['compile']['loss'] == 'categorical_crossentropy'

    def test_regression_template_structure(self):
        """Test regression template has correct structure."""
        template = get_template('keras_regression', n_features=8)

        assert template['layers'][-1]['activation'] == 'linear'
        assert template['compile']['loss'] == 'mse'
