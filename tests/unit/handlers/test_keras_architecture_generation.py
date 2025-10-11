"""
Unit tests for Keras architecture generation in ml_training_local_path handlers.

Tests the automatic generation of architecture specifications using the template
system before calling ML Engine training.
"""

import pytest
from src.engines.trainers.keras_templates import get_template


class TestKerasArchitectureGeneration:
    """Test automatic Keras architecture generation for training."""

    def test_architecture_generated_for_binary_classification(self):
        """Test that binary classification template is generated correctly."""
        model_type = "keras_binary_classification"
        n_features = 2
        kernel_initializer = "glorot_uniform"

        architecture = get_template(
            model_type=model_type,
            n_features=n_features,
            kernel_initializer=kernel_initializer
        )

        # Verify architecture structure
        assert 'layers' in architecture
        assert 'compile' in architecture

        # Verify layers
        assert len(architecture['layers']) == 2
        assert architecture['layers'][0]['type'] == 'Dense'
        assert architecture['layers'][0]['units'] == n_features
        assert architecture['layers'][0]['activation'] == 'relu'
        assert architecture['layers'][0]['kernel_initializer'] == kernel_initializer

        # Verify output layer
        assert architecture['layers'][1]['units'] == 1
        assert architecture['layers'][1]['activation'] == 'sigmoid'

        # Verify compile config
        assert architecture['compile']['loss'] == 'binary_crossentropy'
        assert architecture['compile']['optimizer'] == 'adam'
        assert 'accuracy' in architecture['compile']['metrics']

    def test_architecture_generated_for_regression(self):
        """Test that regression template is generated correctly."""
        model_type = "keras_regression"
        n_features = 5
        kernel_initializer = "he_normal"

        architecture = get_template(
            model_type=model_type,
            n_features=n_features,
            kernel_initializer=kernel_initializer
        )

        # Verify architecture structure
        assert 'layers' in architecture
        assert 'compile' in architecture

        # Verify layers
        assert len(architecture['layers']) == 2
        assert architecture['layers'][0]['units'] == n_features

        # Verify output layer for regression
        assert architecture['layers'][1]['units'] == 1
        assert architecture['layers'][1]['activation'] == 'linear'

        # Verify compile config for regression
        assert architecture['compile']['loss'] == 'mse'
        assert 'mae' in architecture['compile']['metrics']

    def test_architecture_includes_kernel_initializer(self):
        """Test that kernel initializer from config is applied to all layers."""
        model_type = "keras_binary_classification"
        n_features = 3
        kernel_initializer = "random_normal"

        architecture = get_template(
            model_type=model_type,
            n_features=n_features,
            kernel_initializer=kernel_initializer
        )

        # Verify all layers have the specified initializer
        for layer in architecture['layers']:
            if layer['type'] == 'Dense':
                assert layer['kernel_initializer'] == kernel_initializer

    def test_n_features_calculation(self):
        """Test that n_features is correctly calculated from feature columns."""
        # Simulate feature columns from session
        feature_columns = ['sqft', 'bedrooms', 'bathrooms', 'age', 'garage']
        n_features = len(feature_columns)

        assert n_features == 5

        # Verify it works with the template
        architecture = get_template(
            model_type="keras_regression",
            n_features=n_features,
            kernel_initializer="glorot_uniform"
        )

        assert architecture['layers'][0]['units'] == 5

    def test_complete_hyperparameters_structure(self):
        """Test that complete hyperparameters dict has all required keys."""
        # Simulate the config dict from Keras workflow
        config = {
            'epochs': 100,
            'batch_size': 32,
            'kernel_initializer': 'glorot_uniform',
            'verbose': 1,
            'validation_split': 0.2
        }

        # Simulate architecture generation
        n_features = 2
        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features,
            kernel_initializer=config['kernel_initializer']
        )

        # Build complete hyperparameters dict
        hyperparameters = {
            **config,
            'architecture': architecture,
            'n_features': n_features
        }

        # Verify all required keys are present
        assert 'architecture' in hyperparameters
        assert 'epochs' in hyperparameters
        assert 'batch_size' in hyperparameters
        assert 'kernel_initializer' in hyperparameters
        assert 'verbose' in hyperparameters
        assert 'validation_split' in hyperparameters
        assert 'n_features' in hyperparameters

        # Verify architecture structure
        assert 'layers' in hyperparameters['architecture']
        assert 'compile' in hyperparameters['architecture']

        # Verify values
        assert hyperparameters['epochs'] == 100
        assert hyperparameters['batch_size'] == 32
        assert hyperparameters['n_features'] == 2

    def test_multiclass_classification_template(self):
        """Test multiclass classification template generation."""
        model_type = "keras_multiclass_classification"
        n_features = 4
        n_classes = 3
        kernel_initializer = "glorot_uniform"

        architecture = get_template(
            model_type=model_type,
            n_features=n_features,
            n_classes=n_classes,
            kernel_initializer=kernel_initializer
        )

        # Verify multiclass-specific configuration
        assert architecture['layers'][1]['units'] == n_classes
        assert architecture['layers'][1]['activation'] == 'softmax'
        assert architecture['compile']['loss'] == 'categorical_crossentropy'

    def test_template_with_different_initializers(self):
        """Test that different initializers are correctly applied."""
        initializers = [
            'glorot_uniform',
            'glorot_normal',
            'he_uniform',
            'he_normal',
            'random_normal',
            'random_uniform'
        ]

        for init in initializers:
            architecture = get_template(
                model_type="keras_binary_classification",
                n_features=3,
                kernel_initializer=init
            )

            # Verify initializer is applied
            assert architecture['layers'][0]['kernel_initializer'] == init
            assert architecture['layers'][1]['kernel_initializer'] == init
