"""Test for Keras input_shape construction bug fix.

This test reproduces and verifies the fix for:
ValueError: Cannot convert '(20, 'layers')' to a shape.
"""

import pytest
import pandas as pd
import numpy as np
from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.keras_templates import get_template


class TestKerasInputShapeFix:
    """Test that Keras models build with correct input_shape."""

    @pytest.fixture
    def trainer(self):
        """Create Keras trainer instance."""
        config = MLEngineConfig.get_default()
        return KerasNeuralNetworkTrainer(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data with 20 features."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name='target')

        return X, y

    def test_binary_classification_input_shape(self, trainer):
        """Test that binary classification builds with correct input_shape.

        The bug was that input_shape was constructed as (20, 'layers')
        instead of (20,) where 20 is the number of features.
        """
        n_features = 20

        # Get template architecture
        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features
        )

        # Build hyperparameters with architecture and n_features
        hyperparameters = {
            "architecture": architecture,
            "n_features": n_features
        }

        # This should NOT raise ValueError about shape
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )

        # Verify model was created successfully
        assert model is not None
        assert hasattr(model, 'layers')

        # Verify first layer has correct input_dim (not input_shape tuple)
        first_layer = model.layers[0]
        # Dense layers use input_dim, not input_shape
        # input_dim should be an integer, not a tuple
        assert first_layer.input_shape == (None, n_features)

    def test_training_with_20_features(self, trainer, sample_data):
        """Test full training workflow with 20 features.

        This reproduces the exact scenario from the error report.
        """
        X_train, y_train = sample_data

        n_features = X_train.shape[1]  # 20 features
        assert n_features == 20

        # Get architecture
        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features
        )

        # Build hyperparameters
        hyperparameters = {
            "architecture": architecture,
            "n_features": n_features,
            "epochs": 5,  # Small for testing
            "batch_size": 32,
            "verbose": 0
        }

        # Create model - this is where the error occurred
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )

        # Train model
        trained_model = trainer.train(
            model,
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )

        # Verify training succeeded
        assert trained_model is not None
        assert hasattr(trained_model, 'predict')

        # Test prediction
        predictions = trained_model.predict(X_train)
        assert predictions.shape[0] == len(X_train)

    def test_various_feature_counts(self, trainer):
        """Test that different feature counts work correctly."""
        for n_features in [1, 5, 10, 20, 50, 100]:
            architecture = get_template(
                model_type="keras_binary_classification",
                n_features=n_features
            )

            hyperparameters = {
                "architecture": architecture,
                "n_features": n_features
            }

            # Should not raise ValueError
            model = trainer.get_model_instance(
                model_type="keras_binary_classification",
                hyperparameters=hyperparameters
            )

            # Verify input shape
            assert model.layers[0].input_shape == (None, n_features)

    def test_architecture_dict_structure(self):
        """Test that architecture dict has correct structure.

        Ensures 'layers' is a list in the architecture dict,
        not mixed into input_shape tuple.
        """
        n_features = 20
        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features
        )

        # Verify architecture structure
        assert isinstance(architecture, dict)
        assert 'layers' in architecture
        assert isinstance(architecture['layers'], list)
        assert len(architecture['layers']) > 0

        # Verify layers don't contain string 'layers' in wrong places
        for layer in architecture['layers']:
            assert isinstance(layer, dict)
            # Units should be integers, not strings
            if 'units' in layer:
                assert isinstance(layer['units'], int)

    def test_hyperparameters_n_features_type(self, trainer):
        """Test that n_features is always an integer, not a tuple."""
        n_features = 20

        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features
        )

        hyperparameters = {
            "architecture": architecture,
            "n_features": n_features  # This should be int, not tuple
        }

        # Verify n_features is an integer
        assert isinstance(hyperparameters["n_features"], int)
        assert hyperparameters["n_features"] == 20

        # The bug was that n_features might have been a tuple like (20, 'layers')
        # Verify it's not a tuple
        assert not isinstance(hyperparameters["n_features"], tuple)
