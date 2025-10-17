"""
Regression test for Bug #4: Keras Single-Output Binary Classification Predictions.

This test reproduces the bug where ALL predictions are 0 when using
Keras binary classification models with single-output architecture (sigmoid).

Bug Scenario:
- Model architecture: Dense(20, relu) → Dense(1, sigmoid)
- Output shape: (n_samples, 1) - single probability per row
- Current code applies argmax() which always returns 0 for shape (n, 1)
- Expected: Mixed 0/1 predictions based on threshold at 0.5

Test Framework: pytest
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestKerasSingleOutputPrediction:
    """Test suite for Keras single-output binary classification predictions."""

    @pytest.fixture
    def ml_engine(self):
        """Initialize ML Engine with default config."""
        config = MLEngineConfig.get_default()
        return MLEngine(config)

    @pytest.fixture
    def existing_model_id(self):
        """
        Use existing trained Keras binary classification model.

        This model has single-output architecture:
        - Layer 1: Dense(20, activation='relu')
        - Layer 2: Dense(1, activation='sigmoid')
        """
        return "model_7715560927_keras_binary_classification_20251014_044444"

    @pytest.fixture
    def test_data(self):
        """
        Create test data matching model's expected features.

        Model expects 20 features: Attribute1 through Attribute20.
        - 13 categorical attributes (encoded strings)
        - 7 numeric attributes

        Uses actual categories from trained model's encoders.
        """
        np.random.seed(42)
        n_samples = 50

        # Category map from trained model's encoders
        CATEGORY_MAP = {
            "Attribute1": ['A11', 'A12', 'A13', 'A14'],
            "Attribute3": ['A30', 'A31', 'A32', 'A33', 'A34'],
            "Attribute4": ['A40', 'A41', 'A410', 'A42', 'A43', 'A44', 'A45', 'A46', 'A49'],
            "Attribute6": ['A61', 'A62', 'A63', 'A64', 'A65'],
            "Attribute7": ['A71', 'A72', 'A73', 'A74', 'A75'],
            "Attribute9": ['A91', 'A92', 'A93', 'A94'],
            "Attribute10": ['A101', 'A102', 'A103'],
            "Attribute12": ['A121', 'A122', 'A123', 'A124'],
            "Attribute14": ['A141', 'A142', 'A143'],
            "Attribute15": ['A151', 'A152', 'A153'],
            "Attribute17": ['A171', 'A172', 'A173', 'A174'],
            "Attribute19": ['A191', 'A192'],
            "Attribute20": ['A201', 'A202'],
        }

        # Create dataframe with mixed categorical and numeric columns
        data = {}
        for i in range(1, 21):
            col_name = f'Attribute{i}'
            if col_name in CATEGORY_MAP:
                # Categorical column - use valid categories
                data[col_name] = np.random.choice(CATEGORY_MAP[col_name], n_samples)
            else:
                # Numeric column - use random integers
                data[col_name] = np.random.randint(1, 100, n_samples)

        return pd.DataFrame(data)

    def test_predictions_not_all_zeros(self, ml_engine, existing_model_id, test_data):
        """
        BUG REPRODUCTION: Verify predictions are NOT all zeros.

        Current bug: argmax() on shape (n, 1) always returns 0
        Expected: Mixed 0 and 1 predictions based on model probabilities
        """
        user_id = 7715560927

        # Run prediction
        result = ml_engine.predict(
            user_id=user_id,
            model_id=existing_model_id,
            data=test_data
        )

        predictions = result['predictions']
        unique_values = set(predictions)

        # BUG ASSERTION: Predictions should contain BOTH 0 and 1
        # Current bug: All predictions are 0
        assert len(unique_values) > 1, (
            f"BUG: All predictions are the same value! "
            f"Expected mixed 0/1, got only: {unique_values}"
        )

        # Verify predictions contain both classes
        assert 0 in unique_values, "Predictions should contain class 0"
        assert 1 in unique_values, "Predictions should contain class 1"

    def test_prediction_statistics_are_realistic(self, ml_engine, existing_model_id, test_data):
        """
        Verify prediction statistics are realistic for binary classification.

        Bug symptom: All zeros → mean=0, std=0, min=0, max=0
        Expected: mean in (0, 1), std > 0, min=0, max=1
        """
        user_id = 7715560927

        # Run prediction
        result = ml_engine.predict(
            user_id=user_id,
            model_id=existing_model_id,
            data=test_data
        )

        predictions = pd.Series(result['predictions'])

        # Calculate statistics
        mean_pred = predictions.mean()
        std_pred = predictions.std()
        min_pred = predictions.min()
        max_pred = predictions.max()

        # Assertions for realistic binary classification statistics
        assert 0 < mean_pred < 1, (
            f"BUG: Mean should be between 0 and 1, got {mean_pred}"
        )
        assert std_pred > 0, (
            f"BUG: Standard deviation should be > 0, got {std_pred}"
        )
        assert min_pred == 0, f"Min should be 0, got {min_pred}"
        assert max_pred == 1, f"Max should be 1, got {max_pred}"

    def test_predictions_are_binary(self, ml_engine, existing_model_id, test_data):
        """
        Verify predictions are strictly binary (0 or 1).

        Should not contain probabilities or other values.
        """
        user_id = 7715560927

        # Run prediction
        result = ml_engine.predict(
            user_id=user_id,
            model_id=existing_model_id,
            data=test_data
        )

        predictions = result['predictions']

        # All predictions must be either 0 or 1
        for pred in predictions:
            assert pred in [0, 1], (
                f"Prediction must be 0 or 1, got {pred}"
            )

    def test_output_shape_is_1d_array(self, ml_engine, existing_model_id, test_data):
        """
        Verify prediction output is 1D array, not 2D.

        Expected shape: (n_samples,)
        Bug symptom: Could be shape (n_samples, 1) if not properly flattened
        """
        user_id = 7715560927

        # Run prediction
        result = ml_engine.predict(
            user_id=user_id,
            model_id=existing_model_id,
            data=test_data
        )

        predictions = result['predictions']

        # Convert to numpy array to check shape
        predictions_array = np.array(predictions)

        # Should be 1D array
        assert predictions_array.ndim == 1, (
            f"Predictions should be 1D array, got {predictions_array.ndim}D"
        )
        assert predictions_array.shape == (len(test_data),), (
            f"Expected shape ({len(test_data)},), got {predictions_array.shape}"
        )
