"""
TDD Test for Keras Prediction Output Format.

Bug Report:
When user ran prediction with Keras binary classification model, bot crashed with error:
"TypeError: complex() first argument must be a string or a number, not 'list'"

This occurred at prediction_handlers.py:1065 when trying to calculate statistics:
    'mean': float(pd.Series(predictions).mean())

Root Cause:
Keras binary classification models return 2D probability arrays from .predict():
    [[0.7, 0.3],  # Row 1: P(class=0)=0.7, P(class=1)=0.3
     [0.8, 0.2],  # Row 2: P(class=0)=0.8, P(class=1)=0.2
     ...]

When converted to .tolist(), these become nested lists: [[0.7, 0.3], [0.8, 0.2], ...]
pandas.Series cannot calculate mean() on nested lists, causing TypeError.

Fix:
In ml_engine.py predict(), convert Keras probabilities to class labels using argmax:
    predictions = np.argmax(predictions, axis=1)  # → [0, 0, 1, 0, ...]

This test verifies the fix.
"""

import pytest
from pathlib import Path
import pandas as pd
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestKerasPredictionOutputFormat:
    """Test that Keras models return 1D class labels, not 2D probability arrays."""

    def test_keras_binary_classification_returns_1d_labels(self):
        """
        REGRESSION TEST: Verify Keras predictions return 1D class labels, not 2D probabilities.

        Bug Scenario:
        1. User ran /predict with German credit dataset (200 rows × 20 features)
        2. Selected Keras binary classification model
        3. Bot executed prediction
        4. ml_engine.predict() returned nested lists: [[0.7, 0.3], [0.8, 0.2], ...]
        5. prediction_handlers.py tried: pd.Series([[0.7, 0.3], ...]).mean()
        6. Error: "TypeError: complex() argument cannot be 'list'"
        7. Prediction failed

        Expected Behavior:
        Keras binary classification predictions should return:
        - 1D array of class labels: [0, 1, 0, 1, ...]
        - NOT 2D probability arrays: [[0.7, 0.3], [0.8, 0.2], ...]
        - Statistics (mean, std, min, max) should calculate successfully
        """
        # Find a Keras binary classification model
        models_dir = Path("models/user_7715560927")

        if not models_dir.exists():
            pytest.skip("No user models found for testing")

        # Find Keras binary classification model
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            pytest.skip("No model directories found")

        keras_model = None
        for model_dir in model_dirs:
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if (metadata.get('model_format') == 'keras' and
                    metadata.get('task_type') == 'classification'):
                    keras_model = model_dir.name
                    break

        if not keras_model:
            pytest.skip("No Keras classification model found for testing")

        user_id = 7715560927
        model_id = keras_model

        # Initialize ML Engine
        config = MLEngineConfig.get_default()
        ml_engine = MLEngine(config)

        # Get model metadata to know expected features
        model_info = ml_engine.get_model_info(user_id, model_id)
        feature_columns = model_info.get('feature_columns', [])

        if not feature_columns:
            pytest.skip("Model has no feature_columns in metadata")

        # Create test data matching model's training features
        n_samples = 10
        test_data = pd.DataFrame({
            col: [0.5] * n_samples for col in feature_columns
        })

        # Run prediction
        result = ml_engine.predict(
            user_id=user_id,
            model_id=model_id,
            data=test_data
        )

        # Verify result structure
        assert 'predictions' in result, "Result must contain 'predictions' key"
        predictions = result['predictions']

        # CRITICAL ASSERTION 1: predictions must be a list
        assert isinstance(predictions, list), \
            f"predictions must be a list, got {type(predictions)}"

        # CRITICAL ASSERTION 2: predictions must be 1D (list of scalars, not nested lists)
        assert len(predictions) > 0, "predictions should not be empty"

        first_prediction = predictions[0]
        assert not isinstance(first_prediction, list), \
            f"BUG: Keras predictions are 2D (nested lists)! " \
            f"First prediction: {first_prediction} is a {type(first_prediction)}. " \
            f"Expected scalar (int or float), not list. " \
            f"Fix: Convert probabilities to class labels using argmax."

        # CRITICAL ASSERTION 3: All predictions should be scalars (int or float)
        for i, pred in enumerate(predictions):
            assert isinstance(pred, (int, float, type(None))), \
                f"BUG: Prediction {i} is {type(pred)}, expected int/float. " \
                f"Value: {pred}. This will cause pandas mean() to fail!"

        # CRITICAL ASSERTION 4: For binary classification, predictions should be 0 or 1
        for i, pred in enumerate(predictions):
            if pred is not None:
                assert pred in [0, 1] or (isinstance(pred, (int, float)) and 0 <= pred <= 1), \
                    f"Binary classification prediction {i} should be 0, 1, or probability in [0,1], got {pred}"

        # CRITICAL ASSERTION 5: Statistics should be calculable (this is what failed in production)
        try:
            import pandas as pd
            series = pd.Series(predictions)
            mean_val = float(series.mean())
            std_val = float(series.std())
            min_val = float(series.min())
            max_val = float(series.max())

            # If we get here without error, the fix is working
            assert True, "Statistics calculated successfully!"

        except TypeError as e:
            if "complex() first argument" in str(e) or "list" in str(e):
                pytest.fail(
                    f"BUG STILL EXISTS: Cannot calculate statistics on predictions! "
                    f"Error: {e}. "
                    f"Predictions structure: {predictions[:3]}... "
                    f"This means Keras is still returning 2D arrays."
                )
            else:
                raise

    def test_prediction_output_compatible_with_pandas_stats(self):
        """
        Test that predictions can be used with pandas statistics functions.

        This directly tests the failure scenario from production.
        """
        models_dir = Path("models/user_7715560927")

        if not models_dir.exists():
            pytest.skip("No user models found for testing")

        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            pytest.skip("No model directories found")

        # Find any Keras model
        keras_model = None
        for model_dir in model_dirs:
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if metadata.get('model_format') == 'keras':
                    keras_model = model_dir.name
                    break

        if not keras_model:
            pytest.skip("No Keras model found")

        user_id = 7715560927
        model_id = keras_model

        # Initialize ML Engine
        config = MLEngineConfig.get_default()
        ml_engine = MLEngine(config)

        # Get features
        model_info = ml_engine.get_model_info(user_id, model_id)
        feature_columns = model_info.get('feature_columns', [])

        if not feature_columns:
            pytest.skip("No features")

        # Create test data
        test_data = pd.DataFrame({
            col: [0.5] * 5 for col in feature_columns
        })

        # Run prediction
        result = ml_engine.predict(user_id=user_id, model_id=model_id, data=test_data)
        predictions = result['predictions']

        # This is the exact code that failed in production (line 1065)
        try:
            statistics = {
                'mean': float(pd.Series(predictions).mean()),
                'std': float(pd.Series(predictions).std()),
                'min': float(pd.Series(predictions).min()),
                'max': float(pd.Series(predictions).max()),
                'median': float(pd.Series(predictions).median())
            }

            # Verify statistics are valid numbers
            for key, value in statistics.items():
                assert isinstance(value, (int, float)), \
                    f"{key} should be numeric, got {type(value)}"
                assert not pd.isna(value) or key == 'std', \
                    f"{key} should not be NaN (except std can be NaN for single value)"

        except TypeError as e:
            pytest.fail(
                f"BUG: Cannot calculate statistics on predictions! "
                f"This is the exact production error. "
                f"Error: {e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
