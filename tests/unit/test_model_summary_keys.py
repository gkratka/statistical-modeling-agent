"""
TDD Test for Model Summary Key Consistency.

Bug Report:
When user selected 20 features for prediction, bot showed "Feature Mismatch: Model requires 0 features".
This occurred because get_model_summary() returns the key "features" but prediction_handlers.py
expects "feature_columns".

Root Cause:
Line 471 in model_manager.py uses "features" as the key name, creating inconsistency
with the rest of the codebase which uses "feature_columns".

Fix:
Change line 471 from:
    "features": metadata.get("feature_columns", [])
To:
    "feature_columns": metadata.get("feature_columns", [])

This test verifies the fix.
"""

import pytest
from pathlib import Path
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestModelSummaryKeyConsistency:
    """Test that get_model_summary() returns standard key names."""

    def test_get_model_summary_uses_feature_columns_key(self):
        """
        REGRESSION TEST: Verify get_model_summary() returns 'feature_columns' not 'features'.

        Bug Scenario:
        1. User trained 20+ Keras models with 20 features
        2. User ran /predict and selected 20 features
        3. Bot loaded models but get_model_summary() returned 'features' key
        4. prediction_handlers.py expected 'feature_columns' key
        5. Key mismatch → model_info.get('feature_columns', []) returned []
        6. Error message: "Model requires 0 features, you selected 20"

        Expected Behavior:
        get_model_summary() must return 'feature_columns' to match:
        - metadata.json structure
        - list_models() return format
        - prediction handler expectations
        """
        # Find a real trained model from the user's models directory
        models_dir = Path("models/user_7715560927")

        if not models_dir.exists():
            pytest.skip("No user models found for testing")

        # Find first model directory
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            pytest.skip("No model directories found")

        model_id = model_dirs[0].name
        user_id = 7715560927

        # Initialize ML Engine
        config = MLEngineConfig.get_default()
        ml_engine = MLEngine(config)

        # Get model summary
        model_info = ml_engine.get_model_info(user_id, model_id)

        # CRITICAL ASSERTION 1: Must have 'feature_columns' key (not 'features')
        assert 'feature_columns' in model_info, \
            f"BUG: get_model_summary() must return 'feature_columns' key, got keys: {list(model_info.keys())}"

        # CRITICAL ASSERTION 2: Must NOT have 'features' key (inconsistent naming)
        assert 'features' not in model_info, \
            "BUG: get_model_summary() should use 'feature_columns' not 'features' for consistency"

        # Verify feature_columns is a list
        assert isinstance(model_info['feature_columns'], list), \
            "'feature_columns' value must be a list"

        # Verify n_features matches length of feature_columns
        assert model_info['n_features'] == len(model_info['feature_columns']), \
            "n_features should match length of feature_columns list"

    def test_model_summary_keys_match_metadata_structure(self):
        """
        Test that get_model_summary() keys match what's in metadata.json.

        This ensures consistency between:
        - Raw metadata.json files
        - list_models() output
        - get_model_info() / get_model_summary() output
        """
        models_dir = Path("models/user_7715560927")

        if not models_dir.exists():
            pytest.skip("No user models found for testing")

        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            pytest.skip("No model directories found")

        model_id = model_dirs[0].name
        user_id = 7715560927

        # Initialize ML Engine
        config = MLEngineConfig.get_default()
        ml_engine = MLEngine(config)

        # Get model summary
        model_info = ml_engine.get_model_info(user_id, model_id)

        # Read metadata.json directly
        import json
        metadata_path = models_dir / model_id / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Verify critical keys match
        assert model_info.get('model_type') == metadata.get('model_type')
        assert model_info.get('task_type') == metadata.get('task_type')

        # CRITICAL: feature_columns must match
        assert model_info.get('feature_columns') == metadata.get('feature_columns'), \
            "get_model_summary() feature_columns must match metadata.json"

        # Verify n_features is consistent
        expected_n_features = len(metadata.get('feature_columns', []))
        assert model_info.get('n_features') == expected_n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
