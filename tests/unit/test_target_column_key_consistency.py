"""
TDD Test for Target Column Key Consistency in get_model_summary().

Bug Report:
User saw "Your model predicts: None" in prediction workflow instead of "Your model predicts: class".
This occurred because get_model_summary() returns the key "target" but prediction_handlers.py
expects "target_column".

Root Cause:
Line 472 in model_manager.py returns "target": metadata.get("target_column")
But prediction_handlers.py line 691 does: model_info.get('target_column')

This causes model_info.get('target_column') to return None, resulting in:
- "Your model predicts: None" in UI
- "None_predicted" as default column name
- User confusion

Fix:
Change line 472 in model_manager.py from:
    "target": metadata.get("target_column")
To:
    "target_column": metadata.get("target_column")

This test verifies the fix.
"""

import pytest
from pathlib import Path
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestTargetColumnKeyConsistency:
    """Test that get_model_summary() returns standard key names for target column."""

    def test_get_model_summary_uses_target_column_key(self):
        """
        REGRESSION TEST: Verify get_model_summary() returns 'target_column' not 'target'.

        Bug Scenario:
        1. User trained Keras binary classification models with target='class'
        2. User ran /predict and selected model
        3. Bot displayed "Your model predicts: None" instead of "class"
        4. Bot suggested "None_predicted" as column name instead of "class_predicted"
        5. Root cause: prediction_handlers.py expected 'target_column' key
        6. But get_model_summary() returned 'target' key
        7. Result: model_info.get('target_column') → None

        Expected Behavior:
        get_model_summary() must return 'target_column' to match:
        - metadata.json structure
        - prediction handler expectations
        - consistent naming with 'feature_columns'
        """
        # Find a real trained model from user's models directory
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

        # CRITICAL ASSERTION 1: Must have 'target_column' key (not 'target')
        assert 'target_column' in model_info, \
            f"BUG: get_model_summary() must return 'target_column' key, got keys: {list(model_info.keys())}"

        # CRITICAL ASSERTION 2: Must NOT have 'target' key (inconsistent naming)
        assert 'target' not in model_info, \
            "BUG: get_model_summary() should use 'target_column' not 'target' for consistency"

        # Verify target_column value matches metadata
        import json
        metadata_path = models_dir / model_id / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        expected_target = metadata.get('target_column')
        assert model_info.get('target_column') == expected_target, \
            f"target_column value mismatch: got {model_info.get('target_column')}, expected {expected_target}"

        # Verify target_column is not None or empty
        assert model_info.get('target_column') is not None, \
            "target_column should not be None"
        assert model_info.get('target_column') != "", \
            "target_column should not be empty string"

    def test_model_summary_keys_match_metadata_target(self):
        """
        Test that get_model_summary() target_column matches metadata.json.

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

        # CRITICAL: target_column must match metadata
        assert model_info.get('target_column') == metadata.get('target_column'), \
            f"get_model_summary() target_column must match metadata.json: " \
            f"got '{model_info.get('target_column')}', expected '{metadata.get('target_column')}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
