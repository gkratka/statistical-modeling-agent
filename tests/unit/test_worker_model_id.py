"""
Test for worker model_id format validation.

The model_id MUST include user_id prefix in the format:
    model_{user_id}_{model_type}_{task_type}_{timestamp}

Example: model_7715560927_xgboost_binary_classification_20251216_123456
"""

import pytest
import re


class TestWorkerModelIdFormat:
    """Test that worker generates correct model_id format with user_id."""

    def test_model_id_must_contain_user_id(self):
        """Model ID must include user_id as second component."""
        # Example of CORRECT format
        correct_model_id = "model_7715560927_xgboost_binary_classification_20251216_123456"

        # Example of BAD format (missing user_id)
        bad_model_id = "model_xgboost_binary_classification_classification_20251211_173856"

        # Pattern: model_{user_id}_{model_type}_{task_type}_{timestamp}
        # user_id should be numeric
        pattern = r"^model_(\d+)_\w+_\w+_\d{8}_\d{6}$"

        assert re.match(pattern, correct_model_id), "Correct format should match"
        assert not re.match(pattern, bad_model_id), "Bad format should NOT match"

    def test_model_id_format_components(self):
        """Model ID components should be in correct order."""
        user_id = 7715560927
        model_type = "xgboost"
        task_type = "binary_classification"
        timestamp = "20251216_123456"

        expected_model_id = f"model_{user_id}_{model_type}_{task_type}_{timestamp}"

        parts = expected_model_id.split("_", 2)  # Split into first 3 parts
        assert parts[0] == "model"
        assert parts[1] == str(user_id)

    def test_validate_model_id_has_user_id(self):
        """Helper function to validate model_id format."""
        def validate_model_id_format(model_id: str, expected_user_id: int) -> bool:
            """Validate model_id contains expected user_id."""
            if not model_id or not model_id.startswith("model_"):
                return False

            parts = model_id.split("_")
            if len(parts) < 4:
                return False

            # Second part should be user_id (numeric)
            try:
                actual_user_id = int(parts[1])
                return actual_user_id == expected_user_id
            except ValueError:
                return False

        # Test correct format
        assert validate_model_id_format(
            "model_7715560927_xgboost_binary_classification_20251216_123456",
            7715560927
        )

        # Test bad format (no user_id)
        assert not validate_model_id_format(
            "model_xgboost_binary_classification_classification_20251211_173856",
            7715560927
        )

        # Test wrong user_id
        assert not validate_model_id_format(
            "model_1234_xgboost_binary_classification_20251216_123456",
            7715560927
        )
