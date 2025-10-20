"""
Test model naming validation and default name generation.

Tests that generated default names pass validation rules, ensuring
consistency between _generate_default_name and _validate_model_name.

This test suite specifically addresses the issue where generated names
with comma characters fail validation.
"""

import pytest
from datetime import datetime
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestModelNamingValidation:
    """Test default name generation and validation consistency."""

    @pytest.fixture
    def ml_engine(self):
        """Create MLEngine instance for testing."""
        config = MLEngineConfig.get_default()
        return MLEngine(config)

    def test_generate_default_name_passes_validation(self, ml_engine):
        """Test that generated default names pass validation (main regression test)."""
        # Test data with various model types
        test_cases = [
            ("xgboost_binary_classification", "classification", "2025-10-20T05:37:51Z"),
            ("keras_binary_classification", "classification", "2025-01-14T21:44:00Z"),
            ("keras_multiclass_classification", "classification", "2025-06-15T10:30:00Z"),
            ("keras_regression", "regression", "2025-03-20T14:22:15Z"),
            ("random_forest", "classification", "2025-05-10T08:15:30Z"),
            ("logistic", "classification", "2025-12-01T16:45:22Z"),
            ("linear", "regression", "2025-08-19T11:30:45Z"),
        ]

        for model_type, task_type, created_at in test_cases:
            # Generate default name
            generated_name = ml_engine._generate_default_name(
                model_type=model_type,
                task_type=task_type,
                created_at=created_at
            )

            # Validate the generated name
            is_valid, error_msg = ml_engine._validate_model_name(generated_name)

            # Assert validation passes
            assert is_valid is True, (
                f"Generated name '{generated_name}' for model type '{model_type}' "
                f"failed validation: {error_msg}"
            )
            assert error_msg is None

    def test_xgboost_default_name_format(self, ml_engine):
        """Test XGBoost models get correct display name format."""
        test_cases = [
            ("xgboost_binary_classification", "XGBoost Binary Classification"),
            ("xgboost_multiclass_classification", "XGBoost Multiclass Classification"),
            ("xgboost_regression", "XGBoost Regression"),
        ]

        created_at = "2025-10-20T05:37:51Z"

        for model_type, expected_prefix in test_cases:
            generated_name = ml_engine._generate_default_name(
                model_type=model_type,
                task_type="classification",
                created_at=created_at
            )

            # Should start with expected prefix
            assert generated_name.startswith(expected_prefix), (
                f"Expected name to start with '{expected_prefix}', "
                f"got '{generated_name}'"
            )

            # Should contain date
            assert "Oct" in generated_name
            assert "20" in generated_name
            assert "2025" in generated_name

            # Should be valid
            is_valid, _ = ml_engine._validate_model_name(generated_name)
            assert is_valid is True

    def test_keras_default_name_format(self, ml_engine):
        """Test Keras models get simplified display names."""
        test_cases = [
            ("keras_binary_classification", "Binary Classification"),
            ("keras_multiclass_classification", "Multiclass Classification"),
            ("keras_regression", "Neural Network Regression"),
        ]

        created_at = "2025-01-14T21:44:00Z"

        for model_type, expected_prefix in test_cases:
            generated_name = ml_engine._generate_default_name(
                model_type=model_type,
                task_type="classification" if "class" in model_type else "regression",
                created_at=created_at
            )

            # Should start with simplified name
            assert generated_name.startswith(expected_prefix), (
                f"Expected name to start with '{expected_prefix}', "
                f"got '{generated_name}'"
            )

            # Should be valid
            is_valid, _ = ml_engine._validate_model_name(generated_name)
            assert is_valid is True

    def test_validate_model_name_rejects_comma(self, ml_engine):
        """Test that validation correctly rejects names with commas."""
        # These should all fail validation
        invalid_names = [
            "Model Name, With Comma",
            "XGBoost Binary Classification - Oct 20, 2025",  # Original failing case
            "Binary Classification - Jan 14, 2025",
            "Test, Model",
        ]

        for name in invalid_names:
            is_valid, error_msg = ml_engine._validate_model_name(name)
            assert is_valid is False, f"Name '{name}' should be invalid (contains comma)"
            assert error_msg is not None
            assert "letters, numbers, spaces, hyphens, and underscores" in error_msg

    def test_validate_model_name_accepts_valid_chars(self, ml_engine):
        """Test that validation accepts valid character combinations."""
        # These should all pass validation
        valid_names = [
            "Simple Model",
            "XGBoost Binary Classification - Oct 20 2025",  # Fixed format (no comma)
            "Binary Classification - Jan 14 2025",
            "Model_Name_With_Underscores",
            "Model-Name-With-Hyphens",
            "ModelNameCamelCase123",
            "model name lowercase",
            "Random Forest - May 15 2025",
        ]

        for name in valid_names:
            is_valid, error_msg = ml_engine._validate_model_name(name)
            assert is_valid is True, (
                f"Name '{name}' should be valid but got error: {error_msg}"
            )
            assert error_msg is None

    def test_validate_model_name_length_constraints(self, ml_engine):
        """Test name length validation."""
        # Too short
        is_valid, error_msg = ml_engine._validate_model_name("ab")
        assert is_valid is False
        assert "at least 3 characters" in error_msg

        # Minimum valid length
        is_valid, error_msg = ml_engine._validate_model_name("abc")
        assert is_valid is True
        assert error_msg is None

        # Too long
        long_name = "a" * 101
        is_valid, error_msg = ml_engine._validate_model_name(long_name)
        assert is_valid is False
        assert "less than 100 characters" in error_msg

        # Maximum valid length
        max_name = "a" * 100
        is_valid, error_msg = ml_engine._validate_model_name(max_name)
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_special_chars(self, ml_engine):
        """Test that special characters are rejected."""
        invalid_chars = [
            "Model@Name",
            "Model#Name",
            "Model$Name",
            "Model%Name",
            "Model!Name",
            "Model.Name",
            "Model/Name",
            "Model\\Name",
            "Model:Name",
            "Model;Name",
            "Model'Name",
            'Model"Name',
        ]

        for name in invalid_chars:
            is_valid, error_msg = ml_engine._validate_model_name(name)
            assert is_valid is False, f"Name '{name}' should be invalid"
            assert error_msg is not None

    def test_date_format_has_no_comma(self, ml_engine):
        """Test that date format in generated names doesn't contain comma."""
        # Generate several names with different dates
        test_dates = [
            "2025-01-14T21:44:00Z",
            "2025-06-15T10:30:00Z",
            "2025-10-20T05:37:51Z",
            "2025-12-31T23:59:59Z",
        ]

        for date_str in test_dates:
            generated_name = ml_engine._generate_default_name(
                model_type="xgboost_binary_classification",
                task_type="classification",
                created_at=date_str
            )

            # Should not contain comma
            assert "," not in generated_name, (
                f"Generated name '{generated_name}' contains comma character"
            )

            # Should still contain date components
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month_abbr = dt.strftime("%b")
            day = str(dt.day)
            year = str(dt.year)

            assert month_abbr in generated_name
            assert day in generated_name
            assert year in generated_name

    def test_empty_and_whitespace_names(self, ml_engine):
        """Test validation rejects empty and whitespace-only names."""
        invalid_names = [
            "",
            "   ",
            "\t",
            "\n",
            "  \t  \n  ",
        ]

        for name in invalid_names:
            is_valid, error_msg = ml_engine._validate_model_name(name)
            assert is_valid is False
            assert "cannot be empty" in error_msg
