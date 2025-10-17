"""
Unit tests for ML Engine Model Naming Feature.

Tests model naming functionality including validation, generation,
and retrieval of custom and default model names.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, call
import logging

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import ValidationError, ModelNotFoundError


@pytest.fixture
def ml_config(tmp_path):
    """Create ML configuration for testing."""
    return MLEngineConfig(
        models_dir=tmp_path / "models",
        max_models_per_user=50,
        max_model_size_mb=100,
        max_training_time=60,
        max_memory_mb=2048,
        min_training_samples=10,
        default_test_size=0.2,
        default_cv_folds=5,
        default_missing_strategy="mean",
        default_scaling="standard",
        default_hyperparameters={},
        hyperparameter_ranges={}
    )


@pytest.fixture
def ml_engine(ml_config):
    """Create ML Engine instance for testing."""
    return MLEngine(ml_config)


@pytest.fixture
def sample_model_metadata():
    """Create sample model metadata for testing."""
    return {
        "model_id": "model_12345_linear_20251014_123456",
        "model_type": "linear",
        "task_type": "regression",
        "target_column": "price",
        "feature_columns": ["sqft", "bedrooms", "bathrooms"],
        "created_at": "2025-01-14T21:44:00Z",
        "metrics": {"mse": 0.15, "r2": 0.85},
        "custom_name": None
    }


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary models directory with sample models."""
    models_dir = tmp_path / "models" / "user_12345"
    models_dir.mkdir(parents=True)
    return models_dir


class TestValidateModelName:
    """Test _validate_model_name() method."""

    def test_validate_model_name_valid_basic(self, ml_engine):
        """Test validation passes for valid basic name."""
        is_valid, error_msg = ml_engine._validate_model_name("My Model")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_valid_with_numbers(self, ml_engine):
        """Test validation passes for name with numbers."""
        is_valid, error_msg = ml_engine._validate_model_name("Model123")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_valid_with_hyphen(self, ml_engine):
        """Test validation passes for name with hyphens."""
        is_valid, error_msg = ml_engine._validate_model_name("My-Model-Name")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_valid_with_underscore(self, ml_engine):
        """Test validation passes for name with underscores."""
        is_valid, error_msg = ml_engine._validate_model_name("My_Model_Name")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_valid_mixed(self, ml_engine):
        """Test validation passes for name with mixed valid characters."""
        is_valid, error_msg = ml_engine._validate_model_name("Housing Price Model-v2_Final")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_too_short(self, ml_engine):
        """Test validation fails for name too short (<3 chars)."""
        is_valid, error_msg = ml_engine._validate_model_name("ab")
        assert is_valid is False
        assert "at least 3 characters" in error_msg

    def test_validate_model_name_minimum_length(self, ml_engine):
        """Test validation passes for minimum length (3 chars)."""
        is_valid, error_msg = ml_engine._validate_model_name("abc")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_too_long(self, ml_engine):
        """Test validation fails for name too long (>100 chars)."""
        long_name = "a" * 101
        is_valid, error_msg = ml_engine._validate_model_name(long_name)
        assert is_valid is False
        assert "less than 100 characters" in error_msg

    def test_validate_model_name_maximum_length(self, ml_engine):
        """Test validation passes for maximum length (100 chars)."""
        max_name = "a" * 100
        is_valid, error_msg = ml_engine._validate_model_name(max_name)
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_empty_string(self, ml_engine):
        """Test validation fails for empty string."""
        is_valid, error_msg = ml_engine._validate_model_name("")
        assert is_valid is False
        assert "cannot be empty" in error_msg.lower()

    def test_validate_model_name_whitespace_only(self, ml_engine):
        """Test validation fails for whitespace-only string."""
        is_valid, error_msg = ml_engine._validate_model_name("   ")
        assert is_valid is False
        assert "cannot be empty" in error_msg.lower()

    def test_validate_model_name_invalid_forward_slash(self, ml_engine):
        """Test validation fails for name with forward slash."""
        is_valid, error_msg = ml_engine._validate_model_name("model/test")
        assert is_valid is False
        assert "can only contain" in error_msg.lower()

    def test_validate_model_name_invalid_backslash(self, ml_engine):
        """Test validation fails for name with backslash."""
        is_valid, error_msg = ml_engine._validate_model_name("model\\test")
        assert is_valid is False
        assert "can only contain" in error_msg.lower()

    def test_validate_model_name_invalid_special_chars(self, ml_engine):
        """Test validation fails for name with various special characters."""
        invalid_names = [
            "model:test",      # colon
            "model;test",      # semicolon
            "model<test",      # less than
            "model>test",      # greater than
            "model|test",      # pipe
            "model?test",      # question mark
            "model*test",      # asterisk
            "model@test",      # at symbol
            "model!test",      # exclamation
            "model#test",      # hash
            "model$test",      # dollar
            "model%test",      # percent
            "model^test",      # caret
            "model&test",      # ampersand
            "model(test)",     # parentheses
            "model[test]",     # brackets
            "model{test}",     # braces
            'model"test',      # double quote
            "model'test",      # single quote
        ]

        for invalid_name in invalid_names:
            is_valid, error_msg = ml_engine._validate_model_name(invalid_name)
            assert is_valid is False, f"Should fail for: {invalid_name}"
            assert "can only contain" in error_msg.lower()

    def test_validate_model_name_with_leading_whitespace(self, ml_engine):
        """Test validation handles leading whitespace (should be stripped)."""
        is_valid, error_msg = ml_engine._validate_model_name("  Valid Model")
        assert is_valid is True
        assert error_msg is None

    def test_validate_model_name_with_trailing_whitespace(self, ml_engine):
        """Test validation handles trailing whitespace (should be stripped)."""
        is_valid, error_msg = ml_engine._validate_model_name("Valid Model  ")
        assert is_valid is True
        assert error_msg is None


class TestGenerateDefaultName:
    """Test _generate_default_name() method."""

    def test_generate_default_name_basic(self, ml_engine):
        """Test default name generation with basic model type."""
        name = ml_engine._generate_default_name(
            model_type="linear",
            task_type="regression",
            created_at="2025-01-14T21:44:00Z"
        )
        assert name == "Linear Regression - Jan 14, 2025"

    def test_generate_default_name_keras_binary(self, ml_engine):
        """Test default name generation for Keras binary classification."""
        name = ml_engine._generate_default_name(
            model_type="keras_binary_classification",
            task_type="classification",
            created_at="2025-01-10T15:30:00Z"
        )
        assert name == "Binary Classification - Jan 10, 2025"

    def test_generate_default_name_keras_multiclass(self, ml_engine):
        """Test default name generation for Keras multiclass classification."""
        name = ml_engine._generate_default_name(
            model_type="keras_multiclass_classification",
            task_type="classification",
            created_at="2025-02-20T10:00:00Z"
        )
        assert name == "Multiclass Classification - Feb 20, 2025"

    def test_generate_default_name_keras_regression(self, ml_engine):
        """Test default name generation for Keras regression."""
        name = ml_engine._generate_default_name(
            model_type="keras_regression",
            task_type="regression",
            created_at="2025-03-15T08:30:00Z"
        )
        assert name == "Neural Network Regression - Mar 15, 2025"

    def test_generate_default_name_random_forest(self, ml_engine):
        """Test default name generation for Random Forest."""
        name = ml_engine._generate_default_name(
            model_type="random_forest",
            task_type="classification",
            created_at="2025-04-05T12:00:00Z"
        )
        assert name == "Random Forest - Apr 05, 2025"

    def test_generate_default_name_logistic(self, ml_engine):
        """Test default name generation for Logistic Regression."""
        name = ml_engine._generate_default_name(
            model_type="logistic",
            task_type="classification",
            created_at="2025-05-25T14:15:00Z"
        )
        assert name == "Logistic Regression - May 25, 2025"

    def test_generate_default_name_unsimplified(self, ml_engine):
        """Test default name generation for model type without simplification."""
        name = ml_engine._generate_default_name(
            model_type="gradient_boosting",
            task_type="regression",
            created_at="2025-06-01T09:00:00Z"
        )
        assert name == "Gradient Boosting - Jun 01, 2025"

    def test_generate_default_name_date_formatting(self, ml_engine):
        """Test date formatting in default name generation."""
        name = ml_engine._generate_default_name(
            model_type="linear",
            task_type="regression",
            created_at="2025-12-31T23:59:59Z"
        )
        assert "Dec 31, 2025" in name

    def test_generate_default_name_with_timezone(self, ml_engine):
        """Test default name generation with timezone-aware timestamp."""
        name = ml_engine._generate_default_name(
            model_type="linear",
            task_type="regression",
            created_at="2025-07-04T18:00:00+00:00"
        )
        assert name == "Linear Regression - Jul 04, 2025"


class TestSetModelName:
    """Test set_model_name() method."""

    def test_set_model_name_success(self, ml_engine):
        """Test successfully setting a custom model name."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "Housing Price Predictor"

        # Mock model_manager.set_model_name
        ml_engine.model_manager.set_model_name = MagicMock()

        result = ml_engine.set_model_name(user_id, model_id, custom_name)

        assert result is True
        ml_engine.model_manager.set_model_name.assert_called_once_with(
            user_id, model_id, custom_name
        )

    def test_set_model_name_invalid_name_too_short(self, ml_engine):
        """Test set_model_name raises ValidationError for too short name."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "ab"

        with pytest.raises(ValidationError) as exc_info:
            ml_engine.set_model_name(user_id, model_id, custom_name)

        assert "at least 3 characters" in str(exc_info.value)
        assert exc_info.value.field == "custom_name"
        assert exc_info.value.value == custom_name

    def test_set_model_name_invalid_name_special_chars(self, ml_engine):
        """Test set_model_name raises ValidationError for invalid characters."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "Model/Test"

        with pytest.raises(ValidationError) as exc_info:
            ml_engine.set_model_name(user_id, model_id, custom_name)

        assert "can only contain" in str(exc_info.value).lower()

    def test_set_model_name_duplicate_warning(self, ml_engine, caplog):
        """Test set_model_name logs warning for duplicate names."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "Existing Model"

        # Mock get_model_by_name to return existing model
        existing_model = {"model_id": "model_12345_other_20251013"}
        ml_engine.get_model_by_name = MagicMock(return_value=existing_model)
        ml_engine.model_manager.set_model_name = MagicMock()

        with caplog.at_level(logging.WARNING):
            result = ml_engine.set_model_name(user_id, model_id, custom_name)

        assert result is True
        assert "already has a model named" in caplog.text
        assert custom_name in caplog.text

    def test_set_model_name_model_not_found(self, ml_engine):
        """Test set_model_name raises ModelNotFoundError for non-existent model."""
        user_id = 12345
        model_id = "nonexistent_model"
        custom_name = "Valid Name"

        # Mock model_manager to raise exception with "not found"
        ml_engine.model_manager.set_model_name = MagicMock(
            side_effect=Exception("Model not found")
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            ml_engine.set_model_name(user_id, model_id, custom_name)

        assert model_id in str(exc_info.value)

    def test_set_model_name_propagates_other_exceptions(self, ml_engine):
        """Test set_model_name propagates non-not-found exceptions."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "Valid Name"

        # Mock model_manager to raise different exception
        ml_engine.model_manager.set_model_name = MagicMock(
            side_effect=Exception("Some other error")
        )

        with pytest.raises(Exception) as exc_info:
            ml_engine.set_model_name(user_id, model_id, custom_name)

        assert "Some other error" in str(exc_info.value)


class TestGetModelByName:
    """Test get_model_by_name() method."""

    def test_get_model_by_name_found(self, ml_engine):
        """Test retrieving model by custom name when it exists."""
        user_id = 12345
        custom_name = "Housing Predictor"

        # Mock list_models to return models with custom names
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "custom_name": "Housing Predictor",
                "created_at": "2025-01-14T21:44:00Z"
            },
            {
                "model_id": "model_12345_random_forest_20251013",
                "custom_name": "Other Model",
                "created_at": "2025-01-13T10:00:00Z"
            }
        ]
        ml_engine.list_models = MagicMock(return_value=models)

        result = ml_engine.get_model_by_name(user_id, custom_name)

        assert result is not None
        assert result["model_id"] == "model_12345_linear_20251014"
        assert result["custom_name"] == custom_name

    def test_get_model_by_name_not_found(self, ml_engine):
        """Test retrieving model by custom name when it doesn't exist."""
        user_id = 12345
        custom_name = "Nonexistent Model"

        # Mock list_models to return models without matching name
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "custom_name": "Other Model",
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.list_models = MagicMock(return_value=models)

        result = ml_engine.get_model_by_name(user_id, custom_name)

        assert result is None

    def test_get_model_by_name_returns_most_recent(self, ml_engine, caplog):
        """Test get_model_by_name returns most recent when multiple matches."""
        user_id = 12345
        custom_name = "Duplicate Name"

        # Mock list_models to return multiple models with same name
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "custom_name": "Duplicate Name",
                "created_at": "2025-01-14T21:44:00Z"
            },
            {
                "model_id": "model_12345_linear_20251013",
                "custom_name": "Duplicate Name",
                "created_at": "2025-01-13T10:00:00Z"
            },
            {
                "model_id": "model_12345_linear_20251015",
                "custom_name": "Duplicate Name",
                "created_at": "2025-01-15T12:00:00Z"
            }
        ]
        ml_engine.list_models = MagicMock(return_value=models)

        with caplog.at_level(logging.WARNING):
            result = ml_engine.get_model_by_name(user_id, custom_name)

        # Should return the most recent (2025-01-15)
        assert result is not None
        assert result["model_id"] == "model_12345_linear_20251015"
        assert "Multiple models named" in caplog.text
        assert "returning most recent" in caplog.text

    def test_get_model_by_name_no_custom_name(self, ml_engine):
        """Test get_model_by_name with models that have no custom name."""
        user_id = 12345
        custom_name = "Some Name"

        # Mock list_models to return models without custom_name field
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.list_models = MagicMock(return_value=models)

        result = ml_engine.get_model_by_name(user_id, custom_name)

        assert result is None

    def test_get_model_by_name_empty_list(self, ml_engine):
        """Test get_model_by_name when user has no models."""
        user_id = 12345
        custom_name = "Any Name"

        ml_engine.list_models = MagicMock(return_value=[])

        result = ml_engine.get_model_by_name(user_id, custom_name)

        assert result is None


class TestListModels:
    """Test list_models() method with display_name generation."""

    def test_list_models_adds_display_name_for_custom(self, ml_engine):
        """Test list_models adds display_name field using custom_name."""
        user_id = 12345

        # Mock model_manager.list_user_models
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "model_type": "linear",
                "task_type": "regression",
                "custom_name": "My Custom Model",
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.model_manager.list_user_models = MagicMock(return_value=models)

        result = ml_engine.list_models(user_id)

        assert len(result) == 1
        assert result[0]["display_name"] == "My Custom Model"
        assert result[0]["custom_name"] == "My Custom Model"

    def test_list_models_generates_default_display_name(self, ml_engine):
        """Test list_models generates default display_name when no custom_name."""
        user_id = 12345

        # Mock model_manager.list_user_models
        models = [
            {
                "model_id": "model_12345_keras_binary_classification_20251014",
                "model_type": "keras_binary_classification",
                "task_type": "classification",
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.model_manager.list_user_models = MagicMock(return_value=models)

        result = ml_engine.list_models(user_id)

        assert len(result) == 1
        assert result[0]["display_name"] == "Binary Classification - Jan 14, 2025"
        assert result[0]["custom_name"] is None

    def test_list_models_mixed_custom_and_default(self, ml_engine):
        """Test list_models with mix of custom and default names."""
        user_id = 12345

        # Mock model_manager.list_user_models
        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "model_type": "linear",
                "task_type": "regression",
                "custom_name": "Housing Predictor",
                "created_at": "2025-01-14T21:44:00Z"
            },
            {
                "model_id": "model_12345_random_forest_20251013",
                "model_type": "random_forest",
                "task_type": "classification",
                "created_at": "2025-01-13T10:00:00Z"
            }
        ]
        ml_engine.model_manager.list_user_models = MagicMock(return_value=models)

        result = ml_engine.list_models(user_id)

        assert len(result) == 2
        assert result[0]["display_name"] == "Housing Predictor"
        assert result[0]["custom_name"] == "Housing Predictor"
        assert result[1]["display_name"] == "Random Forest - Jan 13, 2025"
        assert result[1]["custom_name"] is None

    def test_list_models_forwards_filters(self, ml_engine):
        """Test list_models forwards task_type and model_type filters."""
        user_id = 12345
        task_type = "regression"
        model_type = "linear"

        ml_engine.model_manager.list_user_models = MagicMock(return_value=[])

        ml_engine.list_models(user_id, task_type=task_type, model_type=model_type)

        ml_engine.model_manager.list_user_models.assert_called_once_with(
            user_id,
            task_type=task_type,
            model_type=model_type
        )

    def test_list_models_empty_list(self, ml_engine):
        """Test list_models with empty model list."""
        user_id = 12345

        ml_engine.model_manager.list_user_models = MagicMock(return_value=[])

        result = ml_engine.list_models(user_id)

        assert result == []

    def test_list_models_handles_none_custom_name(self, ml_engine):
        """Test list_models handles explicit None custom_name."""
        user_id = 12345

        # Mock model_manager.list_user_models
        models = [
            {
                "model_id": "model_12345_logistic_20251014",
                "model_type": "logistic",
                "task_type": "classification",
                "custom_name": None,
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.model_manager.list_user_models = MagicMock(return_value=models)

        result = ml_engine.list_models(user_id)

        assert len(result) == 1
        assert result[0]["display_name"] == "Logistic Regression - Jan 14, 2025"
        assert result[0]["custom_name"] is None


class TestModelNamingEdgeCases:
    """Test edge cases and integration scenarios for model naming."""

    def test_validate_model_name_unicode_characters(self, ml_engine):
        """Test validation behavior with unicode characters."""
        # Unicode characters should fail validation (not in allowed set)
        is_valid, error_msg = ml_engine._validate_model_name("Model über")
        assert is_valid is False
        assert "can only contain" in error_msg.lower()

    def test_generate_default_name_year_rollover(self, ml_engine):
        """Test default name generation across year boundary."""
        name = ml_engine._generate_default_name(
            model_type="linear",
            task_type="regression",
            created_at="2026-01-01T00:00:00Z"
        )
        assert "2026" in name
        assert "Jan 01" in name

    def test_set_model_name_whitespace_trimming(self, ml_engine):
        """Test set_model_name trims whitespace from input."""
        user_id = 12345
        model_id = "model_12345_linear_20251014"
        custom_name = "  Valid Model  "

        ml_engine.model_manager.set_model_name = MagicMock()

        result = ml_engine.set_model_name(user_id, model_id, custom_name)

        # Should succeed because whitespace is trimmed during validation
        assert result is True

    def test_get_model_by_name_case_sensitive(self, ml_engine):
        """Test get_model_by_name is case-sensitive."""
        user_id = 12345

        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "custom_name": "Housing Predictor",
                "created_at": "2025-01-14T21:44:00Z"
            }
        ]
        ml_engine.list_models = MagicMock(return_value=models)

        # Exact match should work
        result = ml_engine.get_model_by_name(user_id, "Housing Predictor")
        assert result is not None

        # Different case should not match
        result = ml_engine.get_model_by_name(user_id, "housing predictor")
        assert result is None

        result = ml_engine.get_model_by_name(user_id, "HOUSING PREDICTOR")
        assert result is None

    def test_list_models_preserves_all_fields(self, ml_engine):
        """Test list_models preserves all original fields while adding display_name."""
        user_id = 12345

        models = [
            {
                "model_id": "model_12345_linear_20251014",
                "model_type": "linear",
                "task_type": "regression",
                "target_column": "price",
                "feature_columns": ["sqft", "bedrooms"],
                "metrics": {"mse": 0.15, "r2": 0.85},
                "created_at": "2025-01-14T21:44:00Z",
                "custom_field": "custom_value"
            }
        ]
        ml_engine.model_manager.list_user_models = MagicMock(return_value=models)

        result = ml_engine.list_models(user_id)

        assert len(result) == 1
        # Check all original fields are preserved
        assert result[0]["model_id"] == "model_12345_linear_20251014"
        assert result[0]["model_type"] == "linear"
        assert result[0]["task_type"] == "regression"
        assert result[0]["target_column"] == "price"
        assert result[0]["feature_columns"] == ["sqft", "bedrooms"]
        assert result[0]["metrics"] == {"mse": 0.15, "r2": 0.85}
        assert result[0]["created_at"] == "2025-01-14T21:44:00Z"
        assert result[0]["custom_field"] == "custom_value"
        # Check new fields are added
        assert "display_name" in result[0]
        assert result[0]["custom_name"] is None
