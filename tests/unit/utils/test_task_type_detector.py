"""
Unit tests for task type detection utility.

Tests automatic detection of regression vs classification tasks based on
target variable characteristics.
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.task_type_detector import (
    detect_task_type,
    validate_model_compatibility,
    get_recommended_models
)


class TestDetectTaskType:
    """Test detect_task_type function."""

    def test_detect_regression_from_continuous_prices(self):
        """Test regression detection from housing prices."""
        prices = pd.Series([100000, 150000, 200000, 250000, 300000, 350000,
                           180000, 220000, 190000, 280000, 310000, 155000,
                           175000, 225000, 265000, 295000, 320000, 185000,
                           210000, 240000])
        assert detect_task_type(prices) == 'regression'

    def test_detect_regression_from_temperatures(self):
        """Test regression detection from temperature measurements."""
        temps = pd.Series([20.5, 21.3, 19.8, 22.1, 20.9, 21.7, 19.5, 22.4,
                          20.2, 21.0, 19.9, 22.3, 20.7, 21.5, 19.6, 22.0,
                          20.4, 21.2, 20.1, 21.8, 19.7, 22.2, 20.8, 21.4])
        assert detect_task_type(temps) == 'regression'

    def test_detect_classification_from_binary(self):
        """Test classification detection from binary 0/1."""
        binary = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1])
        assert detect_task_type(binary) == 'classification'

    def test_detect_classification_from_categories(self):
        """Test classification detection from string categories."""
        categories = pd.Series(['spam', 'ham', 'spam', 'ham', 'spam', 'ham',
                               'spam', 'spam', 'ham', 'spam', 'ham', 'ham'])
        assert detect_task_type(categories) == 'classification'

    def test_detect_classification_from_few_unique_values(self):
        """Test classification detection from few unique numeric values."""
        ratings = pd.Series([1, 2, 3, 1, 2, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3])
        assert detect_task_type(ratings) == 'classification'

    def test_detect_classification_from_boolean(self):
        """Test classification detection from boolean values."""
        boolean = pd.Series([True, False, True, True, False, True, False, False])
        assert detect_task_type(boolean) == 'classification'

    def test_detect_regression_with_many_unique_integers(self):
        """Test regression detection from integers with many unique values."""
        # Simulating square footage or similar continuous measure
        sqft = pd.Series([800, 1200, 1500, 1800, 2000, 2200, 2500, 2800,
                         950, 1100, 1350, 1650, 1900, 2100, 2400, 2700,
                         900, 1250, 1550, 1850, 2050, 2300, 2600, 2900])
        assert detect_task_type(sqft) == 'regression'

    def test_custom_unique_threshold(self):
        """Test with custom threshold for unique values."""
        # 15 unique values - would be regression with default threshold (20)
        # but classification with threshold of 10
        values = pd.Series(list(range(15)) * 2)
        assert detect_task_type(values) == 'classification'
        assert detect_task_type(values, unique_threshold=10) == 'regression'

    def test_object_dtype_is_classification(self):
        """Test that object dtype is always classification."""
        text = pd.Series(['A', 'B', 'C', 'D', 'E'] * 5)
        assert detect_task_type(text) == 'classification'

    def test_category_dtype_is_classification(self):
        """Test that category dtype is always classification."""
        cat = pd.Series(pd.Categorical(['low', 'med', 'high'] * 5))
        assert detect_task_type(cat) == 'classification'


class TestValidateModelCompatibility:
    """Test validate_model_compatibility function."""

    def test_compatible_regression_models(self):
        """Test regression models are compatible with regression task."""
        regression_models = ['linear', 'ridge', 'lasso', 'elasticnet',
                            'polynomial', 'mlp_regression', 'keras_regression']

        for model in regression_models:
            is_compatible, msg = validate_model_compatibility(model, 'regression')
            assert is_compatible is True
            assert msg == ""

    def test_compatible_classification_models(self):
        """Test classification models are compatible with classification task."""
        classification_models = [
            'logistic', 'decision_tree', 'random_forest', 'gradient_boosting',
            'svm', 'naive_bayes', 'mlp_classification',
            'keras_binary_classification', 'keras_multiclass_classification'
        ]

        for model in classification_models:
            is_compatible, msg = validate_model_compatibility(model, 'classification')
            assert is_compatible is True
            assert msg == ""

    def test_incompatible_classification_for_regression(self):
        """Test classification model rejected for regression task."""
        is_compatible, msg = validate_model_compatibility(
            'keras_binary_classification', 'regression'
        )
        assert is_compatible is False
        assert "Mismatch" in msg
        assert "continuous" in msg.lower()
        assert "regression" in msg.lower()

    def test_incompatible_regression_for_classification(self):
        """Test regression model rejected for classification task."""
        is_compatible, msg = validate_model_compatibility(
            'linear', 'classification'
        )
        assert is_compatible is False
        assert "Mismatch" in msg
        assert "categorical" in msg.lower()
        assert "classification" in msg.lower()

    def test_warning_message_contains_recommendations(self):
        """Test warning messages contain model recommendations."""
        _, msg = validate_model_compatibility('logistic', 'regression')
        assert "Recommendation" in msg or "recommend" in msg.lower()

        _, msg2 = validate_model_compatibility('linear', 'classification')
        assert "Recommendation" in msg2 or "recommend" in msg2.lower()


class TestGetRecommendedModels:
    """Test get_recommended_models function."""

    def test_neural_regression_filters_correctly(self):
        """Test neural network category filters to regression models only."""
        models = get_recommended_models('regression', 'neural')

        # Should only contain regression models
        model_types = [m[1] for m in models]
        assert 'mlp_regression' in model_types
        assert 'keras_regression' in model_types

        # Should NOT contain classification models
        assert 'mlp_classification' not in model_types
        assert 'keras_binary_classification' not in model_types
        assert 'keras_multiclass_classification' not in model_types

    def test_neural_classification_filters_correctly(self):
        """Test neural network category filters to classification models only."""
        models = get_recommended_models('classification', 'neural')

        # Should only contain classification models
        model_types = [m[1] for m in models]
        assert 'mlp_classification' in model_types
        assert 'keras_binary_classification' in model_types
        assert 'keras_multiclass_classification' in model_types

        # Should NOT contain regression models
        assert 'mlp_regression' not in model_types
        assert 'keras_regression' not in model_types

    def test_regression_category_unchanged(self):
        """Test regression category returns all regression models."""
        models = get_recommended_models('regression', 'regression')

        model_types = [m[1] for m in models]
        assert 'linear' in model_types
        assert 'ridge' in model_types
        assert 'lasso' in model_types
        assert 'elasticnet' in model_types
        assert 'polynomial' in model_types

    def test_classification_category_unchanged(self):
        """Test classification category returns all classification models."""
        models = get_recommended_models('classification', 'classification')

        model_types = [m[1] for m in models]
        assert 'logistic' in model_types
        assert 'decision_tree' in model_types
        assert 'random_forest' in model_types
        assert 'gradient_boosting' in model_types
        assert 'svm' in model_types
        assert 'naive_bayes' in model_types

    def test_returns_tuples_with_display_name(self):
        """Test that returned models include display names."""
        models = get_recommended_models('regression', 'neural')

        # Should return list of tuples (display_name, model_type)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, tuple)
            assert len(model) == 2
            display_name, model_type = model
            assert isinstance(display_name, str)
            assert isinstance(model_type, str)
            assert len(display_name) > 0
            assert len(model_type) > 0


class TestRealWorldScenarios:
    """Test with realistic datasets."""

    def test_housing_price_prediction(self):
        """Test with realistic housing price data."""
        # Realistic housing data: price, sqft, bedrooms
        prices = pd.Series([171792, 173133, 107516, 118647, 164512, 211933,
                           100459, 176441, 152420, 180631, 208335, 110443,
                           148189, 137907, 196771, 152403, 181167, 108892,
                           89823, 106725])

        # Should detect as regression
        assert detect_task_type(prices) == 'regression'

        # Should recommend regression models for neural network category
        models = get_recommended_models('regression', 'neural')
        model_types = [m[1] for m in models]
        assert 'keras_regression' in model_types
        assert 'keras_binary_classification' not in model_types

    def test_spam_classification(self):
        """Test with spam/ham classification data."""
        labels = pd.Series(['spam', 'ham', 'spam', 'ham', 'spam'] * 10)

        # Should detect as classification
        assert detect_task_type(labels) == 'classification'

        # Should recommend classification models
        models = get_recommended_models('classification', 'neural')
        model_types = [m[1] for m in models]
        assert 'keras_binary_classification' in model_types
        assert 'keras_regression' not in model_types

    def test_credit_approval_binary(self):
        """Test with binary credit approval (0/1)."""
        approval = pd.Series([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1] * 5)

        # Should detect as classification
        assert detect_task_type(approval) == 'classification'

        # Validation should accept classification models
        is_valid, _ = validate_model_compatibility(
            'keras_binary_classification', 'classification'
        )
        assert is_valid is True
