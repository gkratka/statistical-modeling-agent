"""
Unit tests for Keras architecture templates.

Tests template generation for binary/multiclass classification and regression.
"""

import pytest
from src.engines.trainers.keras_templates import (
    get_binary_classification_template,
    get_multiclass_classification_template,
    get_regression_template,
    get_template
)


class TestBinaryClassificationTemplate:
    """Test binary classification template generation."""

    def test_default_binary_template(self):
        """Test default binary classification template."""
        template = get_binary_classification_template(n_features=14)

        assert "layers" in template
        assert "compile" in template
        assert len(template["layers"]) == 2

        # Check hidden layer
        assert template["layers"][0]["type"] == "Dense"
        assert template["layers"][0]["units"] == 14
        assert template["layers"][0]["activation"] == "relu"
        assert template["layers"][0]["kernel_initializer"] == "random_normal"

        # Check output layer
        assert template["layers"][1]["type"] == "Dense"
        assert template["layers"][1]["units"] == 1
        assert template["layers"][1]["activation"] == "sigmoid"

        # Check compile config
        assert template["compile"]["loss"] == "binary_crossentropy"
        assert template["compile"]["optimizer"] == "adam"
        assert "accuracy" in template["compile"]["metrics"]

    def test_custom_initializer(self):
        """Test binary template with custom initializer."""
        template = get_binary_classification_template(
            n_features=10,
            kernel_initializer="glorot_uniform"
        )

        assert template["layers"][0]["kernel_initializer"] == "glorot_uniform"
        assert template["layers"][1]["kernel_initializer"] == "glorot_uniform"


class TestMulticlassClassificationTemplate:
    """Test multiclass classification template generation."""

    def test_default_multiclass_template(self):
        """Test default multiclass classification template."""
        template = get_multiclass_classification_template(
            n_features=10,
            n_classes=5
        )

        assert len(template["layers"]) == 2

        # Check hidden layer
        assert template["layers"][0]["units"] == 10
        assert template["layers"][0]["activation"] == "relu"

        # Check output layer
        assert template["layers"][1]["units"] == 5
        assert template["layers"][1]["activation"] == "softmax"

        # Check compile config
        assert template["compile"]["loss"] == "categorical_crossentropy"

    def test_different_class_counts(self):
        """Test template with different class counts."""
        for n_classes in [3, 5, 10, 20]:
            template = get_multiclass_classification_template(
                n_features=8,
                n_classes=n_classes
            )
            assert template["layers"][1]["units"] == n_classes


class TestRegressionTemplate:
    """Test regression template generation."""

    def test_default_regression_template(self):
        """Test default regression template."""
        template = get_regression_template(n_features=12)

        assert len(template["layers"]) == 2

        # Check hidden layer
        assert template["layers"][0]["units"] == 12
        assert template["layers"][0]["activation"] == "relu"
        assert template["layers"][0]["kernel_initializer"] == "glorot_uniform"

        # Check output layer
        assert template["layers"][1]["units"] == 1
        assert template["layers"][1]["activation"] == "linear"

        # Check compile config
        assert template["compile"]["loss"] == "mse"
        assert "mae" in template["compile"]["metrics"]


class TestGetTemplate:
    """Test unified get_template function."""

    def test_binary_classification_routing(self):
        """Test routing to binary classification template."""
        template = get_template(
            model_type="keras_binary_classification",
            n_features=14
        )

        assert template["layers"][1]["activation"] == "sigmoid"
        assert template["compile"]["loss"] == "binary_crossentropy"

    def test_multiclass_classification_routing(self):
        """Test routing to multiclass classification template."""
        template = get_template(
            model_type="keras_multiclass_classification",
            n_features=10,
            n_classes=5
        )

        assert template["layers"][1]["units"] == 5
        assert template["layers"][1]["activation"] == "softmax"
        assert template["compile"]["loss"] == "categorical_crossentropy"

    def test_regression_routing(self):
        """Test routing to regression template."""
        template = get_template(
            model_type="keras_regression",
            n_features=8
        )

        assert template["layers"][1]["activation"] == "linear"
        assert template["compile"]["loss"] == "mse"

    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError) as exc_info:
            get_template(
                model_type="invalid_model",
                n_features=10
            )
        assert "Unknown Keras model type" in str(exc_info.value)

    def test_custom_kernel_initializer(self):
        """Test custom kernel initializer propagation."""
        template = get_template(
            model_type="keras_binary_classification",
            n_features=14,
            kernel_initializer="he_uniform"
        )

        assert template["layers"][0]["kernel_initializer"] == "he_uniform"
        assert template["layers"][1]["kernel_initializer"] == "he_uniform"
