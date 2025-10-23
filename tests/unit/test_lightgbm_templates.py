"""
Unit tests for LightGBM hyperparameter templates.

Tests template generation for binary/multiclass classification and regression.
"""

import pytest
from src.engines.trainers.lightgbm_templates import (
    get_binary_classification_template,
    get_multiclass_classification_template,
    get_regression_template,
    get_large_dataset_template,
    get_fast_template,
    get_high_accuracy_template,
    get_template,
    TEMPLATE_COMPARISON
)


class TestBinaryClassificationTemplate:
    """Test binary classification template generation."""

    def test_default_binary_template(self):
        """Test default binary classification template."""
        template = get_binary_classification_template()

        # Verify core parameters
        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31
        assert template["learning_rate"] == 0.1
        assert template["feature_fraction"] == 0.8
        assert template["bagging_fraction"] == 0.8
        assert template["bagging_freq"] == 1
        assert template["min_data_in_leaf"] == 20
        assert template["random_state"] == 42
        assert template["n_jobs"] == -1

    def test_custom_n_estimators(self):
        """Test binary template with custom n_estimators."""
        template = get_binary_classification_template(n_estimators=200)
        assert template["n_estimators"] == 200
        assert template["num_leaves"] == 31  # default

    def test_custom_num_leaves(self):
        """Test binary template with custom num_leaves."""
        template = get_binary_classification_template(num_leaves=63)
        assert template["num_leaves"] == 63
        assert template["n_estimators"] == 100  # default

    def test_custom_learning_rate(self):
        """Test binary template with custom learning_rate."""
        template = get_binary_classification_template(learning_rate=0.05)
        assert template["learning_rate"] == 0.05

    def test_all_custom_parameters(self):
        """Test binary template with all custom parameters."""
        template = get_binary_classification_template(
            n_estimators=150,
            num_leaves=50,
            learning_rate=0.08
        )
        assert template["n_estimators"] == 150
        assert template["num_leaves"] == 50
        assert template["learning_rate"] == 0.08


class TestMulticlassClassificationTemplate:
    """Test multiclass classification template generation."""

    def test_default_multiclass_template(self):
        """Test default multiclass classification template."""
        template = get_multiclass_classification_template()

        # Same defaults as binary
        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31
        assert template["learning_rate"] == 0.1
        assert template["feature_fraction"] == 0.8
        assert template["bagging_fraction"] == 0.8

    def test_custom_parameters(self):
        """Test multiclass template with custom parameters."""
        template = get_multiclass_classification_template(
            n_estimators=120,
            num_leaves=40
        )
        assert template["n_estimators"] == 120
        assert template["num_leaves"] == 40


class TestRegressionTemplate:
    """Test regression template generation."""

    def test_default_regression_template(self):
        """Test default regression template."""
        template = get_regression_template()

        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31
        assert template["learning_rate"] == 0.1
        assert template["feature_fraction"] == 0.8
        assert template["bagging_fraction"] == 0.8
        assert template["min_data_in_leaf"] == 20

    def test_custom_parameters(self):
        """Test regression template with custom parameters."""
        template = get_regression_template(
            n_estimators=180,
            learning_rate=0.03
        )
        assert template["n_estimators"] == 180
        assert template["learning_rate"] == 0.03


class TestLargeDatasetTemplate:
    """Test template optimized for large datasets."""

    def test_default_large_dataset_template(self):
        """Test default large dataset template."""
        template = get_large_dataset_template()

        # Higher num_leaves for large datasets
        assert template["num_leaves"] == 63
        # Lower learning rate for stability
        assert template["learning_rate"] == 0.05
        # More aggressive sampling
        assert template["feature_fraction"] == 0.7
        assert template["bagging_fraction"] == 0.7
        assert template["bagging_freq"] == 5
        # Higher minimum samples
        assert template["min_data_in_leaf"] == 100
        # Regularization enabled
        assert template["lambda_l1"] == 0.1
        assert template["lambda_l2"] == 0.1

    def test_large_dataset_vs_standard(self):
        """Test that large dataset template differs from standard."""
        large = get_large_dataset_template()
        standard = get_binary_classification_template()

        assert large["num_leaves"] > standard["num_leaves"]
        assert large["min_data_in_leaf"] > standard["min_data_in_leaf"]
        assert large["lambda_l1"] > standard["lambda_l1"]


class TestFastTemplate:
    """Test template optimized for speed."""

    def test_default_fast_template(self):
        """Test default fast template."""
        template = get_fast_template()

        # Fewer estimators for speed
        assert template["n_estimators"] == 50
        # Fewer leaves for speed
        assert template["num_leaves"] == 15
        # Less data sampling
        assert template["feature_fraction"] == 0.7
        assert template["bagging_fraction"] == 0.7
        # Higher minimum for speed
        assert template["min_data_in_leaf"] == 50
        assert template["min_gain_to_split"] == 0.05

    def test_fast_vs_standard(self):
        """Test that fast template is lighter than standard."""
        fast = get_fast_template()
        standard = get_binary_classification_template()

        assert fast["n_estimators"] < standard["n_estimators"]
        assert fast["num_leaves"] < standard["num_leaves"]
        assert fast["min_data_in_leaf"] > standard["min_data_in_leaf"]


class TestHighAccuracyTemplate:
    """Test template optimized for accuracy."""

    def test_default_high_accuracy_template(self):
        """Test default high accuracy template."""
        template = get_high_accuracy_template()

        # More estimators for accuracy
        assert template["n_estimators"] == 300
        # More leaves for complexity
        assert template["num_leaves"] == 50
        # Lower learning rate
        assert template["learning_rate"] == 0.05
        # More data usage
        assert template["feature_fraction"] == 0.9
        assert template["bagging_fraction"] == 0.9
        # Lower minimum for granularity
        assert template["min_data_in_leaf"] == 10
        # Light regularization
        assert template["lambda_l1"] == 0.01
        assert template["lambda_l2"] == 0.01

    def test_high_accuracy_vs_standard(self):
        """Test that high accuracy template is more complex than standard."""
        accurate = get_high_accuracy_template()
        standard = get_binary_classification_template()

        assert accurate["n_estimators"] > standard["n_estimators"]
        assert accurate["num_leaves"] > standard["num_leaves"]
        assert accurate["feature_fraction"] > standard["feature_fraction"]


class TestGetTemplate:
    """Test unified get_template function."""

    def test_binary_classification_routing(self):
        """Test routing to binary classification template."""
        template = get_template("lightgbm_binary_classification")

        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31

    def test_multiclass_classification_routing(self):
        """Test routing to multiclass classification template."""
        template = get_template("lightgbm_multiclass_classification")

        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31

    def test_regression_routing(self):
        """Test routing to regression template."""
        template = get_template("lightgbm_regression")

        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31

    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError) as exc_info:
            get_template("invalid_lightgbm_model")
        assert "Unknown LightGBM model type" in str(exc_info.value)

    def test_custom_parameters_propagation(self):
        """Test that custom parameters are propagated."""
        template = get_template(
            "lightgbm_binary_classification",
            n_estimators=250,
            num_leaves=70,
            learning_rate=0.02
        )

        assert template["n_estimators"] == 250
        assert template["num_leaves"] == 70
        assert template["learning_rate"] == 0.02


class TestDatasetSizeOptimization:
    """Test dataset size-based optimization."""

    def test_small_dataset_balanced(self):
        """Test small dataset with balanced optimization."""
        template = get_template(
            "lightgbm_binary_classification",
            dataset_size="small",
            optimization="balanced"
        )

        # Should use standard template for small datasets
        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31

    def test_medium_dataset_balanced(self):
        """Test medium dataset with balanced optimization."""
        template = get_template(
            "lightgbm_regression",
            dataset_size="medium",
            optimization="balanced"
        )

        # Should use standard template
        assert template["n_estimators"] == 100
        assert template["num_leaves"] == 31

    def test_large_dataset_balanced(self):
        """Test large dataset with balanced optimization."""
        template = get_template(
            "lightgbm_multiclass_classification",
            dataset_size="large",
            optimization="balanced"
        )

        # Should use large dataset template (with its defaults)
        # The large dataset template has its own defaults that override the get_template defaults
        large_defaults = get_large_dataset_template()
        assert template["num_leaves"] == large_defaults["num_leaves"]
        assert template["min_data_in_leaf"] == large_defaults["min_data_in_leaf"]


class TestOptimizationGoals:
    """Test optimization goal-based template selection."""

    def test_speed_optimization(self):
        """Test speed optimization overrides dataset size."""
        template = get_template(
            "lightgbm_binary_classification",
            dataset_size="large",
            optimization="speed"
        )

        # Should use fast template despite large dataset (with fast template defaults)
        fast_defaults = get_fast_template()
        assert template["n_estimators"] == fast_defaults["n_estimators"]
        assert template["num_leaves"] == fast_defaults["num_leaves"]

    def test_accuracy_optimization(self):
        """Test accuracy optimization overrides dataset size."""
        template = get_template(
            "lightgbm_regression",
            dataset_size="small",
            optimization="accuracy"
        )

        # Should use high accuracy template despite small dataset (with accuracy template defaults)
        accuracy_defaults = get_high_accuracy_template()
        assert template["n_estimators"] == accuracy_defaults["n_estimators"]
        assert template["num_leaves"] == accuracy_defaults["num_leaves"]

    def test_balanced_optimization_with_large_dataset(self):
        """Test balanced optimization with large dataset."""
        template = get_template(
            "lightgbm_binary_classification",
            dataset_size="large",
            optimization="balanced"
        )

        # Should use large dataset template (with large dataset template defaults)
        large_defaults = get_large_dataset_template()
        assert template["num_leaves"] == large_defaults["num_leaves"]
        assert template["lambda_l1"] == large_defaults["lambda_l1"]


class TestLightGBMSpecificParameters:
    """Test LightGBM-specific parameters."""

    def test_num_leaves_parameter(self):
        """Test that templates use num_leaves, not max_depth."""
        templates = [
            get_binary_classification_template(),
            get_multiclass_classification_template(),
            get_regression_template(),
            get_large_dataset_template(),
            get_fast_template(),
            get_high_accuracy_template()
        ]

        for template in templates:
            assert "num_leaves" in template
            assert "max_depth" not in template

    def test_feature_fraction_parameter(self):
        """Test feature_fraction (colsample_bytree equivalent)."""
        templates = [
            get_binary_classification_template(),
            get_large_dataset_template(),
            get_fast_template(),
            get_high_accuracy_template()
        ]

        for template in templates:
            assert "feature_fraction" in template
            assert 0 < template["feature_fraction"] <= 1

    def test_bagging_parameters(self):
        """Test bagging_fraction and bagging_freq."""
        template = get_binary_classification_template()

        assert "bagging_fraction" in template
        assert "bagging_freq" in template
        assert 0 < template["bagging_fraction"] <= 1
        assert template["bagging_freq"] >= 0


class TestRegularizationParameters:
    """Test L1 and L2 regularization parameters."""

    def test_default_no_regularization(self):
        """Test that default templates have no regularization."""
        template = get_binary_classification_template()

        assert template["lambda_l1"] == 0.0
        assert template["lambda_l2"] == 0.0

    def test_large_dataset_has_regularization(self):
        """Test that large dataset template has regularization."""
        template = get_large_dataset_template()

        assert template["lambda_l1"] > 0
        assert template["lambda_l2"] > 0

    def test_high_accuracy_has_light_regularization(self):
        """Test that high accuracy template has light regularization."""
        template = get_high_accuracy_template()

        assert template["lambda_l1"] == 0.01
        assert template["lambda_l2"] == 0.01


class TestTemplateComparison:
    """Test template comparison guide."""

    def test_comparison_exists(self):
        """Test that TEMPLATE_COMPARISON exists and has expected keys."""
        assert "default" in TEMPLATE_COMPARISON
        assert "large_dataset" in TEMPLATE_COMPARISON
        assert "fast" in TEMPLATE_COMPARISON
        assert "high_accuracy" in TEMPLATE_COMPARISON

    def test_comparison_structure(self):
        """Test that each comparison entry has expected fields."""
        for key, value in TEMPLATE_COMPARISON.items():
            assert "speed" in value
            assert "accuracy" in value
            assert "use_case" in value


class TestParameterRanges:
    """Test that parameter values are within reasonable ranges."""

    def test_n_estimators_range(self):
        """Test that n_estimators are within reasonable range."""
        templates = [
            get_binary_classification_template(),
            get_fast_template(),
            get_high_accuracy_template(),
            get_large_dataset_template()
        ]

        for template in templates:
            assert 1 <= template["n_estimators"] <= 500

    def test_num_leaves_range(self):
        """Test that num_leaves are within reasonable range."""
        templates = [
            get_binary_classification_template(),
            get_fast_template(),
            get_high_accuracy_template(),
            get_large_dataset_template()
        ]

        for template in templates:
            assert 1 <= template["num_leaves"] <= 200

    def test_learning_rate_range(self):
        """Test that learning rates are within reasonable range."""
        templates = [
            get_binary_classification_template(),
            get_fast_template(),
            get_high_accuracy_template(),
            get_large_dataset_template()
        ]

        for template in templates:
            assert 0.001 <= template["learning_rate"] <= 0.5

    def test_fraction_parameters_range(self):
        """Test that fraction parameters are between 0 and 1."""
        templates = [
            get_binary_classification_template(),
            get_fast_template(),
            get_high_accuracy_template(),
            get_large_dataset_template()
        ]

        for template in templates:
            assert 0 < template["feature_fraction"] <= 1
            assert 0 < template["bagging_fraction"] <= 1
