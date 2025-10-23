"""
Unit tests for LightGBM Trainer.

Tests LightGBM gradient boosting model training functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.trainers.lightgbm_trainer import LightGBMTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class TestLightGBMTrainer:
    """Test LightGBM trainer functionality."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create ML Engine config with temp directory."""
        config = MLEngineConfig.get_default()
        config.models_dir = tmp_path / "models"
        config.models_dir.mkdir(exist_ok=True)
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create LightGBM trainer instance."""
        return LightGBMTrainer(config)

    @pytest.fixture
    def binary_classification_data(self):
        """Create synthetic binary classification data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100), name="target")
        return X, y

    @pytest.fixture
    def multiclass_classification_data(self):
        """Create synthetic multiclass classification data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(150, 4), columns=[f"f{i}" for i in range(4)])
        y = pd.Series(np.random.randint(0, 3, 150), name="target")
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Create synthetic regression data."""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(120, 3), columns=["a", "b", "c"])
        y = pd.Series(np.random.randn(120), name="y")
        return X, y

    # Test 1: Model instance creation
    def test_get_model_instance_binary_classification(self, trainer):
        """Test creating binary classification model."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {"n_estimators": 50, "num_leaves": 25}
        )

        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["num_leaves"] == 25
        assert params["objective"] == "binary"
        assert params["metric"] == "auc"

    def test_get_model_instance_multiclass_classification(self, trainer):
        """Test creating multiclass classification model."""
        model = trainer.get_model_instance(
            "lightgbm_multiclass_classification",
            {"n_estimators": 75}
        )

        params = model.get_params()
        assert params["n_estimators"] == 75
        assert params["objective"] == "multiclass"
        assert params["metric"] == "multi_logloss"

    def test_get_model_instance_regression(self, trainer):
        """Test creating regression model."""
        model = trainer.get_model_instance(
            "lightgbm_regression",
            {"n_estimators": 150, "learning_rate": 0.05}
        )

        params = model.get_params()
        assert params["n_estimators"] == 150
        assert params["learning_rate"] == 0.05
        assert params["objective"] == "regression"
        assert params["metric"] == "rmse"

    def test_get_model_instance_default_hyperparameters(self, trainer):
        """Test model creation with default hyperparameters."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {}
        )

        params = model.get_params()
        assert params["n_estimators"] == 100
        assert params["num_leaves"] == 31
        assert params["learning_rate"] == 0.1
        assert params["feature_fraction"] == 0.8
        assert params["bagging_fraction"] == 0.8

    def test_get_model_instance_lightgbm_specific_params(self, trainer):
        """Test LightGBM-specific parameters (num_leaves, feature_fraction, bagging)."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {
                "num_leaves": 63,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.9,
                "bagging_freq": 5
            }
        )

        params = model.get_params()
        assert params["num_leaves"] == 63  # LightGBM uses num_leaves, not max_depth
        assert params["feature_fraction"] == 0.7  # colsample_bytree equivalent
        assert params["bagging_fraction"] == 0.9  # subsample equivalent
        assert params["bagging_freq"] == 5

    def test_get_model_instance_invalid_type(self, trainer):
        """Test error handling for invalid model type."""
        with pytest.raises(TrainingError, match="Unknown LightGBM model type"):
            trainer.get_model_instance("invalid_model_type", {})

    # Test 2: Training
    def test_train_binary_classification(self, trainer, binary_classification_data):
        """Test training binary classification model."""
        X, y = binary_classification_data

        model = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_model = trainer.train(model, X, y)

        # Verify model is trained
        assert hasattr(trained_model, 'feature_importances_')
        assert len(trained_model.feature_importances_) == 5

    def test_train_multiclass_classification(self, trainer, multiclass_classification_data):
        """Test training multiclass classification model."""
        X, y = multiclass_classification_data

        model = trainer.get_model_instance("lightgbm_multiclass_classification", {})
        trained_model = trainer.train(model, X, y)

        assert hasattr(trained_model, 'feature_importances_')
        assert len(trained_model.feature_importances_) == 4

    def test_train_regression(self, trainer, regression_data):
        """Test training regression model."""
        X, y = regression_data

        model = trainer.get_model_instance("lightgbm_regression", {})
        trained_model = trainer.train(model, X, y)

        assert hasattr(trained_model, 'feature_importances_')
        assert len(trained_model.feature_importances_) == 3

    # Test 3: Metrics calculation
    def test_calculate_metrics_binary_classification(self, trainer):
        """Test metric calculation for binary classification."""
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])

        metrics = trainer.calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_calculate_metrics_multiclass_classification(self, trainer):
        """Test metric calculation for multiclass classification."""
        y_true = pd.Series([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1])

        metrics = trainer.calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

    def test_calculate_metrics_regression(self, trainer):
        """Test metric calculation for regression."""
        y_true = pd.Series([1.5, 2.3, 3.1, 4.7, 5.2])
        y_pred = np.array([1.6, 2.1, 3.3, 4.5, 5.4])

        metrics = trainer.calculate_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_calculate_metrics_with_auc_roc(self, trainer, binary_classification_data):
        """Test AUC-ROC calculation for binary classification."""
        X, y = binary_classification_data

        model = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_model = trainer.train(model, X, y)

        y_pred = trained_model.predict(X)
        metrics = trainer.calculate_metrics(y, y_pred, trained_model, X, y)

        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

    # Test 4: Validation
    def test_validate_model(self, trainer, binary_classification_data):
        """Test model validation."""
        X, y = binary_classification_data

        # Split data
        split_idx = 80
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train model
        model = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_model = trainer.train(model, X_train, y_train)

        # Validate
        metrics = trainer.validate_model(trained_model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    # Test 5: Feature importance
    def test_feature_importance_extraction(self, trainer, binary_classification_data):
        """Test feature importance extraction."""
        X, y = binary_classification_data
        feature_names = X.columns.tolist()

        model = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_model = trainer.train(model, X, y)

        summary = trainer.get_model_summary(trained_model, "lightgbm_binary_classification", feature_names)

        assert "feature_importance" in summary
        assert len(summary["feature_importance"]) == 5
        assert all(isinstance(v, float) for v in summary["feature_importance"].values())

        # Check that features are sorted by importance
        importances = list(summary["feature_importance"].values())
        assert importances == sorted(importances, reverse=True)

    # Test 6: Model summary
    def test_get_model_summary(self, trainer, regression_data):
        """Test model summary generation."""
        X, y = regression_data
        feature_names = ["a", "b", "c"]

        model = trainer.get_model_instance("lightgbm_regression", {})
        trained_model = trainer.train(model, X, y)

        summary = trainer.get_model_summary(trained_model, "lightgbm_regression", feature_names)

        assert summary["model_type"] == "lightgbm_regression"
        assert summary["n_features"] == 3
        assert summary["feature_names"] == feature_names
        assert summary["framework"] == "lightgbm"
        assert "n_estimators" in summary
        assert "num_leaves" in summary
        assert "learning_rate" in summary
        assert "feature_fraction" in summary
        assert "bagging_fraction" in summary
        assert "feature_importance" in summary

    # Test 7: Hyperparameter merging
    def test_hyperparameter_merging(self, trainer):
        """Test hyperparameter merging with user-provided values."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {
                "n_estimators": 200,
                "num_leaves": 50,
                # feature_fraction should use default (0.8)
            }
        )

        params = model.get_params()
        assert params["n_estimators"] == 200
        assert params["num_leaves"] == 50
        assert params["feature_fraction"] == 0.8  # default

    # Test 8: Supported models
    def test_get_supported_models(self):
        """Test getting list of supported models."""
        supported = LightGBMTrainer.get_supported_models()

        assert "lightgbm_binary_classification" in supported
        assert "lightgbm_multiclass_classification" in supported
        assert "lightgbm_regression" in supported
        assert len(supported) == 3

    # Test 9: String representation
    def test_repr(self, trainer):
        """Test string representation."""
        repr_str = repr(trainer)
        assert "LightGBMTrainer" in repr_str
        assert "lightgbm_binary_classification" in repr_str

    # Test 10: Error scenarios
    def test_train_with_invalid_data(self, trainer):
        """Test training with invalid data raises appropriate error."""
        model = trainer.get_model_instance("lightgbm_binary_classification", {})

        # Empty dataframe
        X_empty = pd.DataFrame()
        y_empty = pd.Series()

        with pytest.raises(TrainingError):
            trainer.train(model, X_empty, y_empty)

    # Test 11: Early stopping (optional)
    def test_train_with_early_stopping(self, trainer, binary_classification_data):
        """Test training with early stopping."""
        X, y = binary_classification_data

        # Split for validation
        split_idx = 80
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_model = trainer.train(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=10,
            verbose=False
        )

        assert hasattr(trained_model, 'feature_importances_')

    # Test 12: All model types end-to-end
    def test_all_model_types_end_to_end(self, trainer, binary_classification_data,
                                        multiclass_classification_data, regression_data):
        """Test complete workflow for all model types."""
        # Binary classification
        X_bin, y_bin = binary_classification_data
        model_bin = trainer.get_model_instance("lightgbm_binary_classification", {})
        trained_bin = trainer.train(model_bin, X_bin, y_bin)
        metrics_bin = trainer.validate_model(trained_bin, X_bin[:20], y_bin[:20])
        assert "accuracy" in metrics_bin

        # Multiclass classification
        X_multi, y_multi = multiclass_classification_data
        model_multi = trainer.get_model_instance("lightgbm_multiclass_classification", {})
        trained_multi = trainer.train(model_multi, X_multi, y_multi)
        metrics_multi = trainer.validate_model(trained_multi, X_multi[:30], y_multi[:30])
        assert "accuracy" in metrics_multi

        # Regression
        X_reg, y_reg = regression_data
        model_reg = trainer.get_model_instance("lightgbm_regression", {})
        trained_reg = trainer.train(model_reg, X_reg, y_reg)
        metrics_reg = trainer.validate_model(trained_reg, X_reg[:25], y_reg[:25])
        assert "r2" in metrics_reg

    # Test 13: LightGBM-specific performance characteristics
    def test_lightgbm_leaf_wise_growth(self, trainer, binary_classification_data):
        """Test that LightGBM uses leaf-wise growth (num_leaves parameter)."""
        X, y = binary_classification_data

        # Create model with specific num_leaves
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {"num_leaves": 15}  # LightGBM-specific parameter
        )
        trained_model = trainer.train(model, X, y)

        params = trained_model.get_params()
        assert params["num_leaves"] == 15
        assert "max_depth" not in params or params.get("max_depth") == -1  # LightGBM default

    # Test 14: Regularization parameters
    def test_regularization_parameters(self, trainer):
        """Test L1 and L2 regularization parameters."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {
                "lambda_l1": 0.1,
                "lambda_l2": 0.2
            }
        )

        params = model.get_params()
        assert params["lambda_l1"] == 0.1
        assert params["lambda_l2"] == 0.2

    # Test 15: Bagging parameters
    def test_bagging_parameters(self, trainer):
        """Test bagging parameters (LightGBM-specific)."""
        model = trainer.get_model_instance(
            "lightgbm_binary_classification",
            {
                "bagging_fraction": 0.7,
                "bagging_freq": 3
            }
        )

        params = model.get_params()
        assert params["bagging_fraction"] == 0.7
        assert params["bagging_freq"] == 3
