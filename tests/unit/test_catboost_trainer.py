"""
Unit tests for CatBoost Trainer.

Tests CatBoost gradient boosting model training functionality including:
- GPU auto-detection with CPU fallback
- Categorical feature auto-detection
- Binary/multiclass classification and regression
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.engines.trainers.catboost_trainer import CatBoostTrainer
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import TrainingError


class TestCatBoostTrainer:
    """Test CatBoost trainer functionality."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create ML Engine config with temp directory."""
        config = MLEngineConfig.get_default()
        config.models_dir = tmp_path / "models"
        config.models_dir.mkdir(exist_ok=True)
        return config

    @pytest.fixture
    def trainer(self, config):
        """Create CatBoost trainer instance."""
        return CatBoostTrainer(config)

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

    @pytest.fixture
    def categorical_data(self):
        """Create data with categorical features."""
        np.random.seed(42)
        df = pd.DataFrame({
            "cat1": pd.Categorical(["A", "B", "C"] * 30),
            "cat2": ["X", "Y"] * 45,
            "num1": np.random.randn(90),
            "num2": np.random.randn(90)
        })
        return df

    # Test Group 1: Model instance creation (6 tests)
    def test_get_model_instance_binary_classification(self, trainer):
        """Test creating binary classification model."""
        model = trainer.get_model_instance(
            "catboost_binary_classification",
            {"iterations": 50, "depth": 4}
        )

        params = model.get_params()
        assert params["iterations"] == 50
        assert params["depth"] == 4
        assert params["loss_function"] == "Logloss"
        assert params["eval_metric"] == "AUC"

    def test_get_model_instance_multiclass_classification(self, trainer):
        """Test creating multiclass classification model."""
        model = trainer.get_model_instance(
            "catboost_multiclass_classification",
            {"iterations": 75}
        )

        params = model.get_params()
        assert params["iterations"] == 75
        assert params["loss_function"] == "MultiClass"
        assert params["eval_metric"] == "MultiClass"

    def test_get_model_instance_regression(self, trainer):
        """Test creating regression model."""
        model = trainer.get_model_instance(
            "catboost_regression",
            {"iterations": 150, "learning_rate": 0.05}
        )

        params = model.get_params()
        assert params["iterations"] == 150
        assert params["learning_rate"] == 0.05
        assert params["loss_function"] == "RMSE"
        assert params["eval_metric"] == "RMSE"

    def test_get_model_instance_default_hyperparameters(self, trainer):
        """Test model creation with default hyperparameters."""
        model = trainer.get_model_instance(
            "catboost_binary_classification",
            {}
        )

        params = model.get_params()
        assert params["iterations"] == 1000
        assert params["depth"] == 6
        assert params["learning_rate"] == 0.03
        assert params["l2_leaf_reg"] == 3

    def test_get_model_instance_custom_params(self, trainer):
        """Test model with custom CatBoost-specific parameters."""
        model = trainer.get_model_instance(
            "catboost_regression",
            {
                "iterations": 200,
                "depth": 8,
                "learning_rate": 0.01,
                "l2_leaf_reg": 5,
                "border_count": 128,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.7
            }
        )

        params = model.get_params()
        assert params["iterations"] == 200
        assert params["depth"] == 8
        assert params["learning_rate"] == 0.01
        assert params["l2_leaf_reg"] == 5
        assert params["border_count"] == 128
        assert params["bootstrap_type"] == "Bernoulli"
        assert params["subsample"] == 0.7

    def test_get_model_instance_unsupported_type(self, trainer):
        """Test error handling for unsupported model type."""
        with pytest.raises(TrainingError, match="Unknown CatBoost model type"):
            trainer.get_model_instance("catboost_invalid", {})

    # Test Group 2: Categorical feature detection (4 tests)
    def test_detect_categorical_features_object_dtype(self, trainer):
        """Test detection of object dtype columns as categorical."""
        df = pd.DataFrame({
            "cat1": ["A", "B", "C"],
            "num1": [1, 2, 3],
            "cat2": ["X", "Y", "Z"]
        })

        cat_features = trainer._detect_categorical_features(df)
        assert "cat1" in cat_features
        assert "cat2" in cat_features
        assert "num1" not in cat_features
        assert len(cat_features) == 2

    def test_detect_categorical_features_category_dtype(self, trainer):
        """Test detection of category dtype columns."""
        df = pd.DataFrame({
            "cat1": pd.Categorical(["A", "B", "C"]),
            "num1": [1.0, 2.0, 3.0],
            "cat2": pd.Categorical([1, 2, 3])
        })

        cat_features = trainer._detect_categorical_features(df)
        assert "cat1" in cat_features
        assert "cat2" in cat_features
        assert "num1" not in cat_features

    def test_detect_categorical_features_mixed_types(self, trainer, categorical_data):
        """Test categorical detection with mixed data types."""
        cat_features = trainer._detect_categorical_features(categorical_data)

        assert "cat1" in cat_features
        assert "cat2" in cat_features
        assert "num1" not in cat_features
        assert "num2" not in cat_features
        assert len(cat_features) == 2

    def test_detect_categorical_features_empty_dataframe(self, trainer):
        """Test categorical detection with no categorical columns."""
        df = pd.DataFrame({
            "num1": [1, 2, 3],
            "num2": [4.0, 5.0, 6.0]
        })

        cat_features = trainer._detect_categorical_features(df)
        assert len(cat_features) == 0

    # Test Group 3: GPU detection (3 tests)
    def test_gpu_detection_real_system(self, trainer):
        """Test GPU detection on real system (GPU or CPU based on hardware)."""
        model = trainer.get_model_instance("catboost_binary_classification", {})
        params = model.get_params()
        # Should be either GPU or CPU based on system
        assert params["task_type"] in ["GPU", "CPU"]

    def test_force_cpu_mode(self, config):
        """Test forcing CPU mode by setting gpu_available to False."""
        trainer = CatBoostTrainer(config)
        trainer._import_catboost()
        trainer._gpu_available = False  # Force CPU mode
        model = trainer.get_model_instance("catboost_binary_classification", {})
        params = model.get_params()
        assert params["task_type"] == "CPU"

    def test_force_gpu_mode(self, config):
        """Test forcing GPU mode by setting gpu_available to True."""
        trainer = CatBoostTrainer(config)
        trainer._import_catboost()
        trainer._gpu_available = True  # Force GPU mode
        model = trainer.get_model_instance("catboost_binary_classification", {})
        params = model.get_params()
        assert params["task_type"] == "GPU"

    # Test Group 4: Training (5 tests)
    def test_train_binary_classification(self, trainer, binary_classification_data):
        """Test training binary classification model."""
        X, y = binary_classification_data
        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10})

        trained_model = trainer.train(model, X, y)

        assert trained_model is not None
        predictions = trained_model.predict(X)
        assert len(predictions) == len(y)

    def test_train_multiclass_classification(self, trainer, multiclass_classification_data):
        """Test training multiclass classification model."""
        X, y = multiclass_classification_data
        model = trainer.get_model_instance("catboost_multiclass_classification", {"iterations": 10})

        trained_model = trainer.train(model, X, y)

        assert trained_model is not None
        predictions = trained_model.predict(X)
        assert len(predictions) == len(y)

    def test_train_regression(self, trainer, regression_data):
        """Test training regression model."""
        X, y = regression_data
        model = trainer.get_model_instance("catboost_regression", {"iterations": 10, "verbose": False})

        trained_model = trainer.train(model, X, y, verbose=False)

        assert trained_model is not None
        predictions = trained_model.predict(X)
        assert len(predictions) == len(y)

    def test_train_with_categorical_features(self, trainer, categorical_data):
        """Test training with auto-detected categorical features."""
        y = pd.Series(np.random.randint(0, 2, len(categorical_data)), name="target")
        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10})

        # CatBoost should auto-handle categorical features
        trained_model = trainer.train(model, categorical_data, y)

        assert trained_model is not None
        predictions = trained_model.predict(categorical_data)
        assert len(predictions) == len(y)

    def test_train_with_early_stopping(self, trainer, binary_classification_data):
        """Test training with early stopping."""
        X, y = binary_classification_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 100})

        trained_model = trainer.train(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=10
        )

        assert trained_model is not None

    # Test Group 5: Metrics (3 tests)
    def test_calculate_metrics_binary_classification(self, trainer, binary_classification_data):
        """Test metric calculation for binary classification."""
        X, y = binary_classification_data
        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10})
        trained_model = trainer.train(model, X, y)

        y_pred = trained_model.predict(X)
        metrics = trainer.calculate_metrics(y, y_pred, trained_model, X, y)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_calculate_metrics_regression(self, trainer, regression_data):
        """Test metric calculation for regression."""
        X, y = regression_data
        model = trainer.get_model_instance("catboost_regression", {"iterations": 10, "verbose": False})
        trained_model = trainer.train(model, X, y, verbose=False)

        y_pred = trained_model.predict(X)
        metrics = trainer.calculate_metrics(y, y_pred, trained_model, X, y)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0

    def test_validate_model(self, trainer, binary_classification_data):
        """Test model validation."""
        X, y = binary_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10})
        trained_model = trainer.train(model, X_train, y_train)

        validation_results = trainer.validate_model(trained_model, X_test, y_test)

        assert isinstance(validation_results, dict)
        assert "accuracy" in validation_results

    # Test Group 6: Model summary (2 tests)
    def test_get_model_summary_binary_classification(self, trainer, binary_classification_data):
        """Test model summary for binary classification."""
        X, y = binary_classification_data
        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10, "verbose": False})
        trained_model = trainer.train(model, X, y, verbose=False)

        summary = trainer.get_model_summary(
            trained_model,
            "catboost_binary_classification",
            list(X.columns)
        )

        assert summary["model_type"] == "catboost_binary_classification"
        assert summary["framework"] == "catboost"
        assert summary["n_features"] == len(X.columns)
        assert "feature_names" in summary
        assert "feature_importance" in summary
        assert "iterations" in summary
        assert "depth" in summary
        assert "learning_rate" in summary

    def test_get_model_summary_with_gpu_info(self, trainer, regression_data):
        """Test model summary includes GPU/CPU task type."""
        X, y = regression_data
        model = trainer.get_model_instance("catboost_regression", {"iterations": 10, "verbose": False})
        trained_model = trainer.train(model, X, y, verbose=False)

        summary = trainer.get_model_summary(
            trained_model,
            "catboost_regression",
            list(X.columns)
        )

        assert "task_type" in summary
        assert summary["task_type"] in ["GPU", "CPU"]

    # Test Group 7: Feature importance (2 tests)
    def test_feature_importance_available(self, trainer, binary_classification_data):
        """Test that feature importance is calculated."""
        X, y = binary_classification_data
        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10, "verbose": False})
        trained_model = trainer.train(model, X, y, verbose=False)

        summary = trainer.get_model_summary(
            trained_model,
            "catboost_binary_classification",
            list(X.columns)
        )

        feature_importance = summary["feature_importance"]
        assert len(feature_importance) == len(X.columns)
        assert all(score >= 0 for score in feature_importance.values())

    def test_feature_importance_sorted(self, trainer, regression_data):
        """Test that feature importance is sorted by score."""
        X, y = regression_data
        model = trainer.get_model_instance("catboost_regression", {"iterations": 10, "verbose": False})
        trained_model = trainer.train(model, X, y, verbose=False)

        summary = trainer.get_model_summary(
            trained_model,
            "catboost_regression",
            list(X.columns)
        )

        importance_values = list(summary["feature_importance"].values())
        assert importance_values == sorted(importance_values, reverse=True)

    # Test Group 8: Error handling (3 tests)
    def test_training_error_handling(self, trainer):
        """Test error handling during training."""
        X = pd.DataFrame({"f1": [1, 2, 3]})
        y = pd.Series([1, 2])  # Mismatched length

        model = trainer.get_model_instance("catboost_binary_classification", {"iterations": 10})

        with pytest.raises(TrainingError, match="CatBoost model training failed"):
            trainer.train(model, X, y)

    def test_import_error_handling(self, config):
        """Test error handling when CatBoost import fails."""
        # Patch at the import source level
        with patch.dict('sys.modules', {'catboost': None}):
            trainer = CatBoostTrainer(config)
            with pytest.raises(TrainingError, match="CatBoost import failed"):
                trainer._import_catboost()

    def test_get_supported_models(self, trainer):
        """Test getting list of supported models."""
        supported = trainer.get_supported_models()

        assert "catboost_binary_classification" in supported
        assert "catboost_multiclass_classification" in supported
        assert "catboost_regression" in supported
        assert len(supported) == 3

    # Test Group 9: Misc (1 test)
    def test_repr(self, trainer):
        """Test string representation."""
        repr_str = repr(trainer)
        assert "CatBoostTrainer" in repr_str
        assert "catboost_binary_classification" in repr_str
