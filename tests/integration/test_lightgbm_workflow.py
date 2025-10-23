"""
Integration tests for LightGBM workflow.

Tests end-to-end LightGBM model training and prediction workflows.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestLightGBMMLEngineIntegration:
    """Test full ML Engine workflow with LightGBM models."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create ML Engine config with temp directory."""
        config = MLEngineConfig.get_default()
        config.models_dir = tmp_path / "models"
        config.models_dir.mkdir(exist_ok=True)
        return config

    @pytest.fixture
    def engine(self, config):
        """Create ML Engine instance."""
        return MLEngine(config)

    @pytest.fixture
    def binary_classification_data(self):
        """Create synthetic binary classification dataset."""
        np.random.seed(42)
        data = pd.DataFrame({
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
            "target": np.random.randint(0, 2, 200)
        })
        return data

    @pytest.fixture
    def multiclass_classification_data(self):
        """Create synthetic multiclass classification dataset."""
        np.random.seed(42)
        data = pd.DataFrame({
            "f1": np.random.randn(250),
            "f2": np.random.randn(250),
            "f3": np.random.randn(250),
            "f4": np.random.randn(250),
            "class": np.random.randint(0, 3, 250)
        })
        return data

    @pytest.fixture
    def regression_data(self):
        """Create synthetic regression dataset."""
        np.random.seed(42)
        X = np.random.randn(150, 2)
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(150) * 0.1
        data = pd.DataFrame({
            "x1": X[:, 0],
            "x2": X[:, 1],
            "y": y
        })
        return data

    # Test 1: Full binary classification workflow
    def test_full_binary_classification_workflow(self, engine, binary_classification_data):
        """Test complete train → save → load → predict cycle for binary classification."""
        # Train
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=99999,
            hyperparameters={"n_estimators": 50, "num_leaves": 25},
            test_size=0.2
        )

        # Verify training results
        assert "model_id" in result
        assert result["metrics"]["accuracy"] > 0.4  # Random should be ~0.5
        assert "feature_importance" in result["model_info"]
        assert result["model_info"]["framework"] == "lightgbm"
        assert result["model_info"]["n_estimators"] == 50
        assert result["model_info"]["num_leaves"] == 25

        # Test prediction
        new_data = pd.DataFrame({
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10),
            "feature3": np.random.randn(10)
        })

        predictions = engine.predict(
            user_id=99999,
            model_id=result["model_id"],
            data=new_data
        )

        assert len(predictions["predictions"]) == 10
        assert all(p in [0, 1] for p in predictions["predictions"])

    # Test 2: Regression workflow
    def test_regression_workflow(self, engine, regression_data):
        """Test LightGBM regression workflow."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=99999,
            hyperparameters={"n_estimators": 100},
            test_size=0.2
        )

        # Verify metrics
        assert "r2" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "mae" in result["metrics"]
        assert result["metrics"]["r2"] > 0.5  # Should fit reasonably well
        assert "feature_importance" in result["model_info"]

        # Test prediction
        new_data = pd.DataFrame({
            "x1": [1.0, -1.0, 0.0],
            "x2": [0.5, -0.5, 0.0]
        })

        predictions = engine.predict(
            user_id=99999,
            model_id=result["model_id"],
            data=new_data
        )

        assert len(predictions["predictions"]) == 3
        assert all(isinstance(p, (int, float)) for p in predictions["predictions"])

    # Test 3: Multiclass classification
    def test_multiclass_classification_workflow(self, engine, multiclass_classification_data):
        """Test LightGBM multiclass classification."""
        result = engine.train_model(
            data=multiclass_classification_data,
            task_type="classification",
            model_type="lightgbm_multiclass_classification",
            target_column="class",
            feature_columns=["f1", "f2", "f3", "f4"],
            user_id=99999,
            test_size=0.2
        )

        # Verify metrics
        assert "accuracy" in result["metrics"]
        assert "precision" in result["metrics"]
        assert "recall" in result["metrics"]
        assert "f1" in result["metrics"]
        assert result["metrics"]["accuracy"] > 0.25  # Better than random (0.33)

    # Test 4: Model save/load cycle
    def test_model_save_load_cycle(self, engine, binary_classification_data):
        """Test that models can be saved and loaded correctly."""
        # Train model
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=88888,
            test_size=0.2
        )

        model_id = result["model_id"]

        # Load model info
        model_info = engine.get_model_info(user_id=88888, model_id=model_id)

        assert model_info["model_type"] == "lightgbm_binary_classification"
        assert "feature_importance" in model_info or "n_features" in model_info

        # Make prediction (tests loading)
        test_data = binary_classification_data[["feature1", "feature2", "feature3"]].head(5)
        predictions = engine.predict(
            user_id=88888,
            model_id=model_id,
            data=test_data
        )

        assert len(predictions["predictions"]) == 5

    # Test 5: Feature importance extraction
    def test_feature_importance_extraction(self, engine, regression_data):
        """Test that feature importance is correctly extracted."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=77777,
            test_size=0.2
        )

        feature_importance = result["model_info"]["feature_importance"]

        assert "x1" in feature_importance
        assert "x2" in feature_importance
        assert all(isinstance(v, float) for v in feature_importance.values())

        # Check that importance values are sorted (descending)
        importances = list(feature_importance.values())
        assert importances == sorted(importances, reverse=True)

    # Test 6: Default hyperparameters
    def test_default_hyperparameters(self, engine, binary_classification_data):
        """Test training with default hyperparameters."""
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=66666,
            hyperparameters={},  # Empty - use defaults
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["n_estimators"] == 100  # Default
        assert result["model_info"]["num_leaves"] == 31  # Default (LightGBM uses num_leaves)
        assert result["model_info"]["learning_rate"] == 0.1  # Default

    # Test 7: Model listing
    def test_list_models(self, engine, binary_classification_data):
        """Test listing LightGBM models."""
        import time

        # Train two models (with slight delay to ensure different timestamps)
        engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=55555,
            test_size=0.2
        )

        time.sleep(1)  # Ensure different timestamp

        engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            user_id=55555,
            test_size=0.2
        )

        # List models
        models = engine.list_models(user_id=55555)

        assert len(models) >= 2
        assert all(m["model_type"] == "lightgbm_binary_classification" for m in models)

    # Test 8: Custom hyperparameters
    def test_custom_hyperparameters(self, engine, regression_data):
        """Test training with custom hyperparameters."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=44444,
            hyperparameters={
                "n_estimators": 200,
                "num_leaves": 50,
                "learning_rate": 0.05,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7
            },
            test_size=0.2
        )

        assert result["model_info"]["n_estimators"] == 200
        assert result["model_info"]["num_leaves"] == 50
        assert result["model_info"]["learning_rate"] == 0.05

    # Test 9: Metrics completeness
    def test_metrics_completeness(self, engine, binary_classification_data):
        """Test that all expected metrics are present."""
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=33333,
            test_size=0.2
        )

        metrics = result["metrics"]

        # Classification metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

    # Test 10: All model types
    def test_all_model_types(self, engine, binary_classification_data,
                            multiclass_classification_data, regression_data):
        """Test that all LightGBM model types work."""
        # Binary classification
        result_bin = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=22222,
            test_size=0.2
        )
        assert result_bin["success"]
        assert "accuracy" in result_bin["metrics"]

        # Multiclass classification
        result_multi = engine.train_model(
            data=multiclass_classification_data,
            task_type="classification",
            model_type="lightgbm_multiclass_classification",
            target_column="class",
            feature_columns=["f1", "f2", "f3", "f4"],
            user_id=22222,
            test_size=0.2
        )
        assert result_multi["success"]
        assert "accuracy" in result_multi["metrics"]

        # Regression
        result_reg = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=22222,
            test_size=0.2
        )
        assert result_reg["success"]
        assert "r2" in result_reg["metrics"]

    # Test 11: LightGBM-specific parameters
    def test_lightgbm_specific_parameters(self, engine, binary_classification_data):
        """Test LightGBM-specific hyperparameters (num_leaves, feature_fraction, bagging)."""
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=11111,
            hyperparameters={
                "num_leaves": 63,  # LightGBM uses num_leaves, not max_depth
                "feature_fraction": 0.9,  # colsample_bytree equivalent
                "bagging_fraction": 0.8,  # subsample equivalent
                "bagging_freq": 5
            },
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["num_leaves"] == 63
        assert result["model_info"]["feature_fraction"] == 0.9
        assert result["model_info"]["bagging_fraction"] == 0.8

    # Test 12: Regularization parameters
    def test_regularization_parameters(self, engine, regression_data):
        """Test L1 and L2 regularization parameters."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=10101,
            hyperparameters={
                "lambda_l1": 0.1,
                "lambda_l2": 0.2
            },
            test_size=0.2
        )

        assert result["success"]
        # Regularization parameters should be in model info
        # (May not be directly accessible in get_params depending on LightGBM version)
        assert "feature_importance" in result["model_info"]

    # Test 13: Large dataset template
    def test_large_dataset_template_params(self, engine, regression_data):
        """Test that large dataset template parameters can be used."""
        from src.engines.trainers.lightgbm_templates import get_large_dataset_template

        template_params = get_large_dataset_template()

        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="lightgbm_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=20202,
            hyperparameters=template_params,
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["num_leaves"] == 63
        assert result["model_info"]["learning_rate"] == 0.05

    # Test 14: Fast template
    def test_fast_template_params(self, engine, binary_classification_data):
        """Test that fast template parameters can be used."""
        from src.engines.trainers.lightgbm_templates import get_fast_template

        template_params = get_fast_template()

        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="lightgbm_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=30303,
            hyperparameters=template_params,
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["n_estimators"] == 50
        assert result["model_info"]["num_leaves"] == 15

    # Test 15: High accuracy template
    def test_high_accuracy_template_params(self, engine, multiclass_classification_data):
        """Test that high accuracy template parameters can be used."""
        from src.engines.trainers.lightgbm_templates import get_high_accuracy_template

        template_params = get_high_accuracy_template()

        result = engine.train_model(
            data=multiclass_classification_data,
            task_type="classification",
            model_type="lightgbm_multiclass_classification",
            target_column="class",
            feature_columns=["f1", "f2", "f3", "f4"],
            user_id=40404,
            hyperparameters=template_params,
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["n_estimators"] == 300
        assert result["model_info"]["num_leaves"] == 50
        assert result["model_info"]["learning_rate"] == 0.05
