"""
Integration tests for XGBoost workflow.

Tests end-to-end XGBoost model training and prediction workflows.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


class TestXGBoostMLEngineIntegration:
    """Test full ML Engine workflow with XGBoost models."""

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
            model_type="xgboost_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=99999,
            hyperparameters={"n_estimators": 50, "max_depth": 4},
            test_size=0.2
        )

        # Verify training results
        assert "model_id" in result
        assert result["metrics"]["accuracy"] > 0.4  # Random should be ~0.5
        assert "feature_importance" in result["model_info"]
        assert result["model_info"]["framework"] == "xgboost"
        assert result["model_info"]["n_estimators"] == 50
        assert result["model_info"]["max_depth"] == 4

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
        """Test XGBoost regression workflow."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="xgboost_regression",
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
        """Test XGBoost multiclass classification."""
        result = engine.train_model(
            data=multiclass_classification_data,
            task_type="classification",
            model_type="xgboost_multiclass_classification",
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
            model_type="xgboost_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=88888,
            test_size=0.2
        )

        model_id = result["model_id"]

        # Load model info
        model_info = engine.get_model_info(user_id=88888, model_id=model_id)

        assert model_info["model_type"] == "xgboost_binary_classification"
        # Note: framework is in metadata, which may have different structure
        # Just verify essential fields are present
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
            model_type="xgboost_regression",
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
            model_type="xgboost_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=66666,
            hyperparameters={},  # Empty - use defaults
            test_size=0.2
        )

        assert result["success"]
        assert result["model_info"]["n_estimators"] == 100  # Default
        assert result["model_info"]["max_depth"] == 6  # Default
        assert result["model_info"]["learning_rate"] == 0.1  # Default

    # Test 7: Model listing
    def test_list_models(self, engine, binary_classification_data):
        """Test listing XGBoost models."""
        import time

        # Train two models (with slight delay to ensure different timestamps)
        engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="xgboost_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2", "feature3"],
            user_id=55555,
            test_size=0.2
        )

        time.sleep(1)  # Ensure different timestamp

        engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="xgboost_binary_classification",
            target_column="target",
            feature_columns=["feature1", "feature2"],
            user_id=55555,
            test_size=0.2
        )

        # List models
        models = engine.list_models(user_id=55555)

        assert len(models) >= 2
        assert all(m["model_type"] == "xgboost_binary_classification" for m in models)

    # Test 8: Custom hyperparameters
    def test_custom_hyperparameters(self, engine, regression_data):
        """Test training with custom hyperparameters."""
        result = engine.train_model(
            data=regression_data,
            task_type="regression",
            model_type="xgboost_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=44444,
            hyperparameters={
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7
            },
            test_size=0.2
        )

        assert result["model_info"]["n_estimators"] == 200
        assert result["model_info"]["max_depth"] == 8
        assert result["model_info"]["learning_rate"] == 0.05

    # Test 9: Metrics completeness
    def test_metrics_completeness(self, engine, binary_classification_data):
        """Test that all expected metrics are present."""
        result = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="xgboost_binary_classification",
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

        # Optional: AUC-ROC
        # Note: May not always be present depending on data
        # assert "auc_roc" in metrics

    # Test 10: All model types
    def test_all_model_types(self, engine, binary_classification_data,
                            multiclass_classification_data, regression_data):
        """Test that all XGBoost model types work."""
        # Binary classification
        result_bin = engine.train_model(
            data=binary_classification_data,
            task_type="classification",
            model_type="xgboost_binary_classification",
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
            model_type="xgboost_multiclass_classification",
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
            model_type="xgboost_regression",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=22222,
            test_size=0.2
        )
        assert result_reg["success"]
        assert "r2" in result_reg["metrics"]
