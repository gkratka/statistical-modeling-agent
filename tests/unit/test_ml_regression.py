"""
Unit tests for ML Regression components.

Tests the regression trainer, validators, and preprocessors.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.regression_trainer import RegressionTrainer
from src.engines.ml_validators import MLValidators
from src.engines.ml_preprocessors import MLPreprocessors
from src.utils.exceptions import (
    DataValidationError,
    TrainingError,
    ValidationError
)


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
        default_hyperparameters={
            "ridge": {"alpha": 1.0},
            "lasso": {"alpha": 1.0}
        },
        hyperparameter_ranges={
            "alpha": [0.0001, 10.0]
        }
    )


@pytest.fixture
def regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target': np.random.randn(100)
    })


class TestMLValidators:
    """Test ML validators."""

    def test_validate_training_data_success(self, regression_data):
        """Test successful training data validation."""
        # Should not raise
        MLValidators.validate_training_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            min_samples=10
        )

    def test_validate_training_data_empty(self):
        """Test validation fails on empty data."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError) as exc_info:
            MLValidators.validate_training_data(
                empty_df,
                target_column='target',
                feature_columns=['feature1'],
                min_samples=10
            )
        assert "empty" in str(exc_info.value).lower()

    def test_validate_training_data_insufficient_samples(self):
        """Test validation fails with insufficient samples."""
        small_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [1, 2, 3]
        })
        with pytest.raises(DataValidationError) as exc_info:
            MLValidators.validate_training_data(
                small_df,
                target_column='target',
                feature_columns=['feature1'],
                min_samples=10
            )
        assert "insufficient" in str(exc_info.value).lower()

    def test_validate_training_data_missing_target(self, regression_data):
        """Test validation fails with missing target column."""
        with pytest.raises(DataValidationError) as exc_info:
            MLValidators.validate_training_data(
                regression_data,
                target_column='nonexistent',
                feature_columns=['feature1'],
                min_samples=10
            )
        assert "target column" in str(exc_info.value).lower()

    def test_validate_training_data_no_variance(self):
        """Test validation fails when target has no variance."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [5, 5, 5, 5, 5]  # All same value
        })
        with pytest.raises(DataValidationError) as exc_info:
            MLValidators.validate_training_data(
                df,
                target_column='target',
                feature_columns=['feature1'],
                min_samples=3
            )
        assert "variance" in str(exc_info.value).lower()

    def test_sanitize_column_name(self):
        """Test column name sanitization."""
        assert MLValidators.sanitize_column_name("valid_name") == "valid_name"
        assert MLValidators.sanitize_column_name("name with spaces") == "name_with_spaces"
        assert MLValidators.sanitize_column_name("123numeric") == "_123numeric"
        assert MLValidators.sanitize_column_name("special!@#chars") == "special___chars"

    def test_validate_test_size(self):
        """Test test_size validation."""
        MLValidators.validate_test_size(0.2)  # Should pass
        MLValidators.validate_test_size(0.3)  # Should pass
        MLValidators.validate_test_size(0.0)  # Now valid (for training on 100% of data)

        with pytest.raises(ValidationError):
            MLValidators.validate_test_size(1.0)  # Invalid

        with pytest.raises(ValidationError):
            MLValidators.validate_test_size(1.5)  # Invalid


class TestMLPreprocessors:
    """Test ML preprocessors."""

    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean strategy."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [10.0, np.nan, 30.0, 40.0]
        })
        result = MLPreprocessors.handle_missing_values(df, strategy="mean")

        assert not result['feature1'].isnull().any()
        assert not result['feature2'].isnull().any()
        assert result['feature1'].iloc[2] == pytest.approx(2.333, rel=0.1)

    def test_handle_missing_values_median(self):
        """Test missing value handling with median strategy."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 100.0]  # Outlier to test median
        })
        result = MLPreprocessors.handle_missing_values(df, strategy="median")

        assert not result['feature1'].isnull().any()
        assert result['feature1'].iloc[2] == 2.0  # Median of [1, 2, 100]

    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop strategy."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })
        result = MLPreprocessors.handle_missing_values(df, strategy="drop")

        assert len(result) == 3  # One row dropped
        assert not result.isnull().any().any()

    def test_scale_features_standard(self):
        """Test standard scaling."""
        train_df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        test_df = pd.DataFrame({
            'feature1': [3.0, 4.0]
        })

        train_scaled, test_scaled, scaler = MLPreprocessors.scale_features(
            train_df, test_df, method="standard"
        )

        # Check that mean is approximately 0 and std is approximately 1
        # Note: pandas .std() uses ddof=1 by default, so value will be ~1.118, not 1.0
        assert train_scaled['feature1'].mean() == pytest.approx(0.0, abs=1e-10)
        assert train_scaled['feature1'].std(ddof=0) == pytest.approx(1.0, abs=0.1)
        assert scaler is not None

    def test_scale_features_none(self):
        """Test no scaling."""
        train_df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})
        test_df = pd.DataFrame({'feature1': [4.0, 5.0]})

        train_scaled, test_scaled, scaler = MLPreprocessors.scale_features(
            train_df, test_df, method="none"
        )

        pd.testing.assert_frame_equal(train_scaled, train_df)
        pd.testing.assert_frame_equal(test_scaled, test_df)
        assert scaler is None


class TestRegressionTrainer:
    """Test regression trainer."""

    def test_trainer_initialization(self, ml_config):
        """Test trainer initializes correctly."""
        trainer = RegressionTrainer(ml_config)
        assert trainer.config == ml_config
        assert len(trainer.SUPPORTED_MODELS) == 5

    def test_get_model_instance_linear(self, ml_config):
        """Test creating linear regression model."""
        trainer = RegressionTrainer(ml_config)
        model = trainer.get_model_instance("linear", {})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_instance_ridge(self, ml_config):
        """Test creating ridge regression model."""
        trainer = RegressionTrainer(ml_config)
        model = trainer.get_model_instance("ridge", {"alpha": 2.0})
        assert model is not None
        assert hasattr(model, 'alpha')
        assert model.alpha == 2.0

    def test_get_model_instance_invalid(self, ml_config):
        """Test creating model with invalid type."""
        trainer = RegressionTrainer(ml_config)
        with pytest.raises(TrainingError) as exc_info:
            trainer.get_model_instance("invalid_model", {})
        assert "unknown" in str(exc_info.value).lower()

    def test_calculate_metrics(self, ml_config):
        """Test metric calculation."""
        trainer = RegressionTrainer(ml_config)
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics = trainer.calculate_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "explained_variance" in metrics

        assert metrics["mse"] > 0
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
        assert metrics["r2"] <= 1.0

    def test_prepare_data(self, ml_config, regression_data):
        """Test data preparation."""
        trainer = RegressionTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            test_size=0.2
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert list(X_train.columns) == ['feature1', 'feature2']

    def test_train_model(self, ml_config, regression_data):
        """Test model training."""
        trainer = RegressionTrainer(ml_config)

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        # Create and train model
        model = trainer.get_model_instance("linear", {})
        trained_model = trainer.train(model, X_train, y_train)

        assert trained_model is not None
        assert hasattr(trained_model, 'coef_')

        # Make predictions
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_get_feature_importance(self, ml_config, regression_data):
        """Test feature importance extraction."""
        trainer = RegressionTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance("linear", {})
        trainer.train(model, X_train, y_train)

        feature_importance = trainer.get_feature_importance(
            model,
            ['feature1', 'feature2']
        )

        assert feature_importance is not None
        assert 'feature1' in feature_importance
        assert 'feature2' in feature_importance
        assert isinstance(feature_importance['feature1'], float)
