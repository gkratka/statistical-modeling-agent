"""
Unit tests for ML Neural Network components.

Tests the neural network trainer for both regression and classification tasks.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.neural_network_trainer import NeuralNetworkTrainer
from src.utils.exceptions import TrainingError


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
            "mlp_regression": {"hidden_layers": [100], "max_iter": 100},
            "mlp_classification": {"hidden_layers": [50], "max_iter": 100}
        },
        hyperparameter_ranges={
            "alpha": [0.0001, 1.0],
            "learning_rate_init": [0.0001, 0.1]
        }
    )


@pytest.fixture
def regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 100

    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Create target with non-linear relationship
    target = (
        feature1 * 2.0 +
        feature2 ** 2 * 0.5 +
        np.sin(feature3) * 1.5 +
        np.random.randn(n_samples) * 0.3
    )

    return pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target
    })


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification dataset."""
    np.random.seed(42)
    n_samples = 100

    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Create non-linear decision boundary
    target = ((feature1 ** 2 + feature2 ** 2) > 1.0).astype(int)

    return pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target
    })


@pytest.fixture
def multiclass_classification_data():
    """Create sample multiclass classification dataset."""
    np.random.seed(42)
    n_samples = 150

    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)

    # Create 3 classes
    target = np.random.choice([0, 1, 2], size=n_samples)

    return pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target
    })


class TestNeuralNetworkTrainer:
    """Test neural network trainer."""

    def test_trainer_initialization(self, ml_config):
        """Test trainer initializes correctly."""
        trainer = NeuralNetworkTrainer(ml_config)
        assert trainer.config == ml_config
        assert len(trainer.SUPPORTED_MODELS) == 2

    def test_get_model_instance_mlp_regression(self, ml_config):
        """Test creating MLP regression model."""
        trainer = NeuralNetworkTrainer(ml_config)
        model = trainer.get_model_instance("mlp_regression", {})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_get_model_instance_mlp_classification(self, ml_config):
        """Test creating MLP classification model."""
        trainer = NeuralNetworkTrainer(ml_config)
        model = trainer.get_model_instance("mlp_classification", {})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_get_model_instance_with_architecture(self, ml_config):
        """Test creating model with custom architecture."""
        trainer = NeuralNetworkTrainer(ml_config)
        model = trainer.get_model_instance(
            "mlp_regression",
            {"hidden_layers": [64, 32], "activation": "tanh"}
        )
        assert model is not None
        assert model.hidden_layer_sizes == (64, 32)
        assert model.activation == "tanh"

    def test_get_model_instance_single_layer(self, ml_config):
        """Test creating model with single hidden layer as int."""
        trainer = NeuralNetworkTrainer(ml_config)
        model = trainer.get_model_instance(
            "mlp_classification",
            {"hidden_layers": 50}
        )
        assert model is not None
        assert model.hidden_layer_sizes == (50,)

    def test_get_model_instance_with_hyperparameters(self, ml_config):
        """Test creating model with various hyperparameters."""
        trainer = NeuralNetworkTrainer(ml_config)
        model = trainer.get_model_instance(
            "mlp_regression",
            {
                "hidden_layers": [100, 50],
                "alpha": 0.001,
                "learning_rate": "adaptive",
                "learning_rate_init": 0.01,
                "max_iter": 300,
                "early_stopping": True
            }
        )
        assert model.alpha == 0.001
        assert model.learning_rate == "adaptive"
        assert model.learning_rate_init == 0.01
        assert model.max_iter == 300
        assert model.early_stopping is True

    def test_get_model_instance_invalid(self, ml_config):
        """Test creating model with invalid type."""
        trainer = NeuralNetworkTrainer(ml_config)
        with pytest.raises(TrainingError) as exc_info:
            trainer.get_model_instance("invalid_model", {})
        assert "unknown" in str(exc_info.value).lower()

    def test_calculate_regression_metrics(self, ml_config):
        """Test regression metrics calculation."""
        trainer = NeuralNetworkTrainer(ml_config)

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

    def test_calculate_classification_metrics(self, ml_config):
        """Test classification metrics calculation."""
        trainer = NeuralNetworkTrainer(ml_config)

        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = pd.Series([0, 0, 1, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8],
                            [0.1, 0.9], [0.85, 0.15], [0.15, 0.85]])

        metrics = trainer.calculate_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        # Perfect predictions
        assert metrics["accuracy"] == 1.0

    def test_prepare_data(self, ml_config, regression_data):
        """Test data preparation."""
        trainer = NeuralNetworkTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3'],
            test_size=0.2
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert list(X_train.columns) == ['feature1', 'feature2', 'feature3']

    def test_train_mlp_regression(self, ml_config, regression_data):
        """Test MLP regression training."""
        trainer = NeuralNetworkTrainer(ml_config)

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        # Create and train model
        model = trainer.get_model_instance(
            "mlp_regression",
            {"hidden_layers": [50, 25], "max_iter": 100}
        )
        trained_model = trainer.train(model, X_train, y_train)

        assert trained_model is not None
        assert hasattr(trained_model, 'coefs_')
        assert hasattr(trained_model, 'n_iter_')

        # Make predictions
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_train_mlp_classification(self, ml_config, binary_classification_data):
        """Test MLP classification training."""
        trainer = NeuralNetworkTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        model = trainer.get_model_instance(
            "mlp_classification",
            {"hidden_layers": [30, 15], "max_iter": 100}
        )
        trained_model = trainer.train(model, X_train, y_train)

        assert trained_model is not None
        assert hasattr(trained_model, 'classes_')

        # Make predictions
        predictions = trained_model.predict(X_test)
        probabilities = trained_model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification

    def test_train_multiclass_mlp(self, ml_config, multiclass_classification_data):
        """Test multiclass MLP classification."""
        trainer = NeuralNetworkTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            multiclass_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance(
            "mlp_classification",
            {"hidden_layers": [20], "max_iter": 100}
        )
        trained_model = trainer.train(model, X_train, y_train)

        # Make predictions
        predictions = trained_model.predict(X_test)
        probabilities = trained_model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape[1] == 3  # 3 classes

    def test_get_model_summary_regression(self, ml_config, regression_data):
        """Test model summary for regression."""
        trainer = NeuralNetworkTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            regression_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        model = trainer.get_model_instance(
            "mlp_regression",
            {"hidden_layers": [64, 32], "activation": "relu", "max_iter": 50}
        )
        trainer.train(model, X_train, y_train)

        summary = trainer.get_model_summary(
            model,
            "mlp_regression",
            ['feature1', 'feature2', 'feature3']
        )

        assert summary["model_type"] == "mlp_regression"
        assert summary["n_features"] == 3
        assert summary["hidden_layer_sizes"] == [64, 32]
        assert summary["n_layers"] == 3  # 2 hidden + 1 output
        assert "n_iterations" in summary
        assert summary["activation"] == "relu"
        assert "n_parameters" in summary
        assert summary["n_parameters"] > 0

    def test_get_model_summary_classification(self, ml_config, binary_classification_data):
        """Test model summary for classification."""
        trainer = NeuralNetworkTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance(
            "mlp_classification",
            {"hidden_layers": [50], "max_iter": 50}
        )
        trainer.train(model, X_train, y_train)

        summary = trainer.get_model_summary(
            model,
            "mlp_classification",
            ['feature1', 'feature2']
        )

        assert summary["model_type"] == "mlp_classification"
        assert summary["n_classes"] == 2
        assert "classes" in summary
        assert len(summary["classes"]) == 2

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        models = NeuralNetworkTrainer.get_supported_models()
        assert len(models) == 2
        assert "mlp_regression" in models
        assert "mlp_classification" in models
