"""
Unit tests for ML Classification components.

Tests the classification trainer, validators, and preprocessors.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.classification_trainer import ClassificationTrainer
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
            "logistic": {"C": 1.0},
            "random_forest": {"n_estimators": 100}
        },
        hyperparameter_ranges={
            "C": [0.001, 100.0],
            "n_estimators": [10, 500]
        }
    )


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification dataset."""
    np.random.seed(42)
    n_samples = 100

    # Create features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)
    feature3 = np.random.randn(n_samples)

    # Create binary target based on features
    target = (feature1 + feature2 * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

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

    # Create features
    feature1 = np.random.randn(n_samples)
    feature2 = np.random.randn(n_samples)

    # Create 3-class target
    target = np.random.choice([0, 1, 2], size=n_samples)

    return pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'target': target
    })


class TestClassificationTrainer:
    """Test classification trainer."""

    def test_trainer_initialization(self, ml_config):
        """Test trainer initializes correctly."""
        trainer = ClassificationTrainer(ml_config)
        assert trainer.config == ml_config
        assert len(trainer.SUPPORTED_MODELS) == 6

    def test_get_model_instance_logistic(self, ml_config):
        """Test creating logistic regression model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("logistic", {})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_get_model_instance_decision_tree(self, ml_config):
        """Test creating decision tree model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("decision_tree", {"max_depth": 5})
        assert model is not None
        assert hasattr(model, 'max_depth')
        assert model.max_depth == 5

    def test_get_model_instance_random_forest(self, ml_config):
        """Test creating random forest model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("random_forest", {"n_estimators": 50})
        assert model is not None
        assert hasattr(model, 'n_estimators')
        assert model.n_estimators == 50

    def test_get_model_instance_gradient_boosting(self, ml_config):
        """Test creating gradient boosting model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("gradient_boosting", {"learning_rate": 0.2})
        assert model is not None
        assert hasattr(model, 'learning_rate')
        assert model.learning_rate == 0.2

    def test_get_model_instance_svm(self, ml_config):
        """Test creating SVM model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("svm", {"kernel": "linear"})
        assert model is not None
        assert hasattr(model, 'kernel')
        assert model.kernel == "linear"

    def test_get_model_instance_naive_bayes(self, ml_config):
        """Test creating naive bayes model."""
        trainer = ClassificationTrainer(ml_config)
        model = trainer.get_model_instance("naive_bayes", {})
        assert model is not None
        assert hasattr(model, 'fit')

    def test_get_model_instance_invalid(self, ml_config):
        """Test creating model with invalid type."""
        trainer = ClassificationTrainer(ml_config)
        with pytest.raises(TrainingError) as exc_info:
            trainer.get_model_instance("invalid_model", {})
        assert "unknown" in str(exc_info.value).lower()

    def test_calculate_metrics_binary(self, ml_config):
        """Test binary classification metrics calculation."""
        trainer = ClassificationTrainer(ml_config)

        # Perfect predictions
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
        assert "confusion_matrix" in metrics

        # Perfect predictions should have accuracy = 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_calculate_metrics_multiclass(self, ml_config):
        """Test multiclass classification metrics calculation."""
        trainer = ClassificationTrainer(ml_config)

        y_true = pd.Series([0, 1, 2, 0, 1, 2])
        y_pred = pd.Series([0, 1, 2, 0, 1, 2])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

        metrics = trainer.calculate_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3  # 3 classes

    def test_prepare_data(self, ml_config, binary_classification_data):
        """Test data preparation."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2'],
            test_size=0.2
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert list(X_train.columns) == ['feature1', 'feature2']

    def test_train_model_logistic(self, ml_config, binary_classification_data):
        """Test logistic regression training."""
        trainer = ClassificationTrainer(ml_config)

        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        # Create and train model
        model = trainer.get_model_instance("logistic", {})
        trained_model = trainer.train(model, X_train, y_train)

        assert trained_model is not None
        assert hasattr(trained_model, 'coef_')

        # Make predictions
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_train_model_random_forest(self, ml_config, binary_classification_data):
        """Test random forest training."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        model = trainer.get_model_instance("random_forest", {"n_estimators": 10})
        trained_model = trainer.train(model, X_train, y_train)

        assert trained_model is not None
        assert hasattr(trained_model, 'feature_importances_')

        # Make predictions
        predictions = trained_model.predict(X_test)
        probabilities = trained_model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)  # Binary classification

    def test_train_model_multiclass(self, ml_config, multiclass_classification_data):
        """Test multiclass classification training."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            multiclass_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance("logistic", {})
        trained_model = trainer.train(model, X_train, y_train)

        # Make predictions
        predictions = trained_model.predict(X_test)
        probabilities = trained_model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probabilities.shape[1] == 3  # 3 classes

    def test_get_feature_importance_logistic(self, ml_config, binary_classification_data):
        """Test feature importance extraction from logistic regression."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance("logistic", {})
        trainer.train(model, X_train, y_train)

        feature_importance = trainer.get_feature_importance(
            model,
            ['feature1', 'feature2']
        )

        assert feature_importance is not None
        assert 'feature1' in feature_importance
        assert 'feature2' in feature_importance
        assert isinstance(feature_importance['feature1'], float)

    def test_get_feature_importance_tree(self, ml_config, binary_classification_data):
        """Test feature importance extraction from tree-based models."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2', 'feature3']
        )

        model = trainer.get_model_instance("random_forest", {"n_estimators": 10})
        trainer.train(model, X_train, y_train)

        feature_importance = trainer.get_feature_importance(
            model,
            ['feature1', 'feature2', 'feature3']
        )

        assert feature_importance is not None
        assert len(feature_importance) == 3
        # Feature importances should sum to approximately 1.0
        assert 0.9 < sum(feature_importance.values()) < 1.1

    def test_get_model_summary(self, ml_config, binary_classification_data):
        """Test model summary generation."""
        trainer = ClassificationTrainer(ml_config)

        X_train, X_test, y_train, y_test = trainer.prepare_data(
            binary_classification_data,
            target_column='target',
            feature_columns=['feature1', 'feature2']
        )

        model = trainer.get_model_instance("random_forest", {"n_estimators": 50, "max_depth": 10})
        trainer.train(model, X_train, y_train)

        summary = trainer.get_model_summary(
            model,
            "random_forest",
            ['feature1', 'feature2']
        )

        assert summary["model_type"] == "random_forest"
        assert summary["n_features"] == 2
        assert summary["n_estimators"] == 50
        assert summary["max_depth"] == 10
        assert summary["n_classes"] == 2
        assert "feature_importance" in summary

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        models = ClassificationTrainer.get_supported_models()
        assert len(models) == 6
        assert "logistic" in models
        assert "random_forest" in models
        assert "svm" in models
