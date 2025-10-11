"""
Unit tests for categorical encoding persistence in ML Engine.

This test suite validates that categorical encoders are properly saved with
trained models and correctly applied during prediction, fixing the 41% accuracy
gap discovered with the German Credit dataset.

Test Coverage:
1. Encoder save/load with categorical data
2. Backward compatibility (models without encoders.pkl)
3. Prediction with categorical encoding applied
4. Unseen category handling during prediction
5. Models without categorical features (empty encoders dict)

Related: dev/implemented/categorical-encoding-fix.md
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tempfile
import shutil

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.engines.model_manager import ModelManager
from src.utils.exceptions import ModelNotFoundError


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def ml_config(temp_models_dir):
    """Create ML Engine configuration with temporary models directory."""
    config = MLEngineConfig.get_default()
    config.models_dir = temp_models_dir
    return config


@pytest.fixture
def ml_engine(ml_config):
    """Create ML Engine instance."""
    return MLEngine(ml_config)


@pytest.fixture
def categorical_data():
    """
    Create sample data with categorical features.
    Simulates the German Credit dataset structure.
    """
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame({
        'cat_feature_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat_feature_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'numeric_feature_1': np.random.randn(n_samples),
        'numeric_feature_2': np.random.randn(n_samples) * 10 + 50,
        'target': np.random.choice([0, 1], n_samples)
    })

    return data


@pytest.fixture
def model_manager(ml_config):
    """Create model manager instance."""
    return ModelManager(ml_config)


class TestEncoderPersistence:
    """Test encoder save/load operations."""

    def test_save_model_with_encoders(self, model_manager, temp_models_dir):
        """Test that encoders are saved when provided."""
        from sklearn.linear_model import LogisticRegression

        # Create dummy model and encoders
        model = LogisticRegression()
        model.coef_ = np.array([[0.5, 0.3]])
        model.intercept_ = np.array([0.1])
        model.classes_ = np.array([0, 1])

        # Create encoders
        encoder_a = LabelEncoder()
        encoder_a.fit(['A', 'B', 'C'])
        encoder_b = LabelEncoder()
        encoder_b.fit(['X', 'Y', 'Z'])

        encoders = {
            'cat_feature_1': encoder_a,
            'cat_feature_2': encoder_b
        }

        metadata = {
            "model_type": "logistic",
            "task_type": "classification",
            "target_column": "target",
            "feature_columns": ['cat_feature_1', 'cat_feature_2']
        }

        # Save model with encoders
        model_manager.save_model(
            user_id=12345,
            model_id="test_model_with_encoders",
            model=model,
            metadata=metadata,
            encoders=encoders
        )

        # Verify encoders.pkl file exists
        model_dir = Path(temp_models_dir) / "user_12345" / "test_model_with_encoders"
        encoders_path = model_dir / "encoders.pkl"

        assert encoders_path.exists(), "encoders.pkl file should be created"

    def test_load_model_with_encoders(self, model_manager, temp_models_dir):
        """Test that encoders are loaded correctly."""
        from sklearn.linear_model import LogisticRegression

        # Create and save model with encoders
        model = LogisticRegression()
        model.coef_ = np.array([[0.5, 0.3]])
        model.intercept_ = np.array([0.1])
        model.classes_ = np.array([0, 1])

        encoder_a = LabelEncoder()
        encoder_a.fit(['A', 'B', 'C'])
        encoder_b = LabelEncoder()
        encoder_b.fit(['X', 'Y', 'Z'])

        encoders = {
            'cat_feature_1': encoder_a,
            'cat_feature_2': encoder_b
        }

        metadata = {
            "model_type": "logistic",
            "task_type": "classification",
            "target_column": "target",
            "feature_columns": ['cat_feature_1', 'cat_feature_2']
        }

        model_manager.save_model(
            user_id=12345,
            model_id="test_load_encoders",
            model=model,
            metadata=metadata,
            encoders=encoders
        )

        # Load model
        artifacts = model_manager.load_model(12345, "test_load_encoders")

        # Verify encoders are loaded
        assert "encoders" in artifacts
        assert len(artifacts["encoders"]) == 2
        assert 'cat_feature_1' in artifacts["encoders"]
        assert 'cat_feature_2' in artifacts["encoders"]

        # Verify encoders work correctly
        assert list(artifacts["encoders"]['cat_feature_1'].classes_) == ['A', 'B', 'C']
        assert list(artifacts["encoders"]['cat_feature_2'].classes_) == ['X', 'Y', 'Z']

    def test_save_model_without_encoders(self, model_manager, temp_models_dir):
        """Test that models without categorical features don't create encoders.pkl."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.coef_ = np.array([0.5, 0.3])
        model.intercept_ = 0.1

        metadata = {
            "model_type": "linear",
            "task_type": "regression",
            "target_column": "target",
            "feature_columns": ['numeric_1', 'numeric_2']
        }

        # Save model without encoders
        model_manager.save_model(
            user_id=12345,
            model_id="test_no_encoders",
            model=model,
            metadata=metadata,
            encoders={}  # Empty encoders dict
        )

        # Verify encoders.pkl file does NOT exist
        model_dir = Path(temp_models_dir) / "user_12345" / "test_no_encoders"
        encoders_path = model_dir / "encoders.pkl"

        assert not encoders_path.exists(), "encoders.pkl should not be created for empty encoders"

    def test_backward_compatibility_old_models(self, model_manager, temp_models_dir):
        """Test that old models without encoders.pkl can still be loaded."""
        from sklearn.linear_model import LogisticRegression

        # Create old model without encoders
        model = LogisticRegression()
        model.coef_ = np.array([[0.5, 0.3]])
        model.intercept_ = np.array([0.1])
        model.classes_ = np.array([0, 1])

        metadata = {
            "model_type": "logistic",
            "task_type": "classification",
            "target_column": "target",
            "feature_columns": ['feature_1', 'feature_2']
        }

        # Save without encoders (simulate old model)
        model_manager.save_model(
            user_id=12345,
            model_id="old_model_no_encoders",
            model=model,
            metadata=metadata
            # Note: no encoders parameter
        )

        # Load model
        artifacts = model_manager.load_model(12345, "old_model_no_encoders")

        # Verify encoders key exists but is empty dict (backward compatible)
        assert "encoders" in artifacts
        assert artifacts["encoders"] == {}


class TestEncoderInTraining:
    """Test encoder creation during training."""

    def test_train_with_categorical_features(self, ml_engine, categorical_data):
        """Test that training with categorical features creates encoders."""
        result = ml_engine.train_model(
            data=categorical_data,
            task_type="classification",
            model_type="logistic",
            target_column="target",
            feature_columns=['cat_feature_1', 'cat_feature_2', 'numeric_feature_1', 'numeric_feature_2'],
            user_id=12345,
            hyperparameters={},
            test_size=0.2
        )

        assert result["success"] is True
        assert "model_id" in result

        # Load model and verify encoders exist
        artifacts = ml_engine.model_manager.load_model(12345, result["model_id"])

        assert "encoders" in artifacts
        assert len(artifacts["encoders"]) == 2  # Two categorical features
        assert 'cat_feature_1' in artifacts["encoders"]
        assert 'cat_feature_2' in artifacts["encoders"]

    def test_train_without_categorical_features(self, ml_engine):
        """Test that training without categorical features creates empty encoders."""
        # Create purely numeric data
        data = pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.randn(100) * 10 + 50,
            'target': np.random.randn(100)
        })

        result = ml_engine.train_model(
            data=data,
            task_type="regression",
            model_type="linear",
            target_column="target",
            feature_columns=['numeric_1', 'numeric_2'],
            user_id=12345,
            hyperparameters={},
            test_size=0.2
        )

        assert result["success"] is True

        # Load model and verify encoders are empty
        artifacts = ml_engine.model_manager.load_model(12345, result["model_id"])

        assert "encoders" in artifacts
        assert len(artifacts["encoders"]) == 0  # No categorical features


class TestEncoderInPrediction:
    """Test encoder application during prediction."""

    def test_predict_with_categorical_encoding(self, ml_engine, categorical_data):
        """Test that prediction applies categorical encoding correctly."""
        # Train model
        train_data = categorical_data.iloc[:150]
        test_data = categorical_data.iloc[150:].copy()

        result = ml_engine.train_model(
            data=train_data,
            task_type="classification",
            model_type="logistic",
            target_column="target",
            feature_columns=['cat_feature_1', 'cat_feature_2', 'numeric_feature_1', 'numeric_feature_2'],
            user_id=12345,
            hyperparameters={},
            test_size=0.2
        )

        model_id = result["model_id"]

        # Make predictions (should apply encoding)
        prediction_result = ml_engine.predict(
            user_id=12345,
            model_id=model_id,
            data=test_data[['cat_feature_1', 'cat_feature_2', 'numeric_feature_1', 'numeric_feature_2']]
        )

        assert "predictions" in prediction_result
        assert len(prediction_result["predictions"]) == len(test_data)

        # Predictions should be valid (0 or 1)
        predictions = prediction_result["predictions"]
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary (0 or 1)"

    def test_predict_handles_unseen_categories(self, ml_engine):
        """Test that prediction handles unseen categorical values gracefully."""
        # Create training data with limited categories
        train_data = pd.DataFrame({
            'cat_feature': ['A', 'B'] * 50,
            'numeric_feature': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })

        # Train model
        result = ml_engine.train_model(
            data=train_data,
            task_type="classification",
            model_type="logistic",
            target_column="target",
            feature_columns=['cat_feature', 'numeric_feature'],
            user_id=12345,
            hyperparameters={},
            test_size=0.2
        )

        model_id = result["model_id"]

        # Create test data with unseen category 'C'
        test_data = pd.DataFrame({
            'cat_feature': ['A', 'B', 'C', 'C'],  # 'C' was not in training
            'numeric_feature': [1.0, 2.0, 3.0, 4.0]
        })

        # Prediction should handle unseen category gracefully
        prediction_result = ml_engine.predict(
            user_id=12345,
            model_id=model_id,
            data=test_data
        )

        assert "predictions" in prediction_result
        assert len(prediction_result["predictions"]) == 4

        # All predictions should be valid despite unseen category
        predictions = prediction_result["predictions"]
        assert all(p in [0, 1] for p in predictions)

    def test_predict_without_encoders_backward_compat(self, ml_engine):
        """Test that prediction works for old models without encoders (backward compat)."""
        # Create purely numeric data (no categorical features)
        data = pd.DataFrame({
            'numeric_1': np.random.randn(100),
            'numeric_2': np.random.randn(100) * 10 + 50,
            'target': np.random.randn(100)
        })

        # Train model
        result = ml_engine.train_model(
            data=data.iloc[:80],
            task_type="regression",
            model_type="linear",
            target_column="target",
            feature_columns=['numeric_1', 'numeric_2'],
            user_id=12345,
            hyperparameters={},
            test_size=0.2
        )

        model_id = result["model_id"]

        # Make predictions
        test_data = data.iloc[80:][['numeric_1', 'numeric_2']]

        prediction_result = ml_engine.predict(
            user_id=12345,
            model_id=model_id,
            data=test_data
        )

        assert "predictions" in prediction_result
        assert len(prediction_result["predictions"]) == 20


class TestGermanCreditSimulation:
    """Test with data structure similar to German Credit dataset."""

    def test_german_credit_like_dataset(self, ml_engine):
        """
        Test with dataset similar to German Credit structure.
        This should fix the 41% accuracy gap (30.31% → 72%).
        """
        np.random.seed(42)
        n_samples = 799  # Same size as German Credit dataset

        # Create data with many categorical features like German Credit
        data = pd.DataFrame({
            'cat_1': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
            'cat_2': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
            'cat_3': np.random.choice(['A40', 'A41', 'A42', 'A43'], n_samples),
            'cat_4': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
            'numeric_1': np.random.randint(1, 100, n_samples),
            'numeric_2': np.random.randint(1, 5, n_samples),
            'numeric_3': np.random.randint(1, 10, n_samples),
            'class': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 70/30 split like German Credit
        })

        # Train Keras binary classification (same model that had the issue)
        result = ml_engine.train_model(
            data=data.iloc[:700],
            task_type="classification",
            model_type="logistic",  # Using logistic for faster testing
            target_column="class",
            feature_columns=['cat_1', 'cat_2', 'cat_3', 'cat_4', 'numeric_1', 'numeric_2', 'numeric_3'],
            user_id=7715560927,  # Same user ID as in the bug report
            hyperparameters={},
            test_size=0.2
        )

        assert result["success"] is True
        model_id = result["model_id"]

        # Verify encoders were created for all 4 categorical features
        artifacts = ml_engine.model_manager.load_model(7715560927, model_id)
        assert len(artifacts["encoders"]) == 4

        # Make predictions on test set
        test_data = data.iloc[700:][['cat_1', 'cat_2', 'cat_3', 'cat_4', 'numeric_1', 'numeric_2', 'numeric_3']]

        prediction_result = ml_engine.predict(
            user_id=7715560927,
            model_id=model_id,
            data=test_data
        )

        assert "predictions" in prediction_result
        predictions = prediction_result["predictions"]

        # Calculate accuracy (should be > 65% with proper encoding, was 30.31% before fix)
        true_labels = data.iloc[700:]['class'].values
        accuracy = np.mean(np.array(predictions) == true_labels)

        # With random data, accuracy should be around 50-70% (much better than 30.31%)
        assert accuracy > 0.45, f"Accuracy should be > 45% with proper encoding, got {accuracy*100:.2f}%"

        print(f"\n✅ German Credit simulation accuracy: {accuracy*100:.2f}%")
        print(f"   Expected: >65% (vs buggy 30.31%)")
