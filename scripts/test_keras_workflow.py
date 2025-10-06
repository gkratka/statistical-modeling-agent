#!/usr/bin/env python3
"""
Keras Workflow Test Script.

Tests Keras neural network integration: Load CSV → Train Model → Save as JSON+H5
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


def create_sample_data(n_samples=1000, n_features=14):
    """Create sample dataset with n_features and binary target."""
    np.random.seed(7)
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    data["target"] = np.random.randint(0, 2, n_samples)
    return data


def create_test_architecture(initializer="random_normal"):
    """Create test architecture with specified kernel initializer."""
    return {
        "layers": [
            {"type": "Dense", "units": 14, "activation": "relu", "kernel_initializer": initializer},
            {"type": "Dense", "units": 1, "activation": "sigmoid", "kernel_initializer": initializer}
        ],
        "compile": {"loss": "binary_crossentropy", "optimizer": "adam", "metrics": ["accuracy"]}
    }


def test_single_model_training():
    """Test training a single Keras model."""
    print("=" * 70)
    print("TEST 1: Single Keras Model Training")
    print("=" * 70)

    data = create_sample_data(n_samples=500, n_features=14)
    print(f"✓ Created sample data: {data.shape}")

    config = MLEngineConfig.get_default()
    engine = MLEngine(config)
    print("✓ Initialized ML Engine\n✓ Defined model architecture")

    result = engine.train_model(
        data=data,
        task_type="classification",
        model_type="keras_binary_classification",
        target_column="target",
        feature_columns=[f"feature_{i}" for i in range(14)],
        user_id=12345,
        hyperparameters={"architecture": create_test_architecture(), "epochs": 10, "batch_size": 32, "verbose": 0},
        test_size=0.2
    )

    print(f"\n✓ Model trained successfully!")
    print(f"  Model ID: {result['model_id']}")
    print(f"  Accuracy: {result['metrics'].get('accuracy', 'N/A'):.4f}")
    print(f"  Loss: {result['metrics'].get('loss', 'N/A'):.4f}")

    # Verify model files
    model_dir = Path(config.models_dir) / f"user_12345" / result['model_id']
    assert (model_dir / "model.json").exists() and (model_dir / "model.weights.h5").exists() and (model_dir / "metadata.json").exists()
    print(f"\n✓ Model files verified:\n  - model.json\n  - model.weights.h5\n  - metadata.json\n")
    return True


def test_no_test_split_training():
    """Test training with test_size=0 (100% training data)."""
    print("\n" + "=" * 70)
    print("TEST 2: Training with test_size=0 (100% training data)")
    print("=" * 70)

    data = create_sample_data(n_samples=300, n_features=14)
    print(f"✓ Created sample data: {data.shape}")

    config = MLEngineConfig.get_default()
    engine = MLEngine(config)

    result = engine.train_model(
        data=data,
        task_type="classification",
        model_type="keras_binary_classification",
        target_column="target",
        feature_columns=[f"feature_{i}" for i in range(14)],
        user_id=12345,
        hyperparameters={"architecture": create_test_architecture("glorot_uniform"), "epochs": 10, "batch_size": 32, "verbose": 0},
        test_size=0.0
    )

    print(f"\n✓ Model trained on 100% of data!")
    print(f"  Model ID: {result['model_id']}")
    print(f"  Accuracy: {result['metrics'].get('accuracy', 'N/A'):.4f}")
    print(f"  Loss: {result['metrics'].get('loss', 'N/A'):.4f}\n")
    return True


def test_multi_model_training():
    """Test training multiple variants."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Model Training (8 Variants)")
    print("=" * 70)

    data = create_sample_data(n_samples=500, n_features=14)
    print(f"✓ Created sample data: {data.shape}\n")

    config = MLEngineConfig.get_default()
    engine = MLEngine(config)

    variants = [
        {"name": "RES_P1", "init": "random_normal", "epochs": 10, "batch": 32},
        {"name": "RES_P2", "init": "random_uniform", "epochs": 10, "batch": 32},
        {"name": "RES_P3", "init": "glorot_uniform", "epochs": 10, "batch": 32},
        {"name": "RES_P4", "init": "he_uniform", "epochs": 10, "batch": 32},
    ]

    results = []
    for i, variant in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] Training {variant['name']}...")
        print(f"  Initializer: {variant['init']}, Epochs: {variant['epochs']}, Batch: {variant['batch']}")

        result = engine.train_model(
            data=data,
            task_type="classification",
            model_type="keras_binary_classification",
            target_column="target",
            feature_columns=[f"feature_{i}" for i in range(14)],
            user_id=12345,
            hyperparameters={"architecture": create_test_architecture(variant['init']), "epochs": variant['epochs'], "batch_size": variant['batch'], "verbose": 0},
            test_size=0.2
        )

        results.append({"name": variant["name"], "model_id": result["model_id"], "accuracy": result["metrics"].get("accuracy", 0), "loss": result["metrics"].get("loss", 0)})
        print(f"  ✓ Accuracy: {result['metrics'].get('accuracy', 0):.4f}")
        print(f"  ✓ Saved as: {result['model_id']}\n")

    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:8} | Accuracy: {r['accuracy']:.4f} | Loss: {r['loss']:.4f}")

    print(f"\n✓ All {len(results)} models trained successfully!\n")
    return True


def test_model_loading():
    """Test loading and using a saved Keras model."""
    print("\n" + "=" * 70)
    print("TEST 4: Load and Predict with Saved Model")
    print("=" * 70)

    data = create_sample_data(n_samples=200, n_features=14)
    config = MLEngineConfig.get_default()
    engine = MLEngine(config)

    result = engine.train_model(
        data=data,
        task_type="classification",
        model_type="keras_binary_classification",
        target_column="target",
        feature_columns=[f"feature_{i}" for i in range(14)],
        user_id=12345,
        hyperparameters={"architecture": create_test_architecture(), "epochs": 5, "batch_size": 32, "verbose": 0},
        test_size=0.2
    )

    model_id = result["model_id"]
    print(f"✓ Trained model: {model_id}")

    # Create new data for prediction
    new_data = pd.DataFrame(
        np.random.randn(10, 14),
        columns=[f"feature_{i}" for i in range(14)]
    )

    # Predict
    predictions = engine.predict(user_id=12345, model_id=model_id, data=new_data)

    print(f"\n✓ Model loaded and predictions made!")
    print(f"  Number of predictions: {predictions['n_predictions']}")
    print(f"  Sample predictions: {predictions['predictions'][:3]}")
    print(f"  ✓ Predictions available: {len(predictions['predictions'])} samples\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("KERAS NEURAL NETWORK WORKFLOW TESTS")
    print("=" * 70)
    print("Testing Keras integration matching user's workflow:")
    print("  - Load CSV → Train Model → Save as JSON+H5")
    print("  - Support for test_size=0 (train on 100% of data)")
    print("  - Multi-variant training workflow")
    print("=" * 70)

    tests = [
        ("Single Model Training", test_single_model_training),
        ("No Test Split (test_size=0)", test_no_test_split_training),
        ("Multi-Model Training", test_multi_model_training),
        ("Model Loading & Prediction", test_model_loading),
    ]

    passed = sum(1 for _, test_func in tests if test_func())
    failed = len(tests) - passed

    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"✓ Passed: {passed}/{len(tests)}")
    print(f"✗ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Keras workflow is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
