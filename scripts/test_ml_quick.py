#!/usr/bin/env python3
"""
Quick Smoke Test for ML Engine.

This script runs essential tests to quickly verify ML Engine is working.
Use this for rapid iteration during development.

Usage:
    python scripts/test_ml_quick.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def print_result(test_name, success, details=""):
    """Print test result."""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status} {test_name}{reset}")
    if details:
        print(f"  → {details}")


def main():
    """Run quick smoke tests."""
    print_header("ML ENGINE QUICK SMOKE TEST")

    # Initialize engine
    config = MLEngineConfig.get_default()
    engine = MLEngine(config)

    results = []

    # Test 1: Basic Regression
    print_header("Test 1: Basic Regression Training")
    try:
        data = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.randn(50)
        })

        result = engine.train_model(
            data=data,
            task_type="regression",
            model_type="linear",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=12345
        )

        r2 = result['metrics']['r2']
        model_id = result['model_id']

        print_result(
            "Regression Training",
            True,
            f"Model: {model_id}, R²: {r2:.3f}"
        )
        results.append(("Regression Training", True))

    except Exception as e:
        print_result("Regression Training", False, str(e))
        results.append(("Regression Training", False))

    # Test 2: Basic Classification
    print_header("Test 2: Basic Classification Training")
    try:
        data = pd.DataFrame({
            'x1': np.concatenate([np.random.randn(25) - 1, np.random.randn(25) + 1]),
            'x2': np.concatenate([np.random.randn(25) + 1, np.random.randn(25) - 1]),
            'y': [0] * 25 + [1] * 25
        })

        result = engine.train_model(
            data=data,
            task_type="classification",
            model_type="logistic",
            target_column="y",
            feature_columns=["x1", "x2"],
            user_id=12345
        )

        acc = result['metrics']['accuracy']
        print_result(
            "Classification Training",
            True,
            f"Accuracy: {acc:.3f}"
        )
        results.append(("Classification Training", True))

    except Exception as e:
        print_result("Classification Training", False, str(e))
        results.append(("Classification Training", False))

    # Test 3: Predictions
    print_header("Test 3: Model Predictions")
    try:
        # Train a simple model
        train_data = pd.DataFrame({
            'x': range(30),
            'y': [i * 2 for i in range(30)]
        })

        train_result = engine.train_model(
            data=train_data,
            task_type="regression",
            model_type="linear",
            target_column="y",
            feature_columns=["x"],
            user_id=99999
        )

        # Make predictions
        test_data = pd.DataFrame({'x': [5, 10, 15]})
        pred_result = engine.predict(
            user_id=99999,
            model_id=train_result['model_id'],
            data=test_data
        )

        preds = pred_result['predictions']
        print_result(
            "Predictions",
            True,
            f"Got {len(preds)} predictions: {preds}"
        )
        results.append(("Predictions", True))

    except Exception as e:
        print_result("Predictions", False, str(e))
        results.append(("Predictions", False))

    # Test 4: Model Management
    print_header("Test 4: Model Management")
    try:
        # List models
        models = engine.list_models(user_id=12345)
        print_result(
            "List Models",
            True,
            f"Found {len(models)} models for user 12345"
        )

        # Get model info
        if models:
            info = engine.get_model_info(
                user_id=12345,
                model_id=models[0]['model_id']
            )
            print_result(
                "Get Model Info",
                True,
                f"Retrieved info for {models[0]['model_id']}"
            )
            results.append(("Model Management", True))
        else:
            print_result("Get Model Info", False, "No models found")
            results.append(("Model Management", False))

    except Exception as e:
        print_result("Model Management", False, str(e))
        results.append(("Model Management", False))

    # Test 5: Error Handling
    print_header("Test 5: Error Handling")
    try:
        # Should raise error for empty data
        engine.train_model(
            data=pd.DataFrame(),
            task_type="regression",
            model_type="linear",
            target_column="y",
            feature_columns=["x"],
            user_id=12345
        )
        print_result("Error Handling", False, "Should have raised error for empty data")
        results.append(("Error Handling", False))

    except Exception:
        print_result(
            "Error Handling",
            True,
            "Correctly raised error for invalid data"
        )
        results.append(("Error Handling", True))

    # Summary
    print_header("SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    if passed == total:
        color = "\033[92m"  # Green
    elif passed >= total * 0.8:
        color = "\033[93m"  # Yellow
    else:
        color = "\033[91m"  # Red

    reset = '\033[0m'
    print(f"\n{color}Tests Passed: {passed}/{total}{reset}")

    if passed < total:
        print("\nFailed Tests:")
        for name, success in results:
            if not success:
                print(f"  ✗ {name}")

    print("\n" + "=" * 60)

    # Return exit code
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
