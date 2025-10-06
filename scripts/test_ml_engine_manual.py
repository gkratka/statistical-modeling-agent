#!/usr/bin/env python3
"""
Comprehensive Manual Testing Script for ML Engine.

This script runs through all phases of ML Engine testing to validate
functionality and help identify issues during development.

Usage:
    python scripts/test_ml_engine_manual.py              # Run all phases
    python scripts/test_ml_engine_manual.py --phase 1    # Run specific phase
    python scripts/test_ml_engine_manual.py --list       # List all phases
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import (
    DataValidationError,
    TrainingError,
    ValidationError,
    PredictionError
)


class TestRunner:
    """Manages test execution and reporting."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.config = MLEngineConfig.get_default()
        self.engine = MLEngine(self.config)

    def log_success(self, test_name: str, details: str = ""):
        """Log successful test."""
        result = {"test": test_name, "status": "✓ PASS", "details": details}
        self.results.append(result)
        print(f"✓ {test_name}: PASS")
        if details:
            print(f"  → {details}")

    def log_failure(self, test_name: str, error: str):
        """Log failed test."""
        result = {"test": test_name, "status": "✗ FAIL", "error": error}
        self.results.append(result)
        print(f"✗ {test_name}: FAIL")
        print(f"  → Error: {error}")

    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r["status"] == "✓ PASS")
        failed = sum(1 for r in self.results if r["status"] == "✗ FAIL")
        total = len(self.results)

        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {passed}/{total} passed, {failed}/{total} failed")
        print("=" * 60)

        if failed > 0:
            print("\nFailed Tests:")
            for r in self.results:
                if r["status"] == "✗ FAIL":
                    print(f"  • {r['test']}: {r['error']}")

    # ========================================================================
    # PHASE 1: Basic Regression Tests
    # ========================================================================

    def phase1_basic_regression(self):
        """Phase 1: Basic regression model training tests."""
        print("\n" + "=" * 60)
        print("PHASE 1: Basic Regression Tests")
        print("=" * 60)

        # Create test data
        data = pd.DataFrame({
            'age': [25, 30, 35, 28, 32, 40, 45, 22, 38, 42],
            'experience': [2, 5, 10, 3, 7, 15, 20, 1, 12, 17],
            'salary': [50000, 60000, 75000, 52000, 65000, 90000, 110000, 45000, 80000, 95000]
        })

        # Test 1.1: Linear Regression
        try:
            result = self.engine.train_model(
                data=data,
                task_type="regression",
                model_type="linear",
                target_column="salary",
                feature_columns=["age", "experience"],
                user_id=12345
            )

            # Validate result structure
            assert "model_id" in result, "Missing model_id"
            assert "metrics" in result, "Missing metrics"
            assert "r2" in result["metrics"], "Missing R² metric"
            assert "model_info" in result, "Missing model_info"

            r2 = result["metrics"]["r2"]
            self.log_success(
                "1.1 Linear Regression",
                f"R²={r2:.3f}, Model ID={result['model_id']}"
            )

        except Exception as e:
            self.log_failure("1.1 Linear Regression", str(e))

        # Test 1.2: Ridge Regression with Hyperparameters
        try:
            result = self.engine.train_model(
                data=data,
                task_type="regression",
                model_type="ridge",
                target_column="salary",
                feature_columns=["age", "experience"],
                user_id=12345,
                hyperparameters={"alpha": 0.5}
            )

            assert "model_id" in result
            assert result["model_info"]["model_type"] == "ridge"

            self.log_success(
                "1.2 Ridge Regression",
                f"Alpha=0.5, R²={result['metrics']['r2']:.3f}"
            )

        except Exception as e:
            self.log_failure("1.2 Ridge Regression", str(e))

        # Test 1.3: All Regression Models
        regression_models = ["linear", "ridge", "lasso", "elasticnet", "polynomial"]
        for model_type in regression_models:
            try:
                result = self.engine.train_model(
                    data=data,
                    task_type="regression",
                    model_type=model_type,
                    target_column="salary",
                    feature_columns=["age", "experience"],
                    user_id=12345
                )
                self.log_success(
                    f"1.3 {model_type.title()} Model",
                    f"R²={result['metrics']['r2']:.3f}"
                )
            except Exception as e:
                self.log_failure(f"1.3 {model_type.title()} Model", str(e))

    # ========================================================================
    # PHASE 2: Data Preprocessing Tests
    # ========================================================================

    def phase2_preprocessing(self):
        """Phase 2: Data preprocessing tests."""
        print("\n" + "=" * 60)
        print("PHASE 2: Data Preprocessing Tests")
        print("=" * 60)

        # Test 2.1: Missing Values Handling
        data_missing = pd.DataFrame({
            'age': [25, None, 35, 28, None, 40, 45, 50, 55, 60],
            'experience': [2, 5, None, 3, 7, 15, None, 18, 20, 22],
            'salary': [50000, 60000, 75000, 52000, 65000, 90000, 110000, 95000, 105000, 115000]
        })

        for strategy in ["mean", "median", "drop"]:
            try:
                result = self.engine.train_model(
                    data=data_missing,
                    task_type="regression",
                    model_type="linear",
                    target_column="salary",
                    feature_columns=["age", "experience"],
                    user_id=12345,
                    preprocessing_config={"missing_strategy": strategy}
                )
                self.log_success(
                    f"2.1 Missing Values ({strategy})",
                    f"R²={result['metrics']['r2']:.3f}"
                )
            except Exception as e:
                self.log_failure(f"2.1 Missing Values ({strategy})", str(e))

        # Test 2.2: Feature Scaling
        data_clean = pd.DataFrame({
            'age': [25, 30, 35, 28, 32, 40, 45, 22, 38, 42],
            'experience': [2, 5, 10, 3, 7, 15, 20, 1, 12, 17],
            'salary': [50000, 60000, 75000, 52000, 65000, 90000, 110000, 45000, 80000, 95000]
        })

        for scaling in ["standard", "minmax", "robust", "none"]:
            try:
                result = self.engine.train_model(
                    data=data_clean,
                    task_type="regression",
                    model_type="linear",
                    target_column="salary",
                    feature_columns=["age", "experience"],
                    user_id=12345,
                    preprocessing_config={"scaling": scaling}
                )
                self.log_success(
                    f"2.2 Scaling ({scaling})",
                    f"R²={result['metrics']['r2']:.3f}"
                )
            except Exception as e:
                self.log_failure(f"2.2 Scaling ({scaling})", str(e))

    # ========================================================================
    # PHASE 3: Classification Tests
    # ========================================================================

    def phase3_classification(self):
        """Phase 3: Classification model tests."""
        print("\n" + "=" * 60)
        print("PHASE 3: Classification Tests")
        print("=" * 60)

        # Binary classification data
        np.random.seed(42)
        data_binary = pd.DataFrame({
            'feature1': np.concatenate([np.random.randn(50) - 1, np.random.randn(50) + 1]),
            'feature2': np.concatenate([np.random.randn(50) + 1, np.random.randn(50) - 1]),
            'target': [0] * 50 + [1] * 50
        })

        # Test 3.1: Logistic Regression
        try:
            result = self.engine.train_model(
                data=data_binary,
                task_type="classification",
                model_type="logistic",
                target_column="target",
                feature_columns=["feature1", "feature2"],
                user_id=12345
            )

            assert "accuracy" in result["metrics"]
            assert "confusion_matrix" in result["metrics"]

            self.log_success(
                "3.1 Logistic Regression",
                f"Accuracy={result['metrics']['accuracy']:.3f}"
            )
        except Exception as e:
            self.log_failure("3.1 Logistic Regression", str(e))

        # Test 3.2: All Classification Models
        classification_models = [
            "logistic", "decision_tree", "random_forest",
            "gradient_boosting", "svm", "naive_bayes"
        ]

        for model_type in classification_models:
            try:
                result = self.engine.train_model(
                    data=data_binary,
                    task_type="classification",
                    model_type=model_type,
                    target_column="target",
                    feature_columns=["feature1", "feature2"],
                    user_id=12345
                )
                self.log_success(
                    f"3.2 {model_type.title()} Classifier",
                    f"Accuracy={result['metrics']['accuracy']:.3f}"
                )
            except Exception as e:
                self.log_failure(f"3.2 {model_type.title()} Classifier", str(e))

    # ========================================================================
    # PHASE 4: Model Lifecycle Tests
    # ========================================================================

    def phase4_model_lifecycle(self):
        """Phase 4: Model lifecycle management tests."""
        print("\n" + "=" * 60)
        print("PHASE 4: Model Lifecycle Tests")
        print("=" * 60)

        # Create test models
        data = pd.DataFrame({
            'x': range(20),
            'y': [i * 2 + np.random.randn() for i in range(20)]
        })

        model_ids = []
        for model_type in ["linear", "ridge"]:
            result = self.engine.train_model(
                data=data,
                task_type="regression",
                model_type=model_type,
                target_column="y",
                feature_columns=["x"],
                user_id=99999
            )
            model_ids.append(result["model_id"])

        # Test 4.1: List Models
        try:
            models = self.engine.list_models(user_id=99999)
            assert len(models) >= 2, f"Expected ≥2 models, got {len(models)}"

            self.log_success(
                "4.1 List Models",
                f"Found {len(models)} models"
            )
        except Exception as e:
            self.log_failure("4.1 List Models", str(e))

        # Test 4.2: Filter by Task Type
        try:
            regression_models = self.engine.list_models(
                user_id=99999,
                task_type="regression"
            )
            assert len(regression_models) >= 2

            self.log_success(
                "4.2 Filter by Task Type",
                f"Found {len(regression_models)} regression models"
            )
        except Exception as e:
            self.log_failure("4.2 Filter by Task Type", str(e))

        # Test 4.3: Get Model Info
        try:
            if model_ids:
                info = self.engine.get_model_info(user_id=99999, model_id=model_ids[0])
                assert "model_type" in info
                assert "metrics" in info

                self.log_success(
                    "4.3 Get Model Info",
                    f"Retrieved info for {model_ids[0]}"
                )
        except Exception as e:
            self.log_failure("4.3 Get Model Info", str(e))

        # Test 4.4: Delete Model
        try:
            if model_ids:
                before_count = len(self.engine.list_models(user_id=99999))
                self.engine.delete_model(user_id=99999, model_id=model_ids[0])
                after_count = len(self.engine.list_models(user_id=99999))

                assert after_count == before_count - 1

                self.log_success(
                    "4.4 Delete Model",
                    f"Deleted {model_ids[0]}, count: {before_count} → {after_count}"
                )
        except Exception as e:
            self.log_failure("4.4 Delete Model", str(e))

    # ========================================================================
    # PHASE 5: Prediction Tests
    # ========================================================================

    def phase5_predictions(self):
        """Phase 5: Model prediction tests."""
        print("\n" + "=" * 60)
        print("PHASE 5: Prediction Tests")
        print("=" * 60)

        # Train a model for prediction
        train_data = pd.DataFrame({
            'age': [25, 30, 35, 28, 32, 40, 45, 22, 38, 42],
            'experience': [2, 5, 10, 3, 7, 15, 20, 1, 12, 17],
            'salary': [50000, 60000, 75000, 52000, 65000, 90000, 110000, 45000, 80000, 95000]
        })

        train_result = self.engine.train_model(
            data=train_data,
            task_type="regression",
            model_type="linear",
            target_column="salary",
            feature_columns=["age", "experience"],
            user_id=88888
        )

        model_id = train_result["model_id"]

        # Test 5.1: Basic Prediction
        try:
            new_data = pd.DataFrame({
                'age': [27, 33, 45],
                'experience': [4, 8, 18]
            })

            predictions = self.engine.predict(
                user_id=88888,
                model_id=model_id,
                data=new_data
            )

            assert "predictions" in predictions
            assert len(predictions["predictions"]) == 3

            self.log_success(
                "5.1 Basic Prediction",
                f"Got {len(predictions['predictions'])} predictions"
            )
        except Exception as e:
            self.log_failure("5.1 Basic Prediction", str(e))

        # Test 5.2: Classification Probabilities
        try:
            # Train classifier
            class_data = pd.DataFrame({
                'x1': np.concatenate([np.random.randn(20) - 1, np.random.randn(20) + 1]),
                'x2': np.concatenate([np.random.randn(20) + 1, np.random.randn(20) - 1]),
                'y': [0] * 20 + [1] * 20
            })

            class_result = self.engine.train_model(
                data=class_data,
                task_type="classification",
                model_type="logistic",
                target_column="y",
                feature_columns=["x1", "x2"],
                user_id=88888
            )

            # Predict with probabilities
            test_data = pd.DataFrame({
                'x1': [0, 1, -1],
                'x2': [0, -1, 1]
            })

            predictions = self.engine.predict(
                user_id=88888,
                model_id=class_result["model_id"],
                data=test_data
            )

            assert "probabilities" in predictions
            assert len(predictions["probabilities"]) == 3

            self.log_success(
                "5.2 Classification Probabilities",
                f"Got probabilities for {len(predictions['probabilities'])} samples"
            )
        except Exception as e:
            self.log_failure("5.2 Classification Probabilities", str(e))

    # ========================================================================
    # PHASE 6: Error Handling Tests
    # ========================================================================

    def phase6_error_handling(self):
        """Phase 6: Error handling tests."""
        print("\n" + "=" * 60)
        print("PHASE 6: Error Handling Tests")
        print("=" * 60)

        # Test 6.1: Empty Data
        try:
            self.engine.train_model(
                data=pd.DataFrame(),
                task_type="regression",
                model_type="linear",
                target_column="y",
                feature_columns=["x"],
                user_id=12345
            )
            self.log_failure("6.1 Empty Data Error", "Should have raised DataValidationError")
        except DataValidationError:
            self.log_success("6.1 Empty Data Error", "Correctly raised DataValidationError")
        except Exception as e:
            self.log_failure("6.1 Empty Data Error", f"Wrong exception: {type(e).__name__}")

        # Test 6.2: Insufficient Samples
        try:
            small_data = pd.DataFrame({'x': [1, 2], 'y': [1, 2]})
            self.engine.train_model(
                data=small_data,
                task_type="regression",
                model_type="linear",
                target_column="y",
                feature_columns=["x"],
                user_id=12345
            )
            self.log_failure("6.2 Insufficient Samples", "Should have raised DataValidationError")
        except DataValidationError:
            self.log_success("6.2 Insufficient Samples", "Correctly raised DataValidationError")
        except Exception as e:
            self.log_failure("6.2 Insufficient Samples", f"Wrong exception: {type(e).__name__}")

        # Test 6.3: Invalid Task Type
        try:
            data = pd.DataFrame({'x': range(20), 'y': range(20)})
            self.engine.train_model(
                data=data,
                task_type="invalid_task",
                model_type="linear",
                target_column="y",
                feature_columns=["x"],
                user_id=12345
            )
            self.log_failure("6.3 Invalid Task Type", "Should have raised ValidationError")
        except ValidationError:
            self.log_success("6.3 Invalid Task Type", "Correctly raised ValidationError")
        except Exception as e:
            self.log_failure("6.3 Invalid Task Type", f"Wrong exception: {type(e).__name__}")

        # Test 6.4: Invalid Model Type
        try:
            data = pd.DataFrame({'x': range(20), 'y': range(20)})
            self.engine.train_model(
                data=data,
                task_type="regression",
                model_type="invalid_model",
                target_column="y",
                feature_columns=["x"],
                user_id=12345
            )
            self.log_failure("6.4 Invalid Model Type", "Should have raised ValidationError")
        except ValidationError:
            self.log_success("6.4 Invalid Model Type", "Correctly raised ValidationError")
        except Exception as e:
            self.log_failure("6.4 Invalid Model Type", f"Wrong exception: {type(e).__name__}")

    # ========================================================================
    # PHASE 7: Advanced Tests
    # ========================================================================

    def phase7_advanced(self):
        """Phase 7: Advanced functionality tests."""
        print("\n" + "=" * 60)
        print("PHASE 7: Advanced Tests")
        print("=" * 60)

        data = pd.DataFrame({
            'x': range(20),
            'y': [i * 2 + np.random.randn() for i in range(20)]
        })

        # Test 7.1: Multiple Models Per User
        try:
            user_id = 77777
            model_types = ["linear", "ridge", "lasso"]

            for model_type in model_types:
                self.engine.train_model(
                    data=data,
                    task_type="regression",
                    model_type=model_type,
                    target_column="y",
                    feature_columns=["x"],
                    user_id=user_id
                )

            models = self.engine.list_models(user_id=user_id)
            assert len(models) >= 3

            self.log_success(
                "7.1 Multiple Models Per User",
                f"User {user_id} has {len(models)} models"
            )
        except Exception as e:
            self.log_failure("7.1 Multiple Models Per User", str(e))

        # Test 7.2: Custom Test Size
        try:
            for test_size in [0.1, 0.2, 0.3]:
                result = self.engine.train_model(
                    data=data,
                    task_type="regression",
                    model_type="linear",
                    target_column="y",
                    feature_columns=["x"],
                    user_id=12345,
                    test_size=test_size
                )
                assert "metrics" in result

            self.log_success("7.2 Custom Test Size", "Test sizes 0.1, 0.2, 0.3 work")
        except Exception as e:
            self.log_failure("7.2 Custom Test Size", str(e))

        # Test 7.3: Get Supported Models
        try:
            for task_type in ["regression", "classification", "neural_network"]:
                models = self.engine.get_supported_models(task_type)
                assert len(models) > 0

            self.log_success(
                "7.3 Get Supported Models",
                f"All task types have supported models"
            )
        except Exception as e:
            self.log_failure("7.3 Get Supported Models", str(e))


def main():
    """Main test execution."""
    parser = argparse.ArgumentParser(description="ML Engine Manual Testing")
    parser.add_argument(
        "--phase",
        type=int,
        choices=range(1, 8),
        help="Run specific phase (1-7)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all test phases"
    )

    args = parser.parse_args()

    phases = {
        1: ("Basic Regression Tests", "phase1_basic_regression"),
        2: ("Data Preprocessing Tests", "phase2_preprocessing"),
        3: ("Classification Tests", "phase3_classification"),
        4: ("Model Lifecycle Tests", "phase4_model_lifecycle"),
        5: ("Prediction Tests", "phase5_predictions"),
        6: ("Error Handling Tests", "phase6_error_handling"),
        7: ("Advanced Tests", "phase7_advanced"),
    }

    if args.list:
        print("\nAvailable Test Phases:")
        print("=" * 60)
        for num, (name, _) in phases.items():
            print(f"  {num}. {name}")
        print("\nUsage: python scripts/test_ml_engine_manual.py --phase <number>")
        return

    runner = TestRunner()

    print("\n" + "=" * 60)
    print("ML ENGINE MANUAL TESTING")
    print("=" * 60)

    if args.phase:
        # Run specific phase
        name, method = phases[args.phase]
        print(f"\nRunning Phase {args.phase}: {name}")
        getattr(runner, method)()
    else:
        # Run all phases
        for num in sorted(phases.keys()):
            name, method = phases[num]
            getattr(runner, method)()

    runner.print_summary()


if __name__ == "__main__":
    main()
