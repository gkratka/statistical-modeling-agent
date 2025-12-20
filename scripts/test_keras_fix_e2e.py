"""End-to-end test for Keras input_shape fix.

Tests that Keras binary classification can train successfully with the fix.
"""

import sys
sys.path.insert(0, '/Users/gkratka/Documents/statistical-modeling-agent')

import numpy as np
import pandas as pd
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.keras_templates import get_template


def test_keras_training_with_20_features():
    """Test full Keras training workflow with 20 features (reproduces error scenario)."""
    print("\n" + "="*80)
    print("Testing Keras Binary Classification Training with 20 Features")
    print("="*80)

    # Create sample data with 20 features
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    # Generate random features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate binary target
    y = pd.Series(np.random.randint(0, 2, n_samples), name='target')

    # Combine into single dataframe
    data = pd.concat([X, y], axis=1)

    print(f"\n✓ Created dataset: {n_samples} samples × {n_features} features")
    print(f"✓ Target distribution: {y.value_counts().to_dict()}")

    # Get architecture template
    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=n_features
    )

    print(f"\n✓ Generated architecture:")
    print(f"  - Layers: {len(architecture['layers'])}")
    print(f"  - First layer units: {architecture['layers'][0]['units']}")
    print(f"  - Output layer: {architecture['layers'][-1]['units']} unit(s) with {architecture['layers'][-1]['activation']} activation")

    # Build hyperparameters
    hyperparameters = {
        "architecture": architecture,
        "n_features": n_features,  # CRITICAL: This must be an integer
        "epochs": 10,  # Small for testing
        "batch_size": 32,
        "verbose": 0,
        "validation_split": 0.2
    }

    print(f"\n✓ Hyperparameters configured:")
    print(f"  - n_features type: {type(hyperparameters['n_features']).__name__}")
    print(f"  - n_features value: {hyperparameters['n_features']}")
    print(f"  - epochs: {hyperparameters['epochs']}")

    # Initialize ML Engine
    config = MLEngineConfig.get_default()
    ml_engine = MLEngine(config)

    print(f"\n⚙️  Starting model training...")

    try:
        # Train model
        result = ml_engine.train_model(
            data=data,
            task_type='neural_network',
            model_type='keras_binary_classification',
            target_column='target',
            feature_columns=[f'feature_{i}' for i in range(n_features)],
            user_id=99999,  # Test user
            hyperparameters=hyperparameters,
            test_size=0.2
        )

        if result['success']:
            print(f"\n✅ SUCCESS! Model trained successfully")
            print(f"\n📊 Training Results:")
            print(f"  - Model ID: {result['model_id']}")
            print(f"  - Training time: {result.get('training_time', 'N/A'):.2f}s")

            metrics = result.get('metrics', {})
            print(f"\n📈 Model Metrics:")
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")

            print(f"\n🎯 The fix is working correctly!")
            print(f"  - No ValueError about shape")
            print(f"  - No 'Cannot convert (20, layers)' error")
            print(f"  - Model built using modern Input layer approach")

            return True
        else:
            print(f"\n❌ FAILED: Training returned success=False")
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False

    except ValueError as e:
        if "'layers'" in str(e) or "Cannot convert" in str(e):
            print(f"\n❌ CRITICAL ERROR: The original bug is still present!")
            print(f"Error: {e}")
            return False
        else:
            print(f"\n❌ ValueError (different issue): {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_catches_tuple_bug():
    """Test that validation now catches the tuple bug early."""
    print("\n" + "="*80)
    print("Testing Validation Catches n_features Tuple Bug")
    print("="*80)

    from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
    from src.utils.exceptions import ValidationError

    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=20
    )

    # Simulate the bug: n_features as tuple
    hyperparameters = {
        "architecture": architecture,
        "n_features": (20, 'layers')  # BUG!
    }

    try:
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )
        print("\n❌ FAILED: Should have raised ValidationError")
        return False

    except ValidationError as e:
        print(f"\n✅ SUCCESS: Validation error caught correctly")
        print(f"Error message: {e}")
        return True

    except Exception as e:
        print(f"\n❌ Wrong error type: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KERAS INPUT_SHAPE FIX - END-TO-END VALIDATION")
    print("="*80)

    results = []

    # Test 1: Full training workflow
    print("\n" + "="*80)
    print("TEST 1: Full Training Workflow")
    results.append(("Full Training", test_keras_training_with_20_features()))

    # Test 2: Validation catches bug
    print("\n" + "="*80)
    print("TEST 2: Validation Catches Tuple Bug")
    results.append(("Validation", test_validation_catches_tuple_bug()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - FIX IS WORKING CORRECTLY!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED - FIX NEEDS REVIEW")
        print("="*80)
        sys.exit(1)
