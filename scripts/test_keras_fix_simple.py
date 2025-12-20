"""Simple test for Keras input_shape fix (no model saving required)."""

import sys
sys.path.insert(0, '/Users/gkratka/Documents/statistical-modeling-agent')

import numpy as np
import pandas as pd
from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.keras_templates import get_template
from src.utils.exceptions import ValidationError


def test_keras_model_building():
    """Test that Keras models can be built with correct input shape."""
    print("\n" + "="*80)
    print("TEST: Keras Model Building with 20 Features")
    print("="*80)

    n_features = 20

    # Get architecture
    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=n_features
    )

    # Build hyperparameters
    hyperparameters = {
        "architecture": architecture,
        "n_features": n_features
    }

    # Create trainer
    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    print(f"\n✓ Testing with n_features={n_features} (type: {type(n_features).__name__})")

    try:
        # Build model
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )

        print(f"✅ Model built successfully!")
        print(f"  - Total layers: {len(model.layers)}")
        print(f"  - Input layer: {model.layers[0].__class__.__name__}")
        print(f"  - First Dense layer: {model.layers[1].__class__.__name__} (units={model.layers[1].units})")
        print(f"  - Output layer: {model.layers[-1].__class__.__name__} (units={model.layers[-1].units})")

        # Verify model can predict
        test_input = np.random.randn(10, n_features)
        predictions = model.predict(test_input, verbose=0)

        print(f"\n✓ Model can make predictions:")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {predictions.shape}")
        print(f"  - Sample predictions: {predictions[:3].flatten()}")

        return True

    except ValueError as e:
        if "'layers'" in str(e) or "Cannot convert" in str(e):
            print(f"\n❌ CRITICAL: Original bug still present!")
            print(f"Error: {e}")
            return False
        else:
            print(f"\n❌ ValueError: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_rejects_tuple():
    """Test that validation rejects tuple n_features."""
    print("\n" + "="*80)
    print("TEST: Validation Rejects Tuple n_features")
    print("="*80)

    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=20
    )

    # Simulate bug: n_features as tuple
    hyperparameters = {
        "architecture": architecture,
        "n_features": (20, 'layers')  # BUG!
    }

    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    print(f"\n✓ Testing with n_features=(20, 'layers') (type: tuple)")

    try:
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )
        print("\n❌ FAILED: Should have raised ValidationError")
        return False

    except ValidationError as e:
        print(f"\n✅ Validation error raised as expected!")
        print(f"  Error message: {str(e)[:100]}...")
        return True

    except Exception as e:
        print(f"\n❌ Wrong error type: {type(e).__name__}")
        print(f"  Error: {e}")
        return False


def test_various_feature_counts():
    """Test with different feature counts."""
    print("\n" + "="*80)
    print("TEST: Various Feature Counts")
    print("="*80)

    feature_counts = [1, 5, 10, 20, 50, 100]
    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    all_passed = True

    for n_features in feature_counts:
        architecture = get_template(
            model_type="keras_binary_classification",
            n_features=n_features
        )

        hyperparameters = {
            "architecture": architecture,
            "n_features": n_features
        }

        try:
            model = trainer.get_model_instance(
                model_type="keras_binary_classification",
                hyperparameters=hyperparameters
            )
            print(f"  ✅ n_features={n_features:3d} - Model built successfully")

        except Exception as e:
            print(f"  ❌ n_features={n_features:3d} - Error: {e}")
            all_passed = False

    if all_passed:
        print(f"\n✅ All feature counts worked!")

    return all_passed


if __name__ == "__main__":
    print("\n" + "="*80)
    print("KERAS INPUT_SHAPE FIX - VALIDATION TESTS")
    print("="*80)

    results = []

    # Test 1: Build model with 20 features
    results.append(("Model Building", test_keras_model_building()))

    # Test 2: Validation rejects tuple
    results.append(("Validation", test_validation_rejects_tuple()))

    # Test 3: Various feature counts
    results.append(("Feature Counts", test_various_feature_counts()))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nKEY FIXES:")
        print("1. Added Input layer to model building (modern Keras best practice)")
        print("2. Removed deprecated input_dim parameter")
        print("3. Added validation to ensure n_features is integer, not tuple")
        print("4. Clear error messages when invalid n_features is provided")
        print("\nThe error 'Cannot convert (20, layers) to a shape' is now:")
        print("  - Prevented by validation at the entry point")
        print("  - Clear error message guides user to fix")
        print("  - Model builds correctly with proper integer n_features")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
