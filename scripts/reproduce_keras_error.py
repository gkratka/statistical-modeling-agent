"""Script to reproduce Keras input_shape error.

Attempts to trigger: ValueError: Cannot convert '(20, 'layers')' to a shape.
"""

import sys
sys.path.insert(0, '/Users/gkratka/Documents/statistical-modeling-agent')

import numpy as np
import pandas as pd
from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
from src.engines.ml_config import MLEngineConfig
from src.engines.trainers.keras_templates import get_template


def test_scenario_1():
    """Test with correctly structured hyperparameters."""
    print("\n=== Scenario 1: Correct hyperparameters ===")

    n_features = 20
    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=n_features
    )

    hyperparameters = {
        "architecture": architecture,
        "n_features": n_features
    }

    print(f"hyperparameters keys: {hyperparameters.keys()}")
    print(f"n_features type: {type(hyperparameters['n_features'])}")
    print(f"n_features value: {hyperparameters['n_features']}")

    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    try:
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )
        print("✅ Model created successfully")
        print(f"Model layers: {len(model.layers)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_scenario_2():
    """Test with n_features as tuple (bug scenario)."""
    print("\n=== Scenario 2: n_features as tuple (BUG) ===")

    n_features = (20, 'layers')  # THIS IS THE BUG!
    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=20  # Template needs integer
    )

    hyperparameters = {
        "architecture": architecture,
        "n_features": n_features  # But this is tuple!
    }

    print(f"hyperparameters keys: {hyperparameters.keys()}")
    print(f"n_features type: {type(hyperparameters['n_features'])}")
    print(f"n_features value: {hyperparameters['n_features']}")

    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    try:
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )
        print("✅ Model created successfully (unexpected!)")
    except ValueError as e:
        if "'layers'" in str(e) or "Cannot convert" in str(e):
            print(f"✅ REPRODUCED THE BUG! Error: {e}")
        else:
            print(f"❌ Different error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        import traceback
        traceback.print_exc()


def test_scenario_3():
    """Test where n_features might come from shape attribute."""
    print("\n=== Scenario 3: n_features from DataFrame.shape ===")

    # Simulate getting n_features from DataFrame
    X = pd.DataFrame(np.random.randn(100, 20))

    # BUG: If someone did this:
    # n_features = (X.shape[1], 'layers')  # Wrong!
    # Instead of:
    # n_features = X.shape[1]  # Correct

    n_features_wrong = (X.shape[1], 'layers')
    n_features_correct = X.shape[1]

    print(f"X.shape = {X.shape}")
    print(f"Correct n_features: {n_features_correct} (type: {type(n_features_correct)})")
    print(f"Wrong n_features: {n_features_wrong} (type: {type(n_features_wrong)})")

    # This would cause the error
    architecture = get_template(
        model_type="keras_binary_classification",
        n_features=20
    )

    hyperparameters = {
        "architecture": architecture,
        "n_features": n_features_wrong  # BUG!
    }

    config = MLEngineConfig.get_default()
    trainer = KerasNeuralNetworkTrainer(config)

    try:
        model = trainer.get_model_instance(
            model_type="keras_binary_classification",
            hyperparameters=hyperparameters
        )
        print("Model created (unexpected)")
    except Exception as e:
        print(f"❌ Error (as expected): {e}")


if __name__ == "__main__":
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
