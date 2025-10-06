#!/usr/bin/env python3
"""
Backtest Keras Model on German Credit Holdout Data.

This script loads the trained Keras model and evaluates it on the
holdout backtest dataset (200 samples).

Usage:
    python scripts/backtest_keras_german_credit.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from tensorflow import keras

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print('=' * 70)


def load_and_preprocess_backtest_data(csv_path):
    """
    Load and preprocess backtest data using same pipeline as training.

    Args:
        csv_path: Path to backtest CSV file

    Returns:
        X_test, y_test: Preprocessed features and labels
    """
    print_header("LOADING BACKTEST DATA")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Separate features and target
    X = df.drop('class', axis=1)
    y = df['class']

    print(f"\nTarget distribution:")
    print(y.value_counts())

    # Convert class labels (1,2) to binary (0,1)
    y = y.map({1: 0, 2: 1})
    print(f"\nBinary target distribution:")
    print(y.value_counts())

    # One-hot encode all categorical features (same as training)
    print(f"\nOriginal features: {X.shape[1]}")
    X_encoded = pd.get_dummies(X, drop_first=False)
    print(f"After one-hot encoding: {X_encoded.shape[1]} features")

    # Convert to numpy arrays with proper dtypes
    X_test = X_encoded.astype('float32').values
    y_test = y.astype('float32').values

    print(f"\nBacktest set: {len(X_test)} samples")

    return X_test, y_test


def evaluate_backtest(model, X_test, y_test):
    """
    Evaluate model on backtest data.

    Args:
        model: Loaded Keras model
        X_test, y_test: Backtest data

    Returns:
        Dictionary of metrics
    """
    print_header("BACKTEST EVALUATION")

    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    test_loss = log_loss(y_test, y_pred_proba)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nBacktest Metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss: {test_loss:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    return {
        'accuracy': float(test_acc),
        'loss': float(test_loss),
        'confusion_matrix': cm.tolist(),
        'y_pred': y_pred.tolist(),
        'y_true': y_test.tolist()
    }


def main():
    """Main execution function."""
    print_header("KERAS MODEL BACKTESTING")

    # Paths
    model_path = Path(__file__).parent.parent / 'models' / 'german_credit_keras_20251004_142150.h5'
    backtest_csv = Path(__file__).parent.parent / 'test_data' / 'german_credit_data_backtest.csv'

    # Verify files exist
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    if not backtest_csv.exists():
        print(f"Error: Backtest data not found at {backtest_csv}")
        return 1

    try:
        # Load trained model
        print_header("LOADING TRAINED MODEL")
        print(f"\nModel path: {model_path}")
        model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully")

        print("\nModel Summary:")
        model.summary()

        # Load and preprocess backtest data
        X_test, y_test = load_and_preprocess_backtest_data(backtest_csv)

        # Evaluate on backtest
        metrics = evaluate_backtest(model, X_test, y_test)

        # Print summary
        print_header("BACKTEST SUMMARY")
        print(f"""
Backtest Results:
- Dataset: german_credit_data_backtest.csv (200 samples)
- Accuracy: {metrics['accuracy']:.4f} (74.37% on validation)
- Loss: {metrics['loss']:.4f} (0.5135 on validation)

Confusion Matrix:
{metrics['confusion_matrix']}

Validation Reference (from training):
- Validation Accuracy: 0.7437
- Validation Loss: 0.5135

Performance Comparison:
- Accuracy Difference: {(metrics['accuracy'] - 0.7437):.4f} ({(metrics['accuracy'] - 0.7437) * 100:+.2f}%)
- Loss Difference: {(metrics['loss'] - 0.5135):.4f} ({(metrics['loss'] - 0.5135) * 100:+.2f}%)
""")

        print_header("SUCCESS")
        print("\n✓ Backtesting complete!")

        return 0

    except Exception as e:
        print_header("ERROR")
        print(f"\n✗ Backtesting failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
