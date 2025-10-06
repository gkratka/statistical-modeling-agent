#!/usr/bin/env python3
"""
Keras Training Script for German Credit Data.

This script trains a binary classification model on the German Credit dataset
using the same architecture as the Telegram Keras workflow for comparison.

Usage:
    python scripts/train_keras_german_credit.py

Outputs:
    - Trained model saved to models/
    - Training metrics and results
    - Detailed comparison report
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def print_header(text):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print('=' * 70)


def load_and_preprocess_data(csv_path):
    """
    Load German credit data and preprocess.

    Args:
        csv_path: Path to CSV file

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print_header("LOADING DATA")

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

    # One-hot encode all categorical features
    print(f"\nOriginal features: {X.shape[1]}")
    X_encoded = pd.get_dummies(X, drop_first=False)
    print(f"After one-hot encoding: {X_encoded.shape[1]} features")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Convert to numpy arrays with proper dtypes
    X_train = X_train.astype('float32').values
    X_test = X_test.astype('float32').values
    y_train = y_train.astype('float32').values
    y_test = y_test.astype('float32').values

    return X_train, X_test, y_train, y_test, X_encoded.columns.tolist()


def build_keras_model(n_features):
    """
    Build Keras binary classification model using default template.

    This matches the architecture from src/engines/trainers/keras_templates.py

    Args:
        n_features: Number of input features

    Returns:
        Compiled Keras model
    """
    print_header("BUILDING MODEL")

    # Architecture matching keras_templates.py binary classification
    model = keras.Sequential([
        layers.Dense(
            n_features,
            activation='relu',
            kernel_initializer='random_normal',
            name='hidden_layer'
        ),
        layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='random_normal',
            name='output_layer'
        )
    ])

    # Compile (matching template)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print("\nModel Architecture:")
    model.summary()

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the Keras model.

    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training history
    """
    print_header("TRAINING MODEL")

    print(f"\nTraining parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation: Test set ({len(X_test)} samples)")

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and generate detailed metrics.

    Args:
        model: Trained Keras model
        X_test, y_test: Test data

    Returns:
        Dictionary of evaluation metrics
    """
    print_header("EVALUATION RESULTS")

    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Metrics:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # Return metrics dict
    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'prediction_probabilities': y_pred_proba.flatten().tolist()
    }


def save_results(model, history, metrics, feature_names, output_dir='models'):
    """
    Save model and results for comparison.

    Args:
        model: Trained Keras model
        history: Training history
        metrics: Evaluation metrics
        feature_names: List of feature names
        output_dir: Output directory
    """
    print_header("SAVING RESULTS")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_file = output_path / f"german_credit_keras_{timestamp}.h5"
    model.save(model_file)
    print(f"\nModel saved to: {model_file}")

    # Prepare results
    results = {
        'timestamp': timestamp,
        'dataset': 'german_credit_data.csv',
        'model_type': 'keras_binary_classification',
        'architecture': {
            'layers': [
                {
                    'type': 'Dense',
                    'units': len(feature_names),
                    'activation': 'relu',
                    'kernel_initializer': 'random_normal'
                },
                {
                    'type': 'Dense',
                    'units': 1,
                    'activation': 'sigmoid',
                    'kernel_initializer': 'random_normal'
                }
            ],
            'compile': {
                'loss': 'binary_crossentropy',
                'optimizer': 'adam',
                'metrics': ['accuracy']
            }
        },
        'hyperparameters': {
            'epochs': len(history.history['loss']),
            'batch_size': 32,
            'n_features': len(feature_names)
        },
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        },
        'final_metrics': metrics,
        'feature_names': feature_names,
        'random_seed': RANDOM_SEED
    }

    # Save results as JSON
    results_file = output_path / f"german_credit_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Print comparison summary
    print_header("COMPARISON SUMMARY")
    print(f"""
This model can be compared with Telegram Keras workflow results:

Expected Telegram Workflow:
1. Upload: german_credit_data.csv
2. Start: /train
3. Select target: 'class' (column 21)
4. Select features: All columns 1-20
5. Model type: keras_binary_classification
6. Architecture: Default template (option 1)
7. Epochs: 50
8. Batch size: 32

Comparison Metrics:
- Test Accuracy: {metrics['test_accuracy']:.4f}
- Test Loss: {metrics['test_loss']:.4f}

Note: Results may differ slightly due to:
- Data preprocessing differences
- Random weight initialization
- Training order variations
""")

    return results


def main():
    """Main execution function."""
    print_header("KERAS GERMAN CREDIT TRAINING")
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    csv_path = Path(__file__).parent.parent / 'test_data' / 'german_credit_data_train.csv'

    if not csv_path.exists():
        print(f"Error: Dataset not found at {csv_path}")
        return 1

    try:
        # Step 1: Load and preprocess
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(csv_path)

        # Step 2: Build model
        model = build_keras_model(n_features=len(feature_names))

        # Step 3: Train
        history = train_model(
            model, X_train, y_train, X_test, y_test,
            epochs=50, batch_size=32
        )

        # Step 4: Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Step 5: Save results
        results = save_results(model, history, metrics, feature_names)

        print_header("SUCCESS")
        print(f"\n✓ Training complete!")
        print(f"✓ Final test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"\nResults saved for Telegram workflow comparison.")

        return 0

    except Exception as e:
        print_header("ERROR")
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
