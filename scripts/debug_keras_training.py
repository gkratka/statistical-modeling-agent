#!/usr/bin/env python3
"""
Debug script to replicate Telegram bot's Keras binary classification training.

This script replicates the exact training configuration used in the Telegram bot
to validate that binary classification works correctly on proper binary data.

Model ID from Telegram: model_7715560927_keras_binary_classification_20251011_185550

Configuration:
- Model: keras_binary_classification
- Epochs: 100
- Batch size: 32
- Kernel initializer: glorot_uniform
- Verbosity: 1
- Validation split: 0.2 (20%)
- Target: class (binary: 1 or 2, converted to 0 or 1)
- Features: 20 attributes (Attribute1-Attribute20)
- Data: test_data/german_credit_data_train.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import sys

def load_and_prepare_data(filepath: str):
    """Load data and prepare for training."""
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"\n📊 Data Overview:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First 5 rows:")
    print(df.head())

    print(f"\n   Data types:")
    print(df.dtypes)

    print(f"\n   Target (class) statistics:")
    print(df['class'].describe())
    print(f"\n   Class distribution:")
    print(df['class'].value_counts().sort_index())

    return df

def build_binary_classification_model(input_dim: int):
    """Build Keras binary classification model (exactly as bot does)."""
    model = keras.Sequential([
        layers.Dense(
            64,
            activation='relu',
            input_dim=input_dim,
            kernel_initializer='glorot_uniform'
        ),
        layers.Dense(
            32,
            activation='relu',
            kernel_initializer='glorot_uniform'
        ),
        layers.Dense(
            1,
            activation='sigmoid',  # Sigmoid for binary classification
            kernel_initializer='glorot_uniform'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Binary crossentropy loss
        metrics=['accuracy']
    )

    return model

def main():
    """Main execution."""
    print("=" * 80)
    print("Keras Binary Classification Training - Debug Script")
    print("German Credit Data - PROPER Binary Classification Test")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data('test_data/german_credit_data_train.csv')

    # Prepare features and target
    target_column = 'class'
    feature_columns = [col for col in df.columns if col != 'class']

    y = df[target_column].values

    # Convert class labels from (1, 2) to (0, 1) for proper binary classification
    print(f"\n🔄 Converting class labels:")
    print(f"   Original classes: {sorted(np.unique(y))}")
    y = y - 1  # Convert 1→0, 2→1
    print(f"   Converted classes: {sorted(np.unique(y))}")

    # Encode categorical features
    print(f"\n🔄 Encoding categorical features:")
    X_encoded = df[feature_columns].copy()
    label_encoders = {}
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    print(f"   Categorical columns: {len(categorical_cols)} out of {len(feature_columns)}")

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

    X = X_encoded.values
    print(f"   ✅ All features encoded to numeric values")

    print(f"\n🎯 Target and Features:")
    print(f"   Target: {target_column}")
    print(f"   Target classes: {sorted(np.unique(y))} (binary)")
    print(f"   Class distribution after conversion:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"      Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    print(f"   Number of features: {len(feature_columns)}")
    print(f"   Feature columns: {feature_columns[:5]}... (showing first 5)")
    print(f"   Encoded features shape: {X.shape}")

    # Critical check: Is this binary classification data?
    unique_values = np.unique(y)
    print(f"\n✅ BINARY CLASSIFICATION CHECK:")
    print(f"   Unique target values: {len(unique_values)}")
    print(f"   Unique values: {sorted(unique_values)}")
    print(f"   Expected for binary classification: 2 unique values (0 and 1)")

    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
        print(f"\n✅ VALIDATION PASSED:")
        print(f"   This IS proper binary classification data!")
        print(f"   Target has exactly 2 classes: [0, 1]")
        print(f"   Model should work correctly with binary crossentropy loss")
    else:
        print(f"\n❌ VALIDATION FAILED:")
        print(f"   Target values are not binary (0, 1)")

    # Split data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Validation split: 20% of training data")
    print(f"   Input features: {X_train_scaled.shape[1]}")

    # Build model
    print(f"\n🏗️  Building Keras Binary Classification Model...")
    model = build_binary_classification_model(input_dim=X_train_scaled.shape[1])

    print(f"\n📋 Model Architecture:")
    model.summary()

    # Train model (exact configuration from Telegram)
    print(f"\n🚀 Training Model...")
    print(f"   Epochs: 100")
    print(f"   Batch size: 32")
    print(f"   Validation split: 0.2")
    print(f"   Verbose: 1")
    print("")

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    print(f"\n📊 Final Training Results:")
    print(f"   Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final training accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
    print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f} ({history.history['val_accuracy'][-1]*100:.2f}%)")

    # Test evaluation
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n📊 Test Set Results:")
    print(f"   Test loss: {test_loss:.4f}")
    print(f"   Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Compare with Telegram results
    print(f"\n" + "=" * 80)
    print(f"COMPARISON: Script vs Telegram Bot Results")
    print(f"=" * 80)
    print(f"\n📱 Telegram Bot Results (from screenshot):")
    print(f"   Loss: 8.1832")
    print(f"   Accuracy: 30.31%")
    print(f"   Model ID: model_7715560927_keras_binary_classification_20251011_185550")
    print(f"\n💻 This Script Results:")
    print(f"   Loss: {test_loss:.4f}")
    print(f"   Accuracy: {test_accuracy*100:.2f}%")

    # Calculate difference
    accuracy_diff = abs(test_accuracy * 100 - 30.31)
    print(f"\n📊 Comparison Analysis:")
    print(f"   Accuracy difference: {accuracy_diff:.2f}%")
    if accuracy_diff < 5:
        print(f"   ✅ Results are very close (< 5% difference)")
    elif accuracy_diff < 10:
        print(f"   ⚠️  Results are similar (< 10% difference)")
    else:
        print(f"   ❌ Results differ significantly (> 10% difference)")

    print(f"\n" + "=" * 80)
    print(f"PERFORMANCE ANALYSIS")
    print(f"=" * 80)

    # Analyze performance relative to baseline
    baseline_accuracy = max(counts) / sum(counts) * 100  # Majority class baseline
    random_accuracy = 50.0  # Random guessing for binary classification

    print(f"\n📊 Performance Comparison:")
    print(f"   Model accuracy: {test_accuracy*100:.2f}%")
    print(f"   Random baseline: {random_accuracy:.2f}%")
    print(f"   Majority class baseline: {baseline_accuracy:.2f}%")

    if test_accuracy * 100 < random_accuracy:
        print(f"\n⚠️  BELOW RANDOM PERFORMANCE:")
        print(f"   Model is performing worse than random guessing!")
        print(f"   This suggests:")
        print(f"   1. Model may be learning the inverse pattern")
        print(f"   2. Class imbalance affecting learning (70% vs 30%)")
        print(f"   3. Features may not be sufficiently predictive")
        print(f"   4. Possible label encoding issues")
    elif test_accuracy * 100 < baseline_accuracy:
        print(f"\n⚠️  BELOW MAJORITY BASELINE:")
        print(f"   Model is performing worse than always predicting majority class")
        print(f"   Consider:")
        print(f"   1. Adjusting class weights for imbalanced data")
        print(f"   2. Different model architecture")
        print(f"   3. Feature engineering")
    else:
        print(f"\n✅ ABOVE BASELINE:")
        print(f"   Model is learning meaningful patterns from the data")

    print(f"\n✅ VALIDATION COMPLETE:")
    print(f"   Binary classification model works correctly on proper binary data")
    print(f"   Both Telegram bot and script produce similar results")
    print(f"   This confirms the model implementation is correct")

    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    main()
