#!/usr/bin/env python3
"""
Analyze prediction accuracy by comparing model predictions to ground truth.

Mapping:
- Prediction 0 → Ground Truth 1 (negative class)
- Prediction 1 → Ground Truth 2 (positive class)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load predictions and ground truth data."""
    project_root = Path(__file__).parent.parent

    # Load predictions (0/1)
    predictions_path = project_root / "data" / "german_credit_data_backtest_predicted.csv"
    predictions_df = pd.read_csv(predictions_path)

    # Load ground truth (1/2)
    ground_truth_path = project_root / "test_data" / "german_credit_data_backtest.csv"
    ground_truth_df = pd.read_csv(ground_truth_path)

    return predictions_df, ground_truth_df


def calculate_accuracy(predictions_df, ground_truth_df):
    """Calculate accuracy with label mapping."""

    # Extract predictions (0/1) and ground truth (1/2)
    predicted = predictions_df['class_predicted'].values
    actual = ground_truth_df['class'].values

    # Map predictions: 0 → 1, 1 → 2
    mapped_predictions = predicted + 1

    # Calculate matches
    matches = (mapped_predictions == actual)
    correct = np.sum(matches)
    total = len(actual)
    accuracy = (correct / total) * 100

    # Build confusion matrix
    # True Negatives: predicted 1 (0+1), actual 1
    tn = np.sum((mapped_predictions == 1) & (actual == 1))

    # False Positives: predicted 2 (1+1), actual 1
    fp = np.sum((mapped_predictions == 2) & (actual == 1))

    # False Negatives: predicted 1 (0+1), actual 2
    fn = np.sum((mapped_predictions == 1) & (actual == 2))

    # True Positives: predicted 2 (1+1), actual 2
    tp = np.sum((mapped_predictions == 2) & (actual == 2))

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        },
        'mapped_predictions': mapped_predictions,
        'actual': actual,
        'matches': matches
    }


def print_report(results):
    """Print detailed accuracy report."""

    print("=" * 70)
    print("MODEL PREDICTION ACCURACY ANALYSIS")
    print("=" * 70)
    print()

    print("LABEL MAPPING:")
    print("  Prediction 0 → Ground Truth 1 (negative class)")
    print("  Prediction 1 → Ground Truth 2 (positive class)")
    print()

    print("-" * 70)
    print("OVERALL ACCURACY")
    print("-" * 70)
    print(f"  Correct Predictions: {results['correct']}/{results['total']}")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print()

    # Confusion Matrix
    cm = results['confusion_matrix']
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(f"                    Predicted")
    print(f"                 Class 1  Class 2")
    print(f"  Actual Class 1    {cm['tn']:3d}      {cm['fp']:3d}     (True Neg, False Pos)")
    print(f"  Actual Class 2    {cm['fn']:3d}      {cm['tp']:3d}     (False Neg, True Pos)")
    print()

    # Per-class metrics
    print("-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)

    # Class 1 (Negative) metrics
    class_1_total = cm['tn'] + cm['fp']
    class_1_accuracy = (cm['tn'] / class_1_total * 100) if class_1_total > 0 else 0
    print(f"  Class 1 (Negative):")
    print(f"    Total samples: {class_1_total}")
    print(f"    Correctly predicted: {cm['tn']}")
    print(f"    Accuracy: {class_1_accuracy:.2f}%")
    print()

    # Class 2 (Positive) metrics
    class_2_total = cm['fn'] + cm['tp']
    class_2_accuracy = (cm['tp'] / class_2_total * 100) if class_2_total > 0 else 0
    print(f"  Class 2 (Positive):")
    print(f"    Total samples: {class_2_total}")
    print(f"    Correctly predicted: {cm['tp']}")
    print(f"    Accuracy: {class_2_accuracy:.2f}%")
    print()

    # Additional metrics
    print("-" * 70)
    print("CLASSIFICATION METRICS")
    print("-" * 70)

    # Precision for Class 2
    precision = (cm['tp'] / (cm['tp'] + cm['fp']) * 100) if (cm['tp'] + cm['fp']) > 0 else 0
    print(f"  Precision (Class 2): {precision:.2f}%")

    # Recall for Class 2
    recall = (cm['tp'] / (cm['tp'] + cm['fn']) * 100) if (cm['tp'] + cm['fn']) > 0 else 0
    print(f"  Recall (Class 2): {recall:.2f}%")

    # F1 Score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    print(f"  F1-Score (Class 2): {f1:.2f}%")
    print()

    # Show some misclassifications
    mismatches = ~results['matches']
    if np.sum(mismatches) > 0:
        print("-" * 70)
        print("SAMPLE MISCLASSIFICATIONS (First 10)")
        print("-" * 70)
        mismatched_indices = np.where(mismatches)[0][:10]
        print(f"  {'Row':<5} {'Predicted':<12} {'Actual':<10} {'Error Type':<15}")
        print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*15}")
        for idx in mismatched_indices:
            pred = results['mapped_predictions'][idx]
            act = results['actual'][idx]
            error_type = "False Positive" if pred == 2 and act == 1 else "False Negative"
            print(f"  {idx+1:<5} {pred:<12} {act:<10} {error_type:<15}")
        print()

    print("=" * 70)


def main():
    """Main execution function."""
    print("\nLoading data...")
    predictions_df, ground_truth_df = load_data()

    print(f"Loaded {len(predictions_df)} predictions")
    print(f"Loaded {len(ground_truth_df)} ground truth labels")
    print()

    print("Calculating accuracy...")
    results = calculate_accuracy(predictions_df, ground_truth_df)

    print()
    print_report(results)


if __name__ == "__main__":
    main()
