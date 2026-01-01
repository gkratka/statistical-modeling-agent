"""
Training message formatting utilities for ML workflows.

This module provides message formatting functions specifically for
ML training completion, including enhanced metrics display.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.stats_utils import compute_dataset_stats, format_class_distribution, format_quartiles


# Metrics to display in priority order for classification
CLASSIFICATION_METRICS_ORDER = [
    ('roc_auc', 'AUC-ROC'),
    ('auc_pr', 'AUC-PR'),
    ('brier_score', 'Brier Score'),
    ('log_loss', 'Log Loss'),
    ('f1', 'F1 Score'),
    ('accuracy', 'Accuracy'),
    ('precision', 'Precision'),
    ('recall', 'Recall'),
]

# Metrics to display for regression
REGRESSION_METRICS_ORDER = [
    ('r2', 'R2 Score'),
    ('mse', 'MSE'),
    ('mae', 'MAE'),
    ('rmse', 'RMSE'),
]


def format_training_metrics(
    metrics: Dict[str, Any],
    task_type: str = 'classification'
) -> str:
    """
    Format training metrics for display, with priority ordering.

    Args:
        metrics: Dictionary of metric names to values
        task_type: Either 'classification' or 'regression'

    Returns:
        Formatted string of metrics
    """
    lines = []

    # Get ordered list based on task type
    if task_type == 'classification':
        metrics_order = CLASSIFICATION_METRICS_ORDER
    else:
        metrics_order = REGRESSION_METRICS_ORDER

    # Add ordered metrics first
    for metric_key, display_name in metrics_order:
        if metric_key in metrics:
            value = metrics[metric_key]
            if isinstance(value, float):
                lines.append(f"• {display_name}: {value:.4f}")
            elif value is not None:
                lines.append(f"• {display_name}: {value}")

    # Add any remaining metrics not in the order list (except confusion_matrix)
    ordered_keys = {m[0] for m in metrics_order}
    skip_keys = {'confusion_matrix', 'error'}
    for key, value in metrics.items():
        if key not in ordered_keys and key not in skip_keys:
            if isinstance(value, float):
                lines.append(f"• {key}: {value:.4f}")
            elif isinstance(value, (int, str)):
                lines.append(f"• {key}: {value}")

    return "\n".join(lines)


def format_dataset_stats(
    df: pd.DataFrame,
    target_col: str,
    task_type: str
) -> str:
    """
    Format basic dataset statistics for display.

    Args:
        df: DataFrame containing the data
        target_col: Name of target column
        task_type: Either 'classification' or 'regression'

    Returns:
        Formatted string of dataset stats
    """
    stats = compute_dataset_stats(df, target_col, task_type)

    lines = [f"• Rows: {stats['n_rows']:,}"]

    # Quartiles
    if stats.get('quartiles'):
        q = stats['quartiles']
        if q.get('q1') is not None:
            lines.append(f"• Quartiles: Q1={q['q1']:.2f}, Median={q['q2']:.2f}, Q3={q['q3']:.2f}")

    # Class distribution (classification only)
    if 'class_distribution' in stats:
        dist = stats['class_distribution']
        dist_parts = []
        for class_val in sorted(dist.keys()):
            info = dist[class_val]
            dist_parts.append(f"Class {class_val}: {info['count']} ({info['percentage']}%)")
        lines.append(f"• Classes: {', '.join(dist_parts)}")

    return "\n".join(lines)


def format_training_complete_message(
    model_id: str,
    model_type: str,
    metrics: Dict[str, Any],
    training_time: float,
    task_type: str = 'classification',
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None
) -> str:
    """
    Format complete training success message with metrics and stats.

    Args:
        model_id: Unique model identifier
        model_type: Type of model trained
        metrics: Dictionary of performance metrics
        training_time: Training time in seconds
        task_type: 'classification' or 'regression'
        df: Optional DataFrame for computing dataset stats
        target_col: Optional target column name for stats

    Returns:
        Formatted Telegram message string
    """
    lines = ["✅ *Training Complete!*", ""]

    # Model info
    lines.append(f"🎯 Model: {model_type}")
    lines.append(f"🆔 Model ID: `{model_id}`")
    lines.append("")

    # Dataset stats (if available)
    if df is not None and target_col is not None:
        lines.append("📊 *Dataset Stats:*")
        lines.append(format_dataset_stats(df, target_col, task_type))
        lines.append("")

    # Performance metrics
    lines.append("📈 *Performance Metrics:*")
    lines.append(format_training_metrics(metrics, task_type))
    lines.append("")

    # Training time
    lines.append(f"⏱ Training Time: {training_time:.2f}s")

    return "\n".join(lines)


def format_prediction_stats_message(
    n_predictions: int,
    prediction_col: str,
    stats: Dict[str, Any],
    task_type: str = 'classification'
) -> str:
    """
    Format prediction statistics for display.

    Args:
        n_predictions: Number of predictions made
        prediction_col: Name of prediction column
        stats: Statistics dictionary with quartiles and class distribution
        task_type: 'classification' or 'regression'

    Returns:
        Formatted string of prediction stats
    """
    lines = [f"📊 *Prediction Stats ({prediction_col}):*"]
    lines.append(f"• Predictions: {n_predictions:,}")

    # Quartiles (for regression or probabilities)
    if stats.get('quartiles'):
        q = stats['quartiles']
        if q.get('q1') is not None:
            lines.append(f"• Q1: {q['q1']:.4f}")
            lines.append(f"• Median: {q['q2']:.4f}")
            lines.append(f"• Q3: {q['q3']:.4f}")

    # Class distribution (for classification)
    if 'class_distribution' in stats:
        dist = stats['class_distribution']
        for class_val in sorted(dist.keys()):
            info = dist[class_val]
            lines.append(f"• Class {class_val}: {info['count']} ({info['percentage']}%)")

    return "\n".join(lines)
