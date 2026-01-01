"""
Statistics utilities for ML workflows.

Provides functions for computing dataset statistics including
row counts, quartiles, and class distributions.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


def compute_dataset_stats(
    df: pd.DataFrame,
    target_col: str,
    task_type: str
) -> Dict[str, Any]:
    """
    Compute basic statistics for a dataset.

    Args:
        df: DataFrame containing the data
        target_col: Name of the target/prediction column
        task_type: Either 'classification' or 'regression'

    Returns:
        Dictionary containing:
            - n_rows: Number of rows
            - quartiles: Dict with q1, q2 (median), q3
            - class_distribution: Dict mapping class -> {count, percentage}
                                 (only for classification)

    Raises:
        KeyError: If target_col not in dataframe
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in dataframe")

    stats: Dict[str, Any] = {}

    # Row count
    stats['n_rows'] = len(df)

    # Handle empty dataframe
    if len(df) == 0:
        stats['quartiles'] = {'q1': None, 'q2': None, 'q3': None}
        return stats

    # Quartiles
    target_values = df[target_col]
    stats['quartiles'] = {
        'q1': float(target_values.quantile(0.25)),
        'q2': float(target_values.quantile(0.50)),  # Median
        'q3': float(target_values.quantile(0.75))
    }

    # Class distribution (classification only)
    if task_type == 'classification':
        value_counts = target_values.value_counts()
        total = len(df)

        class_distribution = {}
        for class_val in sorted(value_counts.index):
            count = int(value_counts[class_val])
            percentage = (count / total) * 100 if total > 0 else 0.0
            class_distribution[class_val] = {
                'count': count,
                'percentage': round(percentage, 1)
            }

        stats['class_distribution'] = class_distribution

    return stats


def format_class_distribution(class_dist: Dict[Any, Dict[str, Any]]) -> str:
    """
    Format class distribution for display in Telegram message.

    Args:
        class_dist: Dictionary from compute_dataset_stats

    Returns:
        Formatted string like "Class 0: 150 (75.0%), Class 1: 50 (25.0%)"
    """
    parts = []
    for class_val, info in sorted(class_dist.items()):
        parts.append(f"Class {class_val}: {info['count']} ({info['percentage']}%)")
    return ", ".join(parts)


def format_quartiles(quartiles: Dict[str, Optional[float]]) -> str:
    """
    Format quartiles for display.

    Args:
        quartiles: Dictionary with q1, q2, q3

    Returns:
        Formatted string like "Q1: 25.0 | Median: 50.0 | Q3: 75.0"
    """
    if quartiles['q1'] is None:
        return "N/A (empty dataset)"

    return (
        f"Q1: {quartiles['q1']:.4f} | "
        f"Median: {quartiles['q2']:.4f} | "
        f"Q3: {quartiles['q3']:.4f}"
    )
