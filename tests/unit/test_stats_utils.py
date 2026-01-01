"""
Unit tests for stats_utils module.

Tests the compute_dataset_stats function for basic statistics
and class distribution computation.
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.stats_utils import compute_dataset_stats


@pytest.fixture
def binary_classification_data():
    """Sample binary classification dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(200),
        'feature2': np.random.randn(200),
        'target': [0] * 150 + [1] * 50  # 75% class 0, 25% class 1
    })


@pytest.fixture
def multiclass_data():
    """Sample multi-class classification dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'target': [0] * 40 + [1] * 35 + [2] * 25  # 3 classes
    })


@pytest.fixture
def regression_data():
    """Sample regression dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'target': np.random.randn(100) * 10 + 50  # Mean ~50
    })


class TestComputeDatasetStats:
    """Tests for compute_dataset_stats function."""

    def test_returns_row_count(self, binary_classification_data):
        """Test that n_rows is returned correctly."""
        stats = compute_dataset_stats(
            binary_classification_data,
            target_col='target',
            task_type='classification'
        )
        assert stats['n_rows'] == 200

    def test_returns_quartiles_classification(self, binary_classification_data):
        """Test quartiles for classification target."""
        stats = compute_dataset_stats(
            binary_classification_data,
            target_col='target',
            task_type='classification'
        )
        assert 'quartiles' in stats
        assert 'q1' in stats['quartiles']
        assert 'q2' in stats['quartiles']
        assert 'q3' in stats['quartiles']

    def test_returns_quartiles_regression(self, regression_data):
        """Test quartiles for regression target."""
        stats = compute_dataset_stats(
            regression_data,
            target_col='target',
            task_type='regression'
        )
        assert 'quartiles' in stats
        q1 = stats['quartiles']['q1']
        q2 = stats['quartiles']['q2']
        q3 = stats['quartiles']['q3']
        # Q1 < Q2 < Q3
        assert q1 < q2 < q3

    def test_class_distribution_binary(self, binary_classification_data):
        """Test class distribution for binary classification."""
        stats = compute_dataset_stats(
            binary_classification_data,
            target_col='target',
            task_type='classification'
        )
        assert 'class_distribution' in stats
        dist = stats['class_distribution']

        # Check class 0
        assert dist[0]['count'] == 150
        assert dist[0]['percentage'] == 75.0

        # Check class 1
        assert dist[1]['count'] == 50
        assert dist[1]['percentage'] == 25.0

    def test_class_distribution_multiclass(self, multiclass_data):
        """Test class distribution for multi-class."""
        stats = compute_dataset_stats(
            multiclass_data,
            target_col='target',
            task_type='classification'
        )
        dist = stats['class_distribution']

        assert dist[0]['count'] == 40
        assert dist[0]['percentage'] == 40.0
        assert dist[1]['count'] == 35
        assert dist[1]['percentage'] == 35.0
        assert dist[2]['count'] == 25
        assert dist[2]['percentage'] == 25.0

    def test_no_class_distribution_for_regression(self, regression_data):
        """Test that regression doesn't include class distribution."""
        stats = compute_dataset_stats(
            regression_data,
            target_col='target',
            task_type='regression'
        )
        assert 'class_distribution' not in stats

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame({'target': []})
        stats = compute_dataset_stats(
            empty_df,
            target_col='target',
            task_type='classification'
        )
        assert stats['n_rows'] == 0

    def test_missing_target_column(self, binary_classification_data):
        """Test handling of missing target column."""
        with pytest.raises(KeyError):
            compute_dataset_stats(
                binary_classification_data,
                target_col='nonexistent',
                task_type='classification'
            )

    def test_single_row(self):
        """Test with single row dataset."""
        single_row = pd.DataFrame({'target': [1]})
        stats = compute_dataset_stats(
            single_row,
            target_col='target',
            task_type='classification'
        )
        assert stats['n_rows'] == 1
        assert stats['class_distribution'][1]['count'] == 1
        assert stats['class_distribution'][1]['percentage'] == 100.0

    def test_prediction_column_stats(self, regression_data):
        """Test stats computation for prediction column."""
        # Add prediction column
        regression_data['predictions'] = np.random.randn(100) * 5 + 45

        stats = compute_dataset_stats(
            regression_data,
            target_col='predictions',
            task_type='regression'
        )
        assert stats['n_rows'] == 100
        assert 'quartiles' in stats


class TestFormatClassDistribution:
    """Tests for class distribution formatting helper."""

    def test_format_output_structure(self, binary_classification_data):
        """Test that formatted output has correct structure."""
        stats = compute_dataset_stats(
            binary_classification_data,
            target_col='target',
            task_type='classification'
        )
        # Verify structure allows easy message formatting
        for class_val, info in stats['class_distribution'].items():
            assert 'count' in info
            assert 'percentage' in info
            assert isinstance(info['count'], int)
            assert isinstance(info['percentage'], float)
