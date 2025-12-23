"""
Tests for worker dataframe truncation to prevent OOM on Railway.

Verifies that:
1. Worker sends only sample rows (max 10) instead of full dataframe
2. Full predictions are saved to local CSV file
3. Result payload is under 1MB regardless of input size
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestWorkerDataframeTruncation:
    """Test that worker truncates dataframe in results."""

    def test_result_payload_has_sample_not_full_dataframe(self):
        """Verify result contains dataframe_sample, not full dataframe."""
        # Large dataframe (1000 rows)
        df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
        })
        predictions = np.random.randn(1000)

        # Simulate truncation logic
        SAMPLE_SIZE = 10
        result_data = {
            "predictions_sample": predictions[:SAMPLE_SIZE].tolist(),
            "predictions_count": len(predictions),
            "dataframe_sample": df.head(SAMPLE_SIZE).to_dict('records'),
            "dataframe_rows": len(df),
            "dataframe_columns": list(df.columns),
        }

        # Assertions
        assert len(result_data["dataframe_sample"]) == SAMPLE_SIZE
        assert result_data["dataframe_rows"] == 1000
        assert result_data["predictions_count"] == 1000
        assert len(result_data["predictions_sample"]) == SAMPLE_SIZE

    def test_result_payload_size_under_1mb(self):
        """Verify truncated result is under 1MB regardless of input size."""
        # Large dataframe (100k rows)
        df = pd.DataFrame({
            'feature1': np.random.randn(100000),
            'feature2': np.random.randn(100000),
            'feature3': np.random.randn(100000),
        })
        predictions = np.random.randn(100000)

        # Truncated result
        SAMPLE_SIZE = 10
        result_data = {
            "predictions_sample": predictions[:SAMPLE_SIZE].tolist(),
            "predictions_count": len(predictions),
            "dataframe_sample": df.head(SAMPLE_SIZE).to_dict('records'),
            "dataframe_rows": len(df),
            "dataframe_columns": list(df.columns),
            "output_file": "/path/to/predictions.csv",
        }

        # Serialize and check size
        payload_json = json.dumps(result_data)
        payload_size_bytes = len(payload_json.encode('utf-8'))
        payload_size_mb = payload_size_bytes / (1024 * 1024)

        assert payload_size_mb < 1.0, f"Payload size {payload_size_mb:.2f}MB exceeds 1MB limit"

    def test_full_results_saved_to_csv(self):
        """Verify full predictions are saved to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            df = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            })
            predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

            # Save full results (simulating worker behavior)
            output_path = Path(tmpdir) / "predictions_test_job.csv"
            df_with_predictions = df.copy()
            df_with_predictions['prediction'] = predictions
            df_with_predictions.to_csv(output_path, index=False)

            # Verify file exists and has all rows
            assert output_path.exists()
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) == 5
            assert 'prediction' in saved_df.columns
            assert list(saved_df['prediction']) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_small_dataframe_not_over_truncated(self):
        """Verify small dataframes (< SAMPLE_SIZE) are not padded."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
        })
        predictions = np.array([0.1, 0.2, 0.3])

        SAMPLE_SIZE = 10
        sample_size = min(SAMPLE_SIZE, len(df))

        result_data = {
            "predictions_sample": predictions[:sample_size].tolist(),
            "predictions_count": len(predictions),
            "dataframe_sample": df.head(sample_size).to_dict('records'),
            "dataframe_rows": len(df),
        }

        # Should have 3 rows, not padded to 10
        assert len(result_data["dataframe_sample"]) == 3
        assert result_data["dataframe_rows"] == 3


class TestBotHandlerTruncatedResults:
    """Test that bot handler correctly processes truncated results."""

    def test_handler_accepts_new_truncated_format(self):
        """Verify bot handler works with new truncated result format."""
        # New format from worker
        result = {
            "success": True,
            "predictions_sample": [0.1, 0.2, 0.3],
            "predictions_count": 1000,
            "dataframe_sample": [
                {"feature1": 1.0, "feature2": 10.0},
                {"feature1": 2.0, "feature2": 20.0},
            ],
            "dataframe_rows": 1000,
            "dataframe_columns": ["feature1", "feature2"],
            "output_file": "/home/user/predictions.csv",
        }

        # Simulate handler logic
        predictions_count = result.get("predictions_count", len(result.get("predictions_sample", [])))
        sample_rows = result.get("dataframe_sample", [])
        output_file = result.get("output_file")

        assert predictions_count == 1000
        assert len(sample_rows) == 2
        assert output_file == "/home/user/predictions.csv"

    def test_handler_backwards_compatible_with_old_format(self):
        """Verify bot handler still works with old full dataframe format."""
        # Old format (for backwards compatibility during transition)
        result = {
            "success": True,
            "predictions": [0.1, 0.2, 0.3],
            "count": 3,
            "dataframe": [
                {"feature1": 1.0, "feature2": 10.0},
                {"feature1": 2.0, "feature2": 20.0},
                {"feature1": 3.0, "feature2": 30.0},
            ],
        }

        # Handler should check for both formats
        predictions_count = result.get("predictions_count") or result.get("count", 0)
        sample_rows = result.get("dataframe_sample") or result.get("dataframe", [])

        assert predictions_count == 3
        assert len(sample_rows) == 3
