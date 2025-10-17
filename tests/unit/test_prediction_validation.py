"""
Unit tests for ML Prediction validation functions.

Tests:
1. Feature parsing and validation
2. CSV preview generation
3. Statistics calculation
4. Temporary file management

Related: src/bot/ml_handlers/prediction_handlers.py
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestFeatureParsing:
    """Test feature input parsing and validation."""

    def test_parse_comma_separated_features(self):
        """Test parsing comma-separated feature names."""
        input_text = "sqft, bedrooms, bathrooms"
        features = [f.strip() for f in input_text.split(',')]

        assert len(features) == 3
        assert features == ['sqft', 'bedrooms', 'bathrooms']

    def test_parse_features_with_extra_spaces(self):
        """Test handling of extra whitespace."""
        input_text = "  sqft  ,   bedrooms  ,  bathrooms  "
        features = [f.strip() for f in input_text.split(',')]

        assert len(features) == 3
        assert features == ['sqft', 'bedrooms', 'bathrooms']

    def test_parse_single_feature(self):
        """Test parsing a single feature."""
        input_text = "sqft"
        features = [f.strip() for f in input_text.split(',')]

        assert len(features) == 1
        assert features == ['sqft']

    def test_empty_feature_detection(self):
        """Test detection of empty features after split."""
        input_text = "sqft, , bedrooms"
        features = [f.strip() for f in input_text.split(',') if f.strip()]

        # Empty feature should be filtered out
        assert len(features) == 2
        assert features == ['sqft', 'bedrooms']

    def test_validate_features_exist_in_dataframe(self):
        """Test validation that features exist in DataFrame."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2],
            'price': [100000, 200000, 300000]
        })

        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # All features exist
        missing = [f for f in selected_features if f not in df.columns]
        assert len(missing) == 0

    def test_detect_invalid_features(self):
        """Test detection of features not in DataFrame."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4]
        })

        selected_features = ['sqft', 'bedrooms', 'bathrooms', 'age']

        # Find invalid features
        invalid = [f for f in selected_features if f not in df.columns]
        assert len(invalid) == 2
        assert set(invalid) == {'bathrooms', 'age'}


class TestCSVPreviewGeneration:
    """Test CSV preview generation for results."""

    def test_generate_preview_first_10_rows(self):
        """Test generating preview of first 10 rows."""
        # Create DataFrame with 15 rows
        df = pd.DataFrame({
            'sqft': range(1000, 1150, 10),
            'price_predicted': range(100000, 115000, 1000)
        })

        # Get first 10 rows
        preview_df = df.head(10)
        preview = preview_df.to_string(index=False)

        # Verify preview contains 10 rows
        lines = preview.strip().split('\n')
        assert len(lines) == 11  # Header + 10 data rows

    def test_preview_format_with_index_false(self):
        """Test that preview doesn't include index column."""
        df = pd.DataFrame({
            'sqft': [1000, 2000],
            'price_predicted': [100000, 200000]
        })

        preview = df.to_string(index=False)

        # Preview should not start with numbers (index)
        assert not preview.strip()[0].isdigit()

    def test_preview_with_small_dataframe(self):
        """Test preview when DataFrame has fewer than 10 rows."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'price_predicted': [100000, 200000, 300000]
        })

        preview_df = df.head(10)  # Will return all 3 rows
        assert len(preview_df) == 3


class TestStatisticsCalculation:
    """Test prediction statistics calculation."""

    def test_calculate_mean(self):
        """Test mean calculation for predictions."""
        predictions = [100.0, 200.0, 300.0, 400.0, 500.0]
        mean = np.mean(predictions)

        assert mean == 300.0

    def test_calculate_std(self):
        """Test standard deviation calculation."""
        predictions = [100.0, 200.0, 300.0, 400.0, 500.0]
        std = np.std(predictions)

        # Standard deviation of [100, 200, 300, 400, 500] ≈ 141.42
        assert pytest.approx(std, rel=1e-2) == 141.42

    def test_calculate_min_max(self):
        """Test min and max calculation."""
        predictions = [100.0, 200.0, 300.0, 400.0, 500.0]

        assert min(predictions) == 100.0
        assert max(predictions) == 500.0

    def test_calculate_median(self):
        """Test median calculation."""
        predictions = [100.0, 200.0, 300.0, 400.0, 500.0]
        median = np.median(predictions)

        assert median == 300.0

    def test_calculate_median_even_length(self):
        """Test median calculation for even-length array."""
        predictions = [100.0, 200.0, 300.0, 400.0]
        median = np.median(predictions)

        # Median of 4 values is average of middle two
        assert median == 250.0

    def test_statistics_with_pandas_series(self):
        """Test statistics calculation using pandas Series."""
        predictions = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])

        statistics = {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'median': float(predictions.median())
        }

        assert statistics['mean'] == 300.0
        assert statistics['min'] == 100.0
        assert statistics['max'] == 500.0
        assert statistics['median'] == 300.0
        # pandas uses sample std (N-1), which gives 158.11 for this data
        assert pytest.approx(statistics['std'], rel=1e-2) == 158.11

    def test_statistics_with_numpy_array(self):
        """Test statistics calculation using numpy array."""
        predictions = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        statistics = {
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'median': float(np.median(predictions))
        }

        assert statistics['mean'] == 300.0
        assert statistics['min'] == 100.0
        assert statistics['max'] == 500.0
        assert statistics['median'] == 300.0


class TestTemporaryFileManagement:
    """Test temporary file creation and cleanup."""

    def test_create_temp_csv(self):
        """Test creating temporary CSV file."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'price_predicted': [100000, 200000, 300000]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            csv_path = tmp.name
            df.to_csv(csv_path, index=False)

        # Verify file exists
        assert Path(csv_path).exists()

        # Verify contents
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['sqft', 'price_predicted']

        # Cleanup
        Path(csv_path).unlink()

    def test_temp_file_naming_with_model_id(self):
        """Test temporary file naming convention."""
        model_id = "model_12345_random_forest"
        user_id = 12345

        # Expected naming pattern
        filename = f"predictions_{user_id}_{model_id}.csv"

        assert "predictions_" in filename
        assert str(user_id) in filename
        assert model_id in filename
        assert filename.endswith('.csv')

    def test_csv_without_index(self):
        """Test CSV export without index column."""
        df = pd.DataFrame({
            'sqft': [1000, 2000],
            'price_predicted': [100000, 200000]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            csv_path = tmp.name
            df.to_csv(csv_path, index=False)

        # Read back
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()

        # First line should be header, not index
        assert first_line == 'sqft,price_predicted'

        # Cleanup
        Path(csv_path).unlink()


class TestPredictionColumnAddition:
    """Test adding prediction column to DataFrame."""

    def test_add_prediction_column_to_dataframe(self):
        """Test adding predictions as a new column."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4]
        })

        predictions = [100000, 200000, 300000]
        prediction_column = 'price_predicted'

        # Add column
        df[prediction_column] = predictions

        # Verify column exists
        assert prediction_column in df.columns
        assert len(df) == 3
        assert list(df[prediction_column]) == predictions

    def test_prediction_column_preserves_original_data(self):
        """Test that adding prediction column doesn't modify original columns."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4]
        })

        original_sqft = df['sqft'].tolist()
        original_bedrooms = df['bedrooms'].tolist()

        # Add predictions
        df['price_predicted'] = [100000, 200000, 300000]

        # Verify original columns unchanged
        assert df['sqft'].tolist() == original_sqft
        assert df['bedrooms'].tolist() == original_bedrooms

    def test_prediction_column_order(self):
        """Test that prediction column appears at end."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4]
        })

        df['price_predicted'] = [100000, 200000, 300000]

        # Prediction column should be last
        columns = list(df.columns)
        assert columns == ['sqft', 'bedrooms', 'price_predicted']


class TestFeatureSubsetExtraction:
    """Test extracting feature subset from DataFrame."""

    def test_extract_selected_features(self):
        """Test extracting only selected features for prediction."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2],
            'age': [5, 10, 15],
            'price': [100000, 200000, 300000]  # Not a feature
        })

        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Extract subset
        feature_df = df[selected_features].copy()

        # Verify only selected features present
        assert list(feature_df.columns) == selected_features
        assert len(feature_df) == 3
        assert 'age' not in feature_df.columns
        assert 'price' not in feature_df.columns

    def test_feature_order_preserved(self):
        """Test that feature order is preserved during extraction."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        # Extract in specific order
        selected_features = ['c', 'a', 'b']
        feature_df = df[selected_features]

        # Verify order
        assert list(feature_df.columns) == ['c', 'a', 'b']


class TestModelMetadataValidation:
    """Test model metadata validation for predictions."""

    def test_model_info_contains_required_fields(self):
        """Test that model info has required fields."""
        model_info = {
            'model_id': 'model_12345_random_forest',
            'model_type': 'random_forest',
            'task_type': 'regression',
            'target_column': 'price',
            'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
            'metrics': {'r2': 0.85, 'mse': 1000.0}
        }

        required_fields = [
            'model_id',
            'model_type',
            'target_column',
            'feature_columns'
        ]

        # All required fields present
        for field in required_fields:
            assert field in model_info

    def test_validate_model_features_match_selected(self):
        """Test validation that model features match selected features."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Should match exactly
        assert set(model_features) == set(selected_features)

    def test_detect_feature_count_mismatch(self):
        """Test detection of feature count mismatch."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['sqft', 'bedrooms']

        # Different lengths
        assert len(model_features) != len(selected_features)

        # Different sets
        assert set(model_features) != set(selected_features)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
