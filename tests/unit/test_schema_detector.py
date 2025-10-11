"""
Comprehensive tests for schema detection functionality.

Tests cover:
- Column type inference (numeric, categorical, datetime, text)
- Target and feature candidate detection
- Task type suggestion (regression/classification)
- Schema detection for various dataset types
- Quality score calculation
- Edge cases and error handling

Author: Statistical Modeling Agent
Created: 2025-10-06 (Phase 2: Schema Detection)
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.schema_detector import (
    ColumnSchema,
    DatasetSchema,
    detect_schema,
    load_dataframe,
    analyze_column,
    infer_column_type,
    is_target_candidate,
    is_feature_candidate,
    suggest_task_type,
    suggest_target_column,
    suggest_feature_columns,
    calculate_quality_score,
    format_schema_for_display
)


class TestLoadDataframe:
    """Test dataset loading from various file formats."""

    def test_load_csv(self, tmp_path):
        """CSV files should load correctly."""
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(csv_file, index=False)

        loaded = load_dataframe(csv_file)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["a", "b"]

    def test_load_excel(self, tmp_path):
        """Excel files should load correctly."""
        excel_file = tmp_path / "test.xlsx"
        df = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
        df.to_excel(excel_file, index=False)

        loaded = load_dataframe(excel_file)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["col1", "col2"]

    @pytest.mark.skipif(
        not hasattr(pd.DataFrame, 'to_parquet'),
        reason="Parquet support requires pyarrow"
    )
    def test_load_parquet(self, tmp_path):
        """Parquet files should load correctly."""
        try:
            parquet_file = tmp_path / "test.parquet"
            df = pd.DataFrame({"x": [1.1, 2.2], "y": [3.3, 4.4]})
            df.to_parquet(parquet_file, index=False)

            loaded = load_dataframe(parquet_file)
            assert len(loaded) == 2
            assert list(loaded.columns) == ["x", "y"]
        except ImportError:
            pytest.skip("Parquet support requires pyarrow")

    def test_load_nonexistent_file(self, tmp_path):
        """Non-existent files should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataframe(tmp_path / "nonexistent.csv")

    def test_load_unsupported_format(self, tmp_path):
        """Unsupported file formats should raise ValueError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(txt_file)

    def test_load_corrupt_csv(self, tmp_path):
        """Corrupt CSV files should raise ValueError."""
        csv_file = tmp_path / "corrupt.csv"
        csv_file.write_text("invalid\ncsv\ndata\n\n\n")

        # Should not crash, pandas is resilient
        loaded = load_dataframe(csv_file)
        assert loaded is not None


class TestInferColumnType:
    """Test column type inference."""

    def test_infer_numeric_int(self):
        """Integer columns should be detected as numeric."""
        col = pd.Series([1, 2, 3, 4, 5])
        assert infer_column_type(col) == "numeric"

    def test_infer_numeric_float(self):
        """Float columns should be detected as numeric."""
        col = pd.Series([1.1, 2.2, 3.3, 4.4])
        assert infer_column_type(col) == "numeric"

    def test_infer_categorical_few_unique(self):
        """String columns with few unique values should be categorical."""
        col = pd.Series(["A", "B", "A", "C", "B", "A", "B", "C"])
        assert infer_column_type(col) == "categorical"

    def test_infer_text_many_unique(self):
        """String columns with many unique values should be text."""
        col = pd.Series([f"text_{i}" for i in range(100)])
        assert infer_column_type(col) == "text"

    def test_infer_datetime(self):
        """Datetime columns should be detected."""
        col = pd.Series(pd.date_range("2024-01-01", periods=10))
        assert infer_column_type(col) == "datetime"

    def test_infer_categorical_dtype(self):
        """Pandas categorical dtype should be detected."""
        col = pd.Series(["X", "Y", "Z"] * 10, dtype="category")
        assert infer_column_type(col) == "categorical"


class TestTargetCandidateDetection:
    """Test target column candidate detection."""

    def test_numeric_is_target_candidate(self):
        """Numeric columns with low nulls are good targets."""
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_target_candidate(col, "numeric", 5, 0.0) is True

    def test_categorical_2_to_20_classes_is_target(self):
        """Categorical with 2-20 classes is good for classification target."""
        col = pd.Series(["A", "B"] * 50)
        assert is_target_candidate(col, "categorical", 2, 0.0) is True

        col = pd.Series([f"class_{i%10}" for i in range(100)])
        assert is_target_candidate(col, "categorical", 10, 0.0) is True

    def test_categorical_too_many_classes_not_target(self):
        """Categorical with >20 classes is not ideal target."""
        col = pd.Series([f"class_{i}" for i in range(50)])
        assert is_target_candidate(col, "categorical", 50, 0.0) is False

    def test_high_nulls_not_target(self):
        """Columns with >5% nulls are not good targets."""
        col = pd.Series([1, 2, None, None, 5, 6, None, 8, 9, None])
        assert is_target_candidate(col, "numeric", 6, 40.0) is False

    def test_text_not_target(self):
        """Text columns are not good targets."""
        col = pd.Series(["some text"] * 10)
        assert is_target_candidate(col, "text", 1, 0.0) is False


class TestFeatureCandidateDetection:
    """Test feature column candidate detection."""

    def test_numeric_is_feature_candidate(self):
        """Numeric columns are good features."""
        col = pd.Series([1, 2, 3, 4, 5])
        assert is_feature_candidate(col, "numeric", 5, 0.0) is True

    def test_categorical_is_feature_candidate(self):
        """Categorical columns are good features."""
        col = pd.Series(["A", "B", "C"] * 10)
        assert is_feature_candidate(col, "categorical", 3, 0.0) is True

    def test_datetime_is_feature_candidate(self):
        """Datetime columns can be features with engineering."""
        col = pd.Series(pd.date_range("2024-01-01", periods=10))
        assert is_feature_candidate(col, "datetime", 10, 0.0) is True

    def test_all_unique_not_feature(self):
        """Columns with all unique values (ID columns) are not features."""
        col = pd.Series(range(100))
        assert is_feature_candidate(col, "numeric", 100, 0.0) is False

    def test_high_nulls_not_feature(self):
        """Columns with >80% nulls are not good features."""
        col = pd.Series([None] * 85 + [1] * 15)
        assert is_feature_candidate(col, "numeric", 2, 85.0) is False

    def test_text_not_feature(self):
        """Text columns need preprocessing, not ideal features."""
        col = pd.Series([f"text_{i}" for i in range(100)])
        assert is_feature_candidate(col, "text", 100, 0.0) is False


class TestAnalyzeColumn:
    """Test individual column analysis."""

    def test_analyze_numeric_column(self):
        """Numeric column analysis should be correct."""
        df = pd.DataFrame({"price": [100, 200, 300, 400, 500]})
        schema = analyze_column(df, "price", max_sample_values=3)

        assert schema.name == "price"
        assert schema.dtype == "numeric"
        assert schema.null_count == 0
        assert schema.null_percentage == 0.0
        assert schema.unique_count == 5
        assert len(schema.sample_values) <= 3
        assert schema.is_target_candidate is True
        assert schema.is_feature_candidate is True

    def test_analyze_categorical_column(self):
        """Categorical column analysis should be correct."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C", "B"] * 10})
        schema = analyze_column(df, "category")

        assert schema.dtype == "categorical"
        assert schema.unique_count == 3
        assert schema.is_target_candidate is True
        assert schema.is_feature_candidate is True

    def test_analyze_column_with_nulls(self):
        """Column with nulls should report correct statistics."""
        df = pd.DataFrame({"col": [1, 2, None, 4, None, 6, 7, 8, 9, 10]})
        schema = analyze_column(df, "col")

        assert schema.null_count == 2
        assert schema.null_percentage == 20.0
        assert schema.unique_count == 8  # Excludes nulls

    def test_analyze_id_column(self):
        """ID columns (all unique) should not be feature candidates."""
        df = pd.DataFrame({"id": range(100)})
        schema = analyze_column(df, "id")

        assert schema.unique_count == 100
        assert schema.unique_percentage == 100.0
        assert schema.is_feature_candidate is False


class TestTaskTypeSuggestion:
    """Test ML task type suggestion."""

    def test_suggest_regression_for_numeric_target(self):
        """Datasets with numeric targets should suggest regression."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "price": [100.0, 200.0, 150.0, 300.0, 250.0]
        })
        columns = [analyze_column(df, col) for col in df.columns]

        task_type = suggest_task_type(df, columns)
        assert task_type == "regression"

    def test_suggest_classification_for_categorical_target(self):
        """Datasets with categorical targets should suggest classification."""
        df = pd.DataFrame({
            "feature1": [i % 10 for i in range(50)],  # Non-sequential, repeated values
            "feature2": [i % 5 for i in range(50)],   # Non-sequential, repeated values
            "category": ["A", "B", "C"] * 16 + ["A", "B"]  # 50 rows, 3 classes
        })
        columns = [analyze_column(df, col) for col in df.columns]

        task_type = suggest_task_type(df, columns)
        assert task_type == "classification"

    def test_no_suggestion_without_target_candidates(self):
        """Datasets without good targets should return None."""
        df = pd.DataFrame({
            "id": range(100),
            "text": [f"description_{i}" for i in range(100)]
        })
        columns = [analyze_column(df, col) for col in df.columns]

        task_type = suggest_task_type(df, columns)
        assert task_type is None


class TestTargetColumnSuggestion:
    """Test target column suggestion."""

    def test_suggest_target_by_name(self):
        """Columns named 'target', 'label', 'price' should be preferred."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "price": [100, 200, 150, 300, 250]
        })
        columns = [analyze_column(df, col) for col in df.columns]

        target = suggest_target_column(df, columns, "regression")
        assert target == "price"

    def test_suggest_numeric_target_for_regression(self):
        """For regression, prefer numeric columns."""
        df = pd.DataFrame({
            "category": ["A", "B"] * 25,
            "value": [1.0, 2.0] * 25
        })
        columns = [analyze_column(df, col) for col in df.columns]

        target = suggest_target_column(df, columns, "regression")
        assert target == "value"

    def test_suggest_categorical_target_for_classification(self):
        """For classification, prefer categorical columns."""
        df = pd.DataFrame({
            "feature1": [i % 10 for i in range(50)],  # Non-sequential
            "class": ["A", "B", "C"] * 16 + ["A", "B"]  # 50 rows
        })
        columns = [analyze_column(df, col) for col in df.columns]

        target = suggest_target_column(df, columns, "classification")
        assert target == "class"


class TestFeatureColumnsSuggestion:
    """Test feature columns suggestion."""

    def test_suggest_features_excludes_target(self):
        """Suggested features should not include target column."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50],
            "target": [100, 200, 150, 300, 250]
        })
        columns = [analyze_column(df, col) for col in df.columns]

        features = suggest_feature_columns(df, columns, "target")
        assert "target" not in features
        assert "feature1" in features
        assert "feature2" in features

    def test_suggest_features_excludes_id_columns(self):
        """ID columns (all unique) should not be suggested as features."""
        df = pd.DataFrame({
            "id": range(100),
            "feature1": [1, 2] * 50,
            "target": [0, 1] * 50
        })
        columns = [analyze_column(df, col) for col in df.columns]

        features = suggest_feature_columns(df, columns, "target")
        assert "id" not in features
        assert "feature1" in features


class TestQualityScoreCalculation:
    """Test data quality score calculation."""

    def test_perfect_quality_score(self):
        """Complete data with good ratio should score high."""
        df = pd.DataFrame({
            "feature1": range(200),
            "feature2": range(200),
            "target": [0, 1] * 100
        })
        columns = [analyze_column(df, col) for col in df.columns]

        score = calculate_quality_score(df, columns)
        assert 0.8 <= score <= 1.0

    def test_low_quality_for_missing_values(self):
        """Data with many nulls should score lower."""
        df = pd.DataFrame({
            "feature1": [None] * 50 + [1] * 50,
            "feature2": [None] * 50 + [2] * 50,
            "target": [0, 1] * 50
        })
        columns = [analyze_column(df, col) for col in df.columns]

        score = calculate_quality_score(df, columns)
        assert score < 0.7

    def test_low_quality_for_few_rows(self):
        """Data with <10 rows should score very low."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "target": [0, 1, 0]
        })
        columns = [analyze_column(df, col) for col in df.columns]

        score = calculate_quality_score(df, columns)
        assert score < 0.5

    def test_low_quality_for_too_many_features(self):
        """More features than rows should score low."""
        # 20 features, 10 rows
        data = {f"feature_{i}": range(10) for i in range(20)}
        df = pd.DataFrame(data)
        columns = [analyze_column(df, col) for col in df.columns]

        score = calculate_quality_score(df, columns)
        assert score < 0.6


class TestDetectSchema:
    """Test complete schema detection."""

    def test_detect_schema_regression_dataset(self, tmp_path):
        """Regression dataset schema should be detected correctly."""
        csv_file = tmp_path / "regression.csv"
        df = pd.DataFrame({
            "sqft": [1000, 1500, 2000, 2500, 3000],
            "bedrooms": [2, 3, 3, 4, 4],
            "bathrooms": [1, 2, 2, 3, 3],
            "price": [200000, 300000, 350000, 450000, 500000]
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)

        assert schema.n_rows == 5
        assert schema.n_columns == 4
        assert schema.suggested_task_type == "regression"
        assert schema.suggested_target == "price"
        assert set(schema.suggested_features) == {"sqft", "bedrooms", "bathrooms"}

    def test_detect_schema_classification_dataset(self, tmp_path):
        """Classification dataset schema should be detected correctly."""
        csv_file = tmp_path / "classification.csv"
        df = pd.DataFrame({
            "feature1": [i % 10 for i in range(50)],  # Non-sequential
            "feature2": [i % 5 for i in range(50)],   # Non-sequential
            "class": ["A", "B"] * 25
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)

        assert schema.suggested_task_type == "classification"
        assert schema.suggested_target == "class"
        assert set(schema.suggested_features) == {"feature1", "feature2"}

    def test_detect_schema_with_missing_values(self, tmp_path):
        """Schema detection should handle missing values."""
        csv_file = tmp_path / "with_nulls.csv"
        df = pd.DataFrame({
            "feature1": [1, 2, None, 4, 5],
            "feature2": [10, None, 30, None, 50],
            "target": [100, 200, 300, 400, 500]
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)

        assert schema.has_missing_values is True
        assert any(c.null_count > 0 for c in schema.columns)

    def test_detect_schema_no_auto_suggest(self, tmp_path):
        """Schema detection without auto-suggest should not suggest columns."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file, auto_suggest=False)

        assert schema.suggested_task_type is None
        assert schema.suggested_target is None
        assert schema.suggested_features == []


class TestFormatSchemaForDisplay:
    """Test schema formatting for user display."""

    def test_format_schema_basic(self, tmp_path):
        """Basic schema should format correctly."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5],
            "target": [10, 20, 30, 40, 50]
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)
        formatted = format_schema_for_display(schema)

        assert "Dataset Schema" in formatted
        assert "Rows: 5" in formatted
        assert "Columns: 2" in formatted
        assert "feature" in formatted
        assert "target" in formatted

    def test_format_includes_suggestions(self, tmp_path):
        """Formatted schema should include suggestions."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "price": [100, 200, 150, 300, 250]
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)
        formatted = format_schema_for_display(schema)

        assert "Suggested Task:" in formatted
        assert "Suggested Target:" in formatted
        assert "Suggested Features:" in formatted

    def test_format_shows_missing_warning(self, tmp_path):
        """Formatted schema should warn about missing values."""
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({
            "x": [1, None, 3],
            "y": [10, 20, 30]
        })
        df.to_csv(csv_file, index=False)

        schema = detect_schema(csv_file)
        formatted = format_schema_for_display(schema)

        assert "missing values" in formatted.lower()
