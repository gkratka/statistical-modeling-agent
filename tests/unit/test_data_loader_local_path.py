"""Unit tests for DataLoader local path functionality."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.processors.data_loader import DataLoader
from src.utils.exceptions import DataError, ValidationError, PathValidationError


class TestDataLoaderLocalPath:

    @pytest.fixture
    def sample_excel(self, tmp_path):
        excel_file = tmp_path / "sample.xlsx"
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        df.to_excel(excel_file, index=False)
        return excel_file

    @pytest.mark.asyncio
    async def test_load_from_local_path_csv_success(self, config_enabled, sample_csv):
        loader = DataLoader(config=config_enabled)

        df, metadata, schema = await loader.load_from_local_path(str(sample_csv))

        assert df is not None
        assert len(df) == 5
        assert 'feature1' in df.columns
        assert metadata['source'] == 'local_path'
        assert metadata['file_type'] == 'csv'
        assert schema is not None
        assert schema.n_rows == 5
        assert schema.n_columns == 3

    @pytest.mark.asyncio
    async def test_load_from_local_path_excel_success(self, config_enabled, sample_excel):
        loader = DataLoader(config=config_enabled)

        df, metadata, schema = await loader.load_from_local_path(str(sample_excel))

        assert df is not None
        assert len(df) == 3
        assert 'x' in df.columns
        assert metadata['source'] == 'local_path'
        assert metadata['file_type'] == 'excel'

    @pytest.mark.asyncio
    async def test_load_from_local_path_relative_path(self, config_enabled, tmp_path, sample_csv):
        loader = DataLoader(config=config_enabled)

        # Use relative path from tmp_path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            df, metadata, schema = await loader.load_from_local_path("./sample.csv")

            assert df is not None
            assert metadata['resolved_path'] != "./sample.csv"
            assert Path(metadata['resolved_path']).is_absolute()
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_load_from_local_path_without_schema(self, config_enabled, sample_csv):
        loader = DataLoader(config=config_enabled)

        df, metadata, schema = await loader.load_from_local_path(
            str(sample_csv),
            detect_schema_flag=False
        )

        assert df is not None
        assert schema is None
        assert metadata['schema_detected'] is False

    @pytest.mark.asyncio
    async def test_load_from_local_path_feature_disabled(self, config_disabled, sample_csv):
        loader = DataLoader(config=config_disabled)

        with pytest.raises(ValidationError, match="disabled"):
            await loader.load_from_local_path(str(sample_csv))

    @pytest.mark.asyncio
    async def test_load_from_local_path_outside_whitelist(self, config_enabled, tmp_path):
        loader = DataLoader(config=config_enabled)

        # Create file in different directory
        other_dir = tmp_path.parent / "other"
        other_dir.mkdir(exist_ok=True)
        csv_file = other_dir / "data.csv"
        df = pd.DataFrame({'a': [1, 2, 3]})
        df.to_csv(csv_file, index=False)

        with pytest.raises(PathValidationError, match="not in allowed"):
            await loader.load_from_local_path(str(csv_file))

    @pytest.mark.asyncio
    async def test_load_from_local_path_nonexistent_file(self, config_enabled, tmp_path):
        loader = DataLoader(config=config_enabled)

        with pytest.raises(PathValidationError, match="not found"):
            await loader.load_from_local_path(str(tmp_path / "nonexistent.csv"))

    @pytest.mark.asyncio
    async def test_load_from_local_path_invalid_extension(self, config_enabled, tmp_path):
        loader = DataLoader(config=config_enabled)

        # Create .txt file
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some data")

        with pytest.raises(PathValidationError, match="extension"):
            await loader.load_from_local_path(str(txt_file))

    @pytest.mark.asyncio
    async def test_load_from_local_path_oversized_file(self, tmp_path):
        # Config with very small size limit
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 0.0001,  # 100 bytes
                'allowed_extensions': ['.csv']
            }
        }
        loader = DataLoader(config=config)

        # Create file larger than limit
        csv_file = tmp_path / "large.csv"
        df = pd.DataFrame({'data': range(1000)})
        df.to_csv(csv_file, index=False)

        with pytest.raises(PathValidationError, match="too large"):
            await loader.load_from_local_path(str(csv_file))

    @pytest.mark.asyncio
    async def test_load_from_local_path_path_traversal(self, config_enabled, tmp_path):
        loader = DataLoader(config=config_enabled)

        # Try path traversal
        attack_path = str(tmp_path / ".." / "etc" / "passwd")

        with pytest.raises(PathValidationError, match="traversal"):
            await loader.load_from_local_path(attack_path)

    @pytest.mark.asyncio
    async def test_load_from_local_path_empty_file(self, config_enabled, tmp_path):
        loader = DataLoader(config=config_enabled)

        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(PathValidationError, match="empty"):
            await loader.load_from_local_path(str(empty_file))

    @pytest.mark.asyncio
    async def test_load_from_local_path_metadata_completeness(self, config_enabled, sample_csv):
        loader = DataLoader(config=config_enabled)

        df, metadata, schema = await loader.load_from_local_path(str(sample_csv))

        # Check required metadata fields
        assert 'source' in metadata
        assert 'original_path' in metadata
        assert 'resolved_path' in metadata
        assert 'shape' in metadata
        assert 'columns' in metadata
        assert 'dtypes' in metadata
        assert 'missing_percentage' in metadata
        assert 'memory_usage_mb' in metadata
        assert 'schema_detected' in metadata

        # Check schema-related metadata
        if metadata['schema_detected']:
            assert 'suggested_task_type' in metadata
            assert 'suggested_target' in metadata
            assert 'suggested_features' in metadata

    @pytest.mark.asyncio
    async def test_load_from_local_path_schema_suggestions(self, config_enabled, sample_csv):
        loader = DataLoader(config=config_enabled)

        df, metadata, schema = await loader.load_from_local_path(str(sample_csv))

        # Check schema suggestions
        assert schema.suggested_task_type in ['regression', 'classification']
        assert schema.suggested_target is not None
        assert len(schema.suggested_features) > 0
        assert schema.suggested_target not in schema.suggested_features


class TestGetLocalPathSummary:

    def test_get_local_path_summary_with_schema(self, tmp_path):
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 10,
                'allowed_extensions': ['.csv']
            }
        }
        loader = DataLoader(config=config)

        # Create mock data
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        metadata = {
            'shape': (3, 2),
            'columns': ['a', 'b'],
            'file_type': 'csv',
            'resolved_path': str(tmp_path / 'test.csv'),
            'missing_percentage': 0.0,
            'duplicate_count': 0,
            'numeric_columns': ['a', 'b'],
            'categorical_columns': [],
            'memory_usage_mb': 0.001,
            'schema_detected': True,
            'suggested_task_type': 'regression',
            'suggested_target': 'b',
            'suggested_features': ['a']
        }

        # Create mock schema
        from src.utils.schema_detector import DatasetSchema, ColumnSchema
        schema = DatasetSchema(
            file_path=str(tmp_path / 'test.csv'),
            n_rows=3,
            n_columns=2,
            columns=[],
            suggested_task_type='regression',
            suggested_target='b',
            suggested_features=['a'],
            memory_usage_mb=0.001,
            has_missing_values=False,
            overall_quality_score=0.95
        )

        summary = loader.get_local_path_summary(df, metadata, schema)

        assert "Data Loaded from Local Path" in summary
        assert "test.csv" in summary
        assert "Auto-Detected" in summary  # Updated format
        assert "Regression" in summary
        assert "Target: `b`" in summary
        assert "Features: `a`" in summary
        assert "95% quality" in summary  # Updated format

    def test_get_local_path_summary_without_schema(self, tmp_path):
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 10,
                'allowed_extensions': ['.csv']
            }
        }
        loader = DataLoader(config=config)

        df = pd.DataFrame({'a': [1, 2, 3]})
        metadata = {
            'shape': (3, 1),
            'columns': ['a'],
            'file_type': 'csv',
            'resolved_path': str(tmp_path / 'test.csv'),
            'missing_percentage': 0.0,
            'duplicate_count': 0,
            'numeric_columns': ['a'],
            'categorical_columns': [],
            'memory_usage_mb': 0.001,
            'schema_detected': False
        }

        summary = loader.get_local_path_summary(df, metadata, schema=None)

        assert "Data Loaded from Local Path" in summary
        assert "Auto-Detected Schema" not in summary
        assert "Ready for ML training" in summary


class TestDataLoaderBackwardCompatibility:

    def test_initialization_without_config(self):
        loader = DataLoader()

        assert loader.local_enabled is False
        assert loader.allowed_directories == []

    def test_initialization_with_config(self, tmp_path):
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 100,
                'allowed_extensions': ['.csv']
            }
        }
        loader = DataLoader(config=config)

        assert loader.local_enabled is True
        assert len(loader.allowed_directories) == 1
        assert loader.local_max_size_mb == 100
