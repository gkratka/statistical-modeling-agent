"""
Unit tests for the data loader module.

This module tests data loading, validation, and error handling
for various file types and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.processors.data_loader import DataLoader, load_telegram_file, validate_dataframe
from src.utils.exceptions import DataError, ValidationError


class TestDataLoader:
    """Test DataLoader class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        assert self.loader.MAX_FILE_SIZE == 10 * 1024 * 1024
        assert '.csv' in self.loader.SUPPORTED_EXTENSIONS
        assert '.xlsx' in self.loader.SUPPORTED_EXTENSIONS
        assert self.loader.MIN_ROWS == 1
        assert self.loader.MAX_MISSING_PERCENTAGE == 90.0

    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert self.loader._get_file_extension("test.csv") == ".csv"
        assert self.loader._get_file_extension("data.xlsx") == ".xlsx"
        assert self.loader._get_file_extension("FILE.CSV") == ".csv"  # Case insensitive
        assert self.loader._get_file_extension("no_extension") == ""

    def test_clean_column_names(self):
        """Test column name cleaning."""
        original = ["Column 1", "Column-2", "Column.3", "Column/4", ""]
        expected = ["Column_1", "Column_2", "Column_3", "Column_4", "column_4"]
        result = self.loader._clean_column_names(pd.Index(original))
        assert result == expected

    def test_clean_column_names_duplicates(self):
        """Test handling of duplicate column names."""
        original = ["name", "name", "name"]
        result = self.loader._clean_column_names(pd.Index(original))
        assert result == ["name", "name_1", "name_2"]

    def test_clean_column_names_special_chars(self):
        """Test cleaning special characters in column names."""
        original = ["user@domain", "price$", "data/info", "test\\file"]
        expected = ["user_domain", "price_", "data_info", "test_file"]
        result = self.loader._clean_column_names(pd.Index(original))
        assert result == expected


class TestFileValidation:
    """Test file metadata validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_validate_file_metadata_valid(self):
        """Test validation of valid file metadata."""
        # Should not raise any exception
        self.loader._validate_file_metadata("test.csv", 1024)
        self.loader._validate_file_metadata("data.xlsx", 5000000)

    def test_validate_file_metadata_too_large(self):
        """Test validation failure for oversized files."""
        with pytest.raises(ValidationError, match="File too large"):
            self.loader._validate_file_metadata("huge.csv", 20 * 1024 * 1024)

    def test_validate_file_metadata_empty(self):
        """Test validation failure for empty files."""
        with pytest.raises(ValidationError, match="File is empty"):
            self.loader._validate_file_metadata("empty.csv", 0)

    def test_validate_file_metadata_unsupported_type(self):
        """Test validation failure for unsupported file types."""
        with pytest.raises(ValidationError, match="Unsupported file type"):
            self.loader._validate_file_metadata("document.pdf", 1024)

    def test_validate_file_metadata_invalid_filename(self):
        """Test validation failure for invalid filenames."""
        with pytest.raises(ValidationError, match="Invalid filename"):
            self.loader._validate_file_metadata("../malicious.csv", 1024)

        with pytest.raises(ValidationError, match="Invalid filename"):
            self.loader._validate_file_metadata("/etc/passwd", 1024)

        with pytest.raises(ValidationError, match="Invalid filename"):
            self.loader._validate_file_metadata("", 1024)


class TestDataFrameValidation:
    """Test DataFrame validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['x', 'y', 'z', 'w', 'v'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })

        metadata = self.loader._validate_dataframe(df, "test.csv")

        assert metadata['shape'] == (5, 3)
        assert len(metadata['columns']) == 3
        assert metadata['missing_percentage'] < 1.0
        assert 'numeric_columns' in metadata
        assert 'categorical_columns' in metadata

    def test_validate_dataframe_too_few_rows(self):
        """Test validation failure for insufficient rows."""
        df = pd.DataFrame()  # Empty DataFrame

        with pytest.raises(DataError, match="empty after reading"):
            self.loader._validate_dataframe(df, "empty.csv")

    def test_validate_dataframe_too_many_rows(self):
        """Test validation failure for too many rows."""
        # Create a DataFrame that exceeds row limit
        with patch.object(self.loader, 'MAX_ROWS', 3):
            df = pd.DataFrame({'A': range(10)})  # 10 rows > 3
            with pytest.raises(DataError, match="Too much data"):
                self.loader._validate_dataframe(df, "big.csv")

    def test_validate_dataframe_too_many_columns(self):
        """Test validation failure for too many columns."""
        with patch.object(self.loader, 'MAX_COLUMNS', 2):
            df = pd.DataFrame({f'col_{i}': [1, 2, 3] for i in range(5)})  # 5 cols > 2
            with pytest.raises(DataError, match="Too many columns"):
                self.loader._validate_dataframe(df, "wide.csv")

    def test_validate_dataframe_too_much_missing_data(self):
        """Test validation failure for excessive missing data."""
        # Create DataFrame with 95% missing data (> 90% threshold)
        data = np.full((100, 10), np.nan)
        data[:5, :] = 1  # Only 5% of data is not missing
        df = pd.DataFrame(data)

        with pytest.raises(DataError, match="Too much missing data"):
            self.loader._validate_dataframe(df, "sparse.csv")

    def test_validate_dataframe_metadata_content(self):
        """Test metadata content for valid DataFrame."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, np.nan],
            'text': ['a', 'b', 'c', 'd'],
            'float_col': [1.1, 2.2, np.nan, 4.4]
        })

        metadata = self.loader._validate_dataframe(df, "test.csv")

        assert metadata['shape'] == (4, 3)
        assert 'numeric' in metadata['numeric_columns']
        assert 'float_col' in metadata['numeric_columns']
        assert 'text' in metadata['categorical_columns']
        assert metadata['missing_percentage'] > 0  # Has some missing data
        assert metadata['has_duplicates'] is False
        assert metadata['duplicate_count'] == 0


class TestCSVProcessing:
    """Test CSV file processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def create_test_csv(self, content: str, encoding: str = 'utf-8') -> Path:
        """Create a temporary CSV file with given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                               delete=False, encoding=encoding)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)

    @pytest.mark.asyncio
    async def test_read_csv_valid(self):
        """Test reading valid CSV file."""
        content = "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"
        csv_path = self.create_test_csv(content)

        try:
            df, metadata = await self.loader._read_csv(csv_path, "test.csv")

            assert df.shape == (3, 3)
            assert list(df.columns) == ['name', 'age', 'city']
            assert metadata['file_type'] == 'csv'
            assert metadata['encoding'] == 'utf-8'
            assert 'age' in metadata['numeric_columns']
            assert 'name' in metadata['categorical_columns']

        finally:
            csv_path.unlink()

    @pytest.mark.asyncio
    async def test_read_csv_latin1_encoding(self):
        """Test reading CSV with Latin-1 encoding."""
        content = "name,age\nJosé,25\nMüller,30"
        csv_path = self.create_test_csv(content, encoding='latin-1')

        try:
            df, metadata = await self.loader._read_csv(csv_path, "test.csv")

            assert df.shape == (2, 2)
            assert metadata['encoding'] == 'latin-1'
            assert 'José' in df['name'].values or 'Müller' in df['name'].values

        finally:
            csv_path.unlink()

    @pytest.mark.asyncio
    async def test_read_csv_empty_file(self):
        """Test reading empty CSV file."""
        csv_path = self.create_test_csv("")

        try:
            with pytest.raises(DataError, match="empty or contains no data"):
                await self.loader._read_csv(csv_path, "empty.csv")
        finally:
            csv_path.unlink()

    @pytest.mark.asyncio
    async def test_read_csv_malformed(self):
        """Test reading malformed CSV file."""
        content = "name,age,city\nJohn,25\nJane,30,LA,Extra"  # Inconsistent columns
        csv_path = self.create_test_csv(content)

        try:
            # Should still work with pandas' error handling
            df, metadata = await self.loader._read_csv(csv_path, "malformed.csv")
            assert df.shape[0] >= 1  # At least header or some data
        except DataError:
            # Also acceptable to fail with DataError
            pass
        finally:
            csv_path.unlink()


class TestExcelProcessing:
    """Test Excel file processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def create_test_excel(self, data_dict: dict, sheet_name: str = 'Sheet1') -> Path:
        """Create a temporary Excel file with given data."""
        df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_file.close()

        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        return Path(temp_file.name)

    @pytest.mark.asyncio
    async def test_read_excel_valid(self):
        """Test reading valid Excel file."""
        data = {
            'product': ['Apple', 'Banana', 'Cherry'],
            'price': [1.20, 0.50, 2.30],
            'quantity': [10, 25, 15]
        }
        excel_path = self.create_test_excel(data)

        try:
            df, metadata = await self.loader._read_excel(excel_path, "test.xlsx")

            assert df.shape == (3, 3)
            assert list(df.columns) == ['product', 'price', 'quantity']
            assert metadata['file_type'] == 'excel'
            assert metadata['sheet_name'] == 'Sheet1'
            assert 'price' in metadata['numeric_columns']

        finally:
            excel_path.unlink()

    @pytest.mark.asyncio
    async def test_read_excel_multiple_sheets(self):
        """Test reading Excel file with multiple sheets."""
        data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}

        # Create Excel with multiple sheets
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_file.close()
        excel_path = Path(temp_file.name)

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame(data).to_excel(writer, sheet_name='First', index=False)
                pd.DataFrame(data).to_excel(writer, sheet_name='Second', index=False)

            df, metadata = await self.loader._read_excel(excel_path, "multi.xlsx")

            assert df.shape == (3, 2)
            assert metadata['sheet_name'] == 'First'  # Should read first sheet
            assert 'Second' in metadata['available_sheets']
            assert len(metadata['available_sheets']) == 2

        finally:
            excel_path.unlink()

    @pytest.mark.asyncio
    async def test_read_excel_file_not_found(self):
        """Test reading non-existent Excel file."""
        fake_path = Path("nonexistent.xlsx")

        with pytest.raises(DataError, match="not found"):
            await self.loader._read_excel(fake_path, "missing.xlsx")


class TestTelegramIntegration:
    """Test Telegram file integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    @pytest.mark.asyncio
    async def test_load_from_telegram_csv(self):
        """Test loading CSV from Telegram."""
        # Create mock Telegram file
        telegram_file = AsyncMock()
        context = MagicMock()

        # Create temporary CSV content
        csv_content = "name,age\nAlice,25\nBob,30"

        async def mock_download(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "test.csv", 1024, context
        )

        assert df.shape == (2, 2)
        assert list(df.columns) == ['name', 'age']
        assert metadata['shape'] == (2, 2)

    @pytest.mark.asyncio
    async def test_load_from_telegram_file_too_large(self):
        """Test Telegram file size validation."""
        telegram_file = AsyncMock()
        context = MagicMock()

        with pytest.raises(ValidationError, match="File too large"):
            await self.loader.load_from_telegram(
                telegram_file, "huge.csv", 20 * 1024 * 1024, context
            )

    @pytest.mark.asyncio
    async def test_load_from_telegram_unsupported_type(self):
        """Test Telegram unsupported file type."""
        telegram_file = AsyncMock()
        context = MagicMock()

        with pytest.raises(ValidationError, match="Unsupported file type"):
            await self.loader.load_from_telegram(
                telegram_file, "document.pdf", 1024, context
            )


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_dataframe_function(self):
        """Test standalone validate_dataframe function."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })

        metadata = validate_dataframe(df)

        assert metadata['shape'] == (3, 2)
        assert len(metadata['columns']) == 2
        assert 'A' in metadata['numeric_columns']
        assert 'B' in metadata['categorical_columns']

    @pytest.mark.asyncio
    async def test_load_telegram_file_function(self):
        """Test standalone load_telegram_file function."""
        telegram_file = AsyncMock()
        context = MagicMock()

        csv_content = "x,y\n1,2\n3,4"

        async def mock_download(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await load_telegram_file(
            telegram_file, "data.csv", 100, context
        )

        assert df.shape == (2, 2)
        assert metadata['file_type'] == 'csv'


class TestDataSummary:
    """Test data summary generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_get_data_summary(self):
        """Test data summary generation."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })

        metadata = {
            'shape': (3, 3),
            'columns': ['name', 'age', 'salary'],
            'file_type': 'csv',
            'encoding': 'utf-8',
            'missing_percentage': 0.0,
            'duplicate_count': 0,
            'memory_usage_mb': 0.1,
            'numeric_columns': ['age', 'salary'],
            'categorical_columns': ['name']
        }

        summary = self.loader.get_data_summary(df, metadata)

        assert "Data Successfully Loaded" in summary
        assert "3 rows × 3 columns" in summary
        assert "CSV" in summary
        assert "utf-8" in summary
        assert "Missing data: 0.0%" in summary
        assert "name, age, salary" in summary


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_data_error_with_shape(self):
        """Test DataError includes shape information."""
        try:
            raise DataError("Test error", data_shape=(100, 10))
        except DataError as e:
            assert e.data_shape == (100, 10)

    def test_validation_error_with_field(self):
        """Test ValidationError includes field information."""
        try:
            raise ValidationError("Test error", field="file_size", value="999999")
        except ValidationError as e:
            assert e.field == "file_size"
            assert e.value == "999999"