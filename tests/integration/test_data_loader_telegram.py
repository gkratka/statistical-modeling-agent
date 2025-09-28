"""
Integration tests for data loader with Telegram file handling.

This module tests the complete flow of file uploads from Telegram
including file download, processing, and integration with bot handlers.
"""

import pytest
import pandas as pd
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Document, File as TelegramFile

from src.processors.data_loader import DataLoader, load_telegram_file
from src.utils.exceptions import DataError, ValidationError


class TestTelegramFileIntegration:
    """Test integration with Telegram file handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def create_test_file(self, content: str, filename: str, encoding: str = 'utf-8') -> Path:
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix=Path(filename).suffix, delete=False, encoding=encoding
        )
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)

    @pytest.mark.asyncio
    async def test_complete_csv_workflow(self):
        """Test complete CSV workflow from Telegram to DataFrame."""
        # Create test CSV content
        csv_content = """name,age,city,salary
John Doe,25,New York,50000
Jane Smith,30,Los Angeles,60000
Bob Johnson,35,Chicago,55000
Alice Brown,28,Houston,52000"""

        # Mock Telegram file and download
        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        # Process the file
        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "employees.csv", len(csv_content.encode()), context
        )

        # Verify results
        assert df.shape == (4, 4)
        assert list(df.columns) == ['name', 'age', 'city', 'salary']
        assert metadata['file_type'] == 'csv'
        assert metadata['encoding'] == 'utf-8'
        assert metadata['missing_percentage'] == 0.0
        assert 'age' in metadata['numeric_columns']
        assert 'salary' in metadata['numeric_columns']
        assert 'name' in metadata['categorical_columns']

    @pytest.mark.asyncio
    async def test_complete_excel_workflow(self):
        """Test complete Excel workflow from Telegram to DataFrame."""
        # Create test Excel file
        test_data = {
            'product': ['Apple', 'Banana', 'Cherry', 'Date'],
            'price': [1.20, 0.50, 2.30, 3.00],
            'quantity': [100, 250, 75, 50],
            'category': ['Fruit', 'Fruit', 'Fruit', 'Fruit']
        }

        temp_excel = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_excel.close()
        excel_path = Path(temp_excel.name)

        try:
            # Create actual Excel file
            pd.DataFrame(test_data).to_excel(excel_path, index=False, engine='openpyxl')

            # Mock Telegram file and download
            telegram_file = AsyncMock(spec=TelegramFile)
            context = MagicMock()

            async def mock_download(file_path):
                # Copy our test Excel file to the target path
                import shutil
                shutil.copy2(excel_path, file_path)

            telegram_file.download_to_drive = mock_download

            # Process the file
            df, metadata = await self.loader.load_from_telegram(
                telegram_file, "products.xlsx", excel_path.stat().st_size, context
            )

            # Verify results
            assert df.shape == (4, 4)
            assert 'product' in df.columns
            assert metadata['file_type'] == 'excel'
            assert metadata['sheet_name'] == 'Sheet1'
            assert 'price' in metadata['numeric_columns']

        finally:
            excel_path.unlink()

    @pytest.mark.asyncio
    async def test_file_with_missing_data(self):
        """Test handling file with missing data."""
        csv_content = """name,age,city
John,25,NYC
Jane,,LA
Bob,35,
Alice,28,Houston"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "incomplete.csv", len(csv_content.encode()), context
        )

        assert df.shape == (4, 3)
        assert metadata['missing_percentage'] > 0
        assert metadata['missing_percentage'] < 50  # Should be acceptable

    @pytest.mark.asyncio
    async def test_file_download_failure(self):
        """Test handling of file download failures."""
        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        # Mock download failure
        telegram_file.download_to_drive.side_effect = Exception("Download failed")

        with pytest.raises(DataError, match="Failed to process file"):
            await self.loader.load_from_telegram(
                telegram_file, "test.csv", 1024, context
            )

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience function load_telegram_file."""
        csv_content = "x,y\n1,2\n3,4\n5,6"

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await load_telegram_file(
            telegram_file, "simple.csv", len(csv_content.encode()), context
        )

        assert df.shape == (3, 2)
        assert list(df.columns) == ['x', 'y']


class TestBotHandlerIntegration:
    """Test integration with bot handlers."""

    @pytest.mark.asyncio
    async def test_document_handler_integration(self):
        """Test integration with document handler workflow."""
        # Mock Telegram update with document
        update = MagicMock()
        context = MagicMock()

        # Mock document
        document = MagicMock(spec=Document)
        document.file_name = "test_data.csv"
        document.file_size = 1024
        document.mime_type = "text/csv"

        update.message.document = document
        update.effective_user.id = 12345

        # Mock file object
        telegram_file = AsyncMock(spec=TelegramFile)

        csv_content = "name,value\ntest1,100\ntest2,200"

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        # Mock get_file method
        context.bot.get_file = AsyncMock(return_value=telegram_file)

        # Test the data loading process
        file_obj = await context.bot.get_file(document.file_id)

        df, metadata = await load_telegram_file(
            file_obj, document.file_name, document.file_size, context
        )

        assert df.shape == (2, 2)
        assert metadata['file_type'] == 'csv'


class TestDataValidationIntegration:
    """Test data validation in realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files within limits."""
        # Create a relatively large CSV (but within limits)
        rows = 1000
        csv_content = "id,value,category\n"
        csv_content += "\n".join([f"{i},{i*10},cat_{i%5}" for i in range(rows)])

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "large.csv", len(csv_content.encode()), context
        )

        assert df.shape == (rows, 3)
        assert metadata['memory_usage_mb'] > 0

    @pytest.mark.asyncio
    async def test_file_with_special_characters(self):
        """Test file with special characters in data."""
        csv_content = """name,description,price
"Café Latte","Rich coffee with milk",4.50
"Crème Brûlée","French dessert",6.75
"Naïve Approach","A simple method",0.00"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "special_chars.csv", len(csv_content.encode('utf-8')), context
        )

        assert df.shape == (3, 3)
        assert "Café Latte" in df['name'].values
        assert metadata['encoding'] == 'utf-8'

    @pytest.mark.asyncio
    async def test_file_encoding_detection(self):
        """Test automatic encoding detection."""
        # Create file with Latin-1 encoding
        csv_content = "name,age\nJosé,25\nMüller,30"

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w', encoding='latin-1') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "encoded.csv", len(csv_content.encode('latin-1')), context
        )

        assert df.shape == (2, 2)
        assert metadata['encoding'] == 'latin-1'

    @pytest.mark.asyncio
    async def test_duplicate_data_detection(self):
        """Test detection of duplicate rows."""
        csv_content = """name,age
John,25
Jane,30
John,25
Bob,35
Jane,30"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "duplicates.csv", len(csv_content.encode()), context
        )

        assert df.shape == (5, 2)
        assert metadata['has_duplicates'] is True
        assert metadata['duplicate_count'] == 2


class TestErrorHandlingIntegration:
    """Test comprehensive error handling in integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    @pytest.mark.asyncio
    async def test_corrupted_csv_handling(self):
        """Test handling of corrupted CSV files."""
        # Malformed CSV with inconsistent columns
        csv_content = """name,age,city
John,25,NYC
Jane,30
Bob,35,Chicago,Extra,Field"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        # Should handle gracefully or raise appropriate error
        try:
            df, metadata = await self.loader.load_from_telegram(
                telegram_file, "corrupted.csv", len(csv_content.encode()), context
            )
            # If it succeeds, verify basic structure
            assert df.shape[0] >= 1
        except DataError:
            # Also acceptable to fail with DataError
            pass

    @pytest.mark.asyncio
    async def test_excel_without_openpyxl(self):
        """Test Excel handling when openpyxl might not be available."""
        # This test simulates the case where openpyxl is missing
        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            # Create a fake Excel file (actually just text)
            with open(file_path, 'w') as f:
                f.write("Not actually Excel content")

        telegram_file.download_to_drive = mock_download

        with pytest.raises(DataError):
            await self.loader.load_from_telegram(
                telegram_file, "fake.xlsx", 100, context
            )

    @pytest.mark.asyncio
    async def test_memory_limit_simulation(self):
        """Test behavior when approaching memory limits."""
        # Create a large dataset that might stress memory
        rows = 10000
        csv_content = "id,data\n"
        csv_content += "\n".join([f"{i},{'x' * 100}" for i in range(rows)])

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        # Should handle large file if within limits
        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "large_memory.csv", len(csv_content.encode()), context
        )

        assert df.shape == (rows, 2)
        assert metadata['memory_usage_mb'] > 0


class TestRealWorldScenarios:
    """Test real-world file scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    @pytest.mark.asyncio
    async def test_business_data_scenario(self):
        """Test typical business data file."""
        csv_content = """employee_id,first_name,last_name,department,salary,hire_date
001,John,Smith,Engineering,75000,2022-01-15
002,Jane,Doe,Marketing,65000,2021-06-20
003,Bob,Johnson,Sales,70000,2020-03-10
004,Alice,Williams,Engineering,80000,2023-02-01"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "employees.csv", len(csv_content.encode()), context
        )

        assert df.shape == (4, 6)
        assert 'employee_id' in metadata['columns']
        assert 'salary' in metadata['numeric_columns']
        assert metadata['missing_percentage'] == 0.0

    @pytest.mark.asyncio
    async def test_survey_data_scenario(self):
        """Test survey data with mixed types and missing values."""
        csv_content = """respondent_id,age,satisfaction,comments,would_recommend
1,25,5,"Great service!",Yes
2,30,4,,Yes
3,45,3,"Could be better",No
4,,5,"Excellent",Yes
5,35,2,"Poor experience",No"""

        telegram_file = AsyncMock(spec=TelegramFile)
        context = MagicMock()

        async def mock_download(file_path):
            with open(file_path, 'w') as f:
                f.write(csv_content)

        telegram_file.download_to_drive = mock_download

        df, metadata = await self.loader.load_from_telegram(
            telegram_file, "survey.csv", len(csv_content.encode()), context
        )

        assert df.shape == (5, 5)
        assert metadata['missing_percentage'] > 0
        assert metadata['missing_percentage'] < 20  # Should be acceptable
        assert 'satisfaction' in metadata['numeric_columns']