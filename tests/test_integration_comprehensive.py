#!/usr/bin/env python3
"""
Comprehensive integration tests for Telegram bot handlers and DataLoader.

Consolidates test_bot_instance_fix.py, test_handlers_diagnostic.py, and test_dataloader_integration.py
using pytest parameterization to reduce code duplication while maintaining full test coverage.
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from telegram import Document, Message, Update, User

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot import handlers
from src.processors.data_loader import DataLoader
from src.utils.exceptions import ValidationError, DataError


class TestHandlerDataLoaderIntegration:
    """Comprehensive integration tests for handlers and DataLoader."""

    @pytest.fixture
    def sample_data(self):
        """Standard test dataframe."""
        return pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'age': [25, 30, 35, 28],
            'salary': [50000, 60000, 70000, 55000],
            'department': ['Engineering', 'Marketing', 'Sales', 'HR']
        })

    @pytest.fixture
    def mock_user(self):
        """Standard mock user."""
        return User(id=12345, first_name="Test", is_bot=False)

    @pytest.fixture
    def mock_context(self):
        """Standard mock context."""
        context = MagicMock()
        context.user_data = {}
        context.bot.get_file = AsyncMock()
        return context

    @pytest.fixture
    def mock_context_with_data(self, sample_data):
        """Mock context with user data loaded."""
        context = MagicMock()
        context.user_data = {
            'data_12345': {
                'dataframe': sample_data,
                'metadata': {'columns': sample_data.columns.tolist(), 'shape': sample_data.shape},
                'file_name': 'test_data.csv'
            }
        }
        return context

    @pytest.mark.parametrize("handler_name,handler_func,message_text", [
        ("start_handler", handlers.start_handler, "/start"),
        ("help_handler", handlers.help_handler, "/help"),
        ("message_handler", handlers.message_handler, "What columns are available?"),
    ])
    @pytest.mark.asyncio
    async def test_handlers_no_development_mode_messages(self, handler_name, handler_func, message_text, mock_user, mock_context):
        """Test that no handler sends old development mode messages."""
        message = Message(
            message_id=1,
            date=time.time(),
            chat=mock_user,
            from_user=mock_user,
            text=message_text
        )
        message.reply_text = AsyncMock()
        update = Update(update_id=1, message=message)

        await handler_func(update, mock_context)

        # Verify no development mode messages
        forbidden_phrases = ["Development Mode", "under development", "not yet implemented", "Coming soon"]
        for call in message.reply_text.call_args_list:
            response_text = call.args[0] if call.args else ""
            for phrase in forbidden_phrases:
                assert phrase not in response_text, f"{handler_name} sent forbidden phrase: '{phrase}'"

    @pytest.mark.asyncio
    async def test_document_handler_full_integration(self, mock_user, mock_context, sample_data):
        """Test complete document handler integration with DataLoader."""
        document = Document(
            file_id="test_file_id",
            file_unique_id="test_unique_id",
            file_name="test.csv",
            mime_type="text/csv",
            file_size=1024
        )

        message = Message(
            message_id=1,
            date=time.time(),
            chat=mock_user,
            from_user=mock_user,
            document=document
        )
        message.reply_text = AsyncMock()
        message.edit_text = AsyncMock()
        update = Update(update_id=1, message=message)

        # Mock successful DataLoader processing
        mock_file = AsyncMock()
        mock_context.bot.get_file.return_value = mock_file

        with patch('src.bot.handlers.DataLoader') as MockDataLoader:
            mock_loader = MockDataLoader.return_value
            mock_loader.load_from_telegram.return_value = (sample_data, {
                'shape': sample_data.shape,
                'columns': sample_data.columns.tolist(),
                'memory_usage_mb': 0.1
            })
            mock_loader.get_data_summary.return_value = "📊 **Data Successfully Loaded**"

            await handlers.document_handler(update, mock_context)

            # Verify DataLoader was used correctly
            MockDataLoader.assert_called_once()
            mock_loader.load_from_telegram.assert_called_once()

            # Verify data was stored
            assert f'data_{mock_user.id}' in mock_context.user_data
            stored_data = mock_context.user_data[f'data_{mock_user.id}']
            assert 'dataframe' in stored_data
            assert 'metadata' in stored_data

    @pytest.mark.parametrize("file_size,should_fail", [
        (1000, False),  # Normal file
        (5 * 1024 * 1024, False),  # 5MB file
        (15 * 1024 * 1024, True),  # 15MB file (too large)
    ])
    def test_dataloader_file_size_validation(self, file_size, should_fail):
        """Test DataLoader validates file sizes correctly."""
        loader = DataLoader()

        if should_fail:
            with pytest.raises(ValidationError, match="too large"):
                loader._validate_file_metadata("test.csv", file_size)
        else:
            loader._validate_file_metadata("test.csv", file_size)  # Should not raise

    @pytest.mark.parametrize("extension,should_pass", [
        (".csv", True),
        (".xlsx", True),
        (".xls", True),
        (".pdf", False),
        (".txt", False),
        (".json", False),
    ])
    def test_dataloader_extension_validation(self, extension, should_pass):
        """Test DataLoader validates file extensions correctly."""
        loader = DataLoader()
        filename = f"test{extension}"

        if should_pass:
            loader._validate_file_metadata(filename, 1000)  # Should not raise
        else:
            with pytest.raises(ValidationError, match="Unsupported file type"):
                loader._validate_file_metadata(filename, 1000)

    def test_dataloader_summary_generation(self, sample_data):
        """Test DataLoader generates proper data summaries."""
        loader = DataLoader()
        metadata = loader._validate_dataframe(sample_data, "test.csv")
        summary = loader.get_data_summary(sample_data, metadata)

        expected_elements = [
            "Data Successfully Loaded",
            f"{len(sample_data)} rows",
            f"{len(sample_data.columns)} columns",
            "Missing data:",
            "File Type:"
        ]

        for element in expected_elements:
            assert element in summary, f"Missing element in summary: {element}"

    @pytest.mark.asyncio
    async def test_message_handler_data_responses(self, mock_user, mock_context_with_data):
        """Test message handler provides helpful responses for data queries."""
        test_messages = [
            "What columns are available?",
            "Show me the data",
            "What's in this file?",
            "Tell me about the dataset"
        ]

        for text in test_messages:
            message = Message(
                message_id=1,
                date=time.time(),
                chat=mock_user,
                from_user=mock_user,
                text=text
            )
            message.reply_text = AsyncMock()
            update = Update(update_id=1, message=message)

            await handlers.message_handler(update, mock_context_with_data)

            # Should provide helpful response about data
            assert message.reply_text.called
            response = message.reply_text.call_args[0][0]
            assert any(col in response for col in ['name', 'age', 'salary', 'department'])


class TestBotHealthAndDiagnostics:
    """Health check and diagnostic tests for bot components."""

    def test_handlers_version_tracking(self):
        """Test that handlers module has proper version tracking."""
        assert hasattr(handlers, 'HANDLERS_VERSION')
        assert handlers.HANDLERS_VERSION == "v2.0"
        assert hasattr(handlers, 'BOT_VERSION')
        assert "DataLoader" in handlers.BOT_VERSION

    @pytest.mark.parametrize("attribute_name", [
        "start_handler",
        "help_handler",
        "message_handler",
        "document_handler",
        "diagnostic_handler",
        "error_handler"
    ])
    def test_handler_functions_exist(self, attribute_name):
        """Test that all expected handler functions exist."""
        assert hasattr(handlers, attribute_name)
        handler_func = getattr(handlers, attribute_name)
        assert asyncio.iscoroutinefunction(handler_func)

    def test_dataloader_import_available(self):
        """Test that DataLoader can be imported and instantiated."""
        loader = DataLoader()
        required_methods = [
            'load_from_telegram',
            'get_data_summary',
            '_validate_file_metadata',
            '_validate_dataframe'
        ]

        for method in required_methods:
            assert hasattr(loader, method), f"DataLoader missing method: {method}"

    @pytest.mark.parametrize("forbidden_text", [
        "Development Mode",
        "File upload handling is under development",
        "This feature is not yet implemented",
        "Coming soon"
    ])
    def test_no_forbidden_strings_in_handlers(self, forbidden_text):
        """Test that handlers source contains no forbidden placeholder text."""
        import inspect
        source = inspect.getsource(handlers)
        assert forbidden_text not in source, f"Found forbidden text: '{forbidden_text}'"

    def test_diagnostic_logging_present(self):
        """Test that diagnostic logging is present in handlers."""
        import inspect
        source = inspect.getsource(handlers)
        assert "DIAGNOSTIC" in source
        assert "logger.info" in source

    @pytest.mark.asyncio
    async def test_diagnostic_handler_functionality(self, mock_user):
        """Test diagnostic handler provides system information."""
        message = Message(
            message_id=1,
            date=time.time(),
            chat=mock_user,
            from_user=mock_user,
            text="/diagnostic"
        )
        message.reply_text = AsyncMock()
        update = Update(update_id=1, message=message)
        context = MagicMock()

        await handlers.diagnostic_handler(update, context)

        assert message.reply_text.called
        response = message.reply_text.call_args[0][0]

        # Should contain diagnostic information
        expected_info = ["Diagnostic Information", "Handlers Version", "DataLoader"]
        for info in expected_info:
            assert info in response, f"Missing diagnostic info: {info}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])