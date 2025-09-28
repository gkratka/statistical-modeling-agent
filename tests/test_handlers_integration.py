#!/usr/bin/env python3
"""
TDD Tests for Handlers Integration - Define expected handler behavior FIRST.

This test suite defines how bot handlers should behave when properly integrated
with DataLoader and parser. Tests should FAIL initially until we fix the handlers.
"""

import pytest
import asyncio
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.handlers import start_handler, help_handler, message_handler, document_handler, error_handler


class TestHandlerVersionDetection:
    """Test that handlers are running the correct version of code."""

    def test_handlers_file_does_not_contain_old_development_mode_text(self):
        """Test: handlers.py should not contain old placeholder messages."""
        handlers_path = project_root / "src" / "bot" / "handlers.py"
        content = handlers_path.read_text()

        # These old messages should NOT exist
        forbidden_texts = [
            "File upload handling is under development",
            "I'm currently under development. Soon I'll",
            "For now, I'm confirming that I can receive and respond to messages!"
        ]

        for forbidden_text in forbidden_texts:
            assert forbidden_text not in content, f"Found old placeholder text: '{forbidden_text}'"

    def test_handlers_file_contains_dataloader_integration(self):
        """Test: handlers.py should contain DataLoader integration code."""
        handlers_path = project_root / "src" / "bot" / "handlers.py"
        content = handlers_path.read_text()

        # Required integration elements
        required_elements = [
            "from src.processors.data_loader import DataLoader",
            "loader.load_from_telegram",
            "get_data_summary",
            "DataLoader v2.0"
        ]

        for element in required_elements:
            assert element in content, f"Missing DataLoader integration: '{element}'"

    def test_handlers_file_has_diagnostic_logging(self):
        """Test: handlers.py should have diagnostic logging for debugging."""
        handlers_path = project_root / "src" / "bot" / "handlers.py"
        content = handlers_path.read_text()

        # Should have diagnostic markers we can detect in logs
        diagnostic_markers = [
            "HANDLERS DIAGNOSTIC",
            "DATALOADER IMPORT",
            "SENDING MESSAGE"
        ]

        found_markers = [marker for marker in diagnostic_markers if marker in content]
        assert len(found_markers) >= 2, f"Need at least 2 diagnostic markers, found: {found_markers}"


class TestStartAndHelpHandlers:
    """Test basic command handlers work correctly."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = AsyncMock()
        update.effective_user.id = 12345
        update.effective_user.username = "testuser"
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock Telegram context."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_start_handler_mentions_dataloader_capabilities(self, mock_update, mock_context):
        """Test: start handler should mention DataLoader capabilities, not development mode."""
        await start_handler(mock_update, mock_context)

        sent_message = mock_update.message.reply_text.call_args[0][0]

        # Should mention actual capabilities
        assert "Statistical analysis" in sent_message or "CSV file" in sent_message
        assert "Development Mode" not in sent_message

    @pytest.mark.asyncio
    async def test_help_handler_provides_dataloader_instructions(self, mock_update, mock_context):
        """Test: help handler should provide instructions for using DataLoader."""
        await help_handler(mock_update, mock_context)

        sent_message = mock_update.message.reply_text.call_args[0][0]

        # Should mention data upload and analysis
        assert "CSV" in sent_message or "upload" in sent_message
        assert "statistical" in sent_message.lower() or "analysis" in sent_message.lower()
        assert "Development Mode" not in sent_message


class TestMessageHandlerBehavior:
    """Test message_handler properly processes user messages."""

    @pytest.fixture
    def mock_update_with_question(self):
        """Create mock update with data question."""
        update = AsyncMock()
        update.effective_user.id = 12345
        update.effective_user.username = "testuser"
        update.message.text = "what columns are in my data?"
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context_with_data(self):
        """Create mock context with user data."""
        context = AsyncMock()
        context.user_data = {
            'data_12345': {
                'dataframe': pd.DataFrame({
                    'age': [25, 30, 35],
                    'income': [50000, 60000, 70000],
                    'satisfaction': [7, 8, 6]
                }),
                'metadata': {
                    'columns': ['age', 'income', 'satisfaction'],
                    'shape': (3, 3)
                },
                'file_name': 'user_survey.csv'
            }
        }
        return context

    @pytest.fixture
    def mock_context_no_data(self):
        """Create mock context without user data."""
        context = AsyncMock()
        context.user_data = {}
        return context

    @pytest.mark.asyncio
    async def test_message_handler_answers_data_questions_when_data_exists(self, mock_update_with_question, mock_context_with_data):
        """Test: message_handler should answer questions about uploaded data."""
        await message_handler(mock_update_with_question, mock_context_with_data)

        sent_message = mock_update_with_question.message.reply_text.call_args[0][0]

        # Should mention the actual columns from the data
        assert "age" in sent_message or "income" in sent_message or "satisfaction" in sent_message
        assert "Development Mode" not in sent_message
        assert "I'm currently under development" not in sent_message

    @pytest.mark.asyncio
    async def test_message_handler_guides_upload_when_no_data(self, mock_update_with_question, mock_context_no_data):
        """Test: message_handler should guide user to upload data when none exists."""
        await message_handler(mock_update_with_question, mock_context_no_data)

        sent_message = mock_update_with_question.message.reply_text.call_args[0][0]

        # Should guide user to upload data
        assert "upload" in sent_message.lower() or "csv" in sent_message.lower() or "file" in sent_message.lower()
        assert "Development Mode" not in sent_message

    @pytest.mark.asyncio
    async def test_message_handler_integrates_with_parser(self):
        """Test: message_handler should integrate with parser for analysis requests."""
        update = AsyncMock()
        update.effective_user.id = 12345
        update.message.text = "calculate mean for age column"
        update.message.reply_text = AsyncMock()

        context = AsyncMock()
        context.user_data = {
            'data_12345': {
                'dataframe': pd.DataFrame({'age': [25, 30, 35]}),
                'metadata': {'columns': ['age']},
                'file_name': 'data.csv'
            }
        }

        # Mock parser integration
        with patch('src.bot.handlers.RequestParser') as MockParser:
            mock_parser = MockParser.return_value
            mock_parser.parse.return_value = MagicMock(
                task_type="stats",
                operation="descriptive_stats",
                parameters={'columns': ['age'], 'statistics': ['mean']}
            )

            await message_handler(update, context)

            sent_message = update.message.reply_text.call_args[0][0]

            # Should NOT show development mode
            assert "Development Mode" not in sent_message
            assert "I'm currently under development" not in sent_message

            # Should show some kind of analysis result or processing message
            assert len(sent_message) > 10  # Should have substantial content


class TestDocumentHandlerBehavior:
    """Test document_handler properly uses DataLoader."""

    @pytest.fixture
    def mock_csv_upload(self):
        """Create mock CSV upload."""
        update = AsyncMock()
        update.effective_user.id = 12345
        update.effective_user.username = "testuser"
        update.message.document.file_name = "survey_data.csv"
        update.message.document.file_size = 5000
        update.message.document.file_id = "test_file_123"
        update.message.reply_text = AsyncMock()
        update.message.edit_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        context = AsyncMock()
        context.bot.get_file = AsyncMock()
        context.user_data = {}
        return context

    @pytest.mark.asyncio
    async def test_document_handler_shows_dataloader_processing_message(self, mock_csv_upload, mock_context):
        """Test: document_handler should show DataLoader processing message."""
        mock_file = AsyncMock()
        mock_context.bot.get_file.return_value = mock_file

        with patch('src.bot.handlers.DataLoader') as MockDataLoader:
            mock_loader = MockDataLoader.return_value
            mock_loader.load_from_telegram.return_value = (
                pd.DataFrame({'col1': [1, 2]}),
                {'shape': (2, 1), 'columns': ['col1'], 'memory_usage_mb': 0.01}
            )
            mock_loader.get_data_summary.return_value = "📊 Data loaded successfully"

            await document_handler(mock_csv_upload, mock_context)

            # Find the processing message
            processing_calls = [call for call in mock_csv_upload.message.reply_text.call_args_list
                              if "Processing your file" in call.args[0]]
            assert len(processing_calls) > 0, "No 'Processing your file' message found"

            processing_message = processing_calls[0].args[0]
            assert "DataLoader v2.0 active" in processing_message

    @pytest.mark.asyncio
    async def test_document_handler_never_shows_old_development_message(self, mock_csv_upload, mock_context):
        """Test: document_handler should NEVER show old development mode message."""
        mock_file = AsyncMock()
        mock_context.bot.get_file.return_value = mock_file

        with patch('src.bot.handlers.DataLoader') as MockDataLoader:
            mock_loader = MockDataLoader.return_value
            mock_loader.load_from_telegram.return_value = (
                pd.DataFrame({'test': [1]}),
                {'shape': (1, 1), 'columns': ['test'], 'memory_usage_mb': 0.01}
            )
            mock_loader.get_data_summary.return_value = "Data loaded"

            await document_handler(mock_csv_upload, mock_context)

            # Check ALL messages sent
            all_messages = []
            for call in mock_csv_upload.message.reply_text.call_args_list:
                all_messages.append(call.args[0])
            for call in mock_csv_upload.message.edit_text.call_args_list:
                all_messages.append(call.args[0])

            # NONE should contain old development text
            for message in all_messages:
                assert "File upload handling is under development" not in message
                assert "Soon I'll" not in message
                assert "For now, I'm confirming" not in message

    @pytest.mark.asyncio
    async def test_document_handler_handles_dataloader_errors(self, mock_csv_upload, mock_context):
        """Test: document_handler should handle DataLoader errors gracefully."""
        mock_file = AsyncMock()
        mock_context.bot.get_file.return_value = mock_file

        with patch('src.bot.handlers.DataLoader') as MockDataLoader:
            mock_loader = MockDataLoader.return_value
            mock_loader.load_from_telegram.side_effect = Exception("File too large")

            await document_handler(mock_csv_upload, mock_context)

            # Should send error message, not development mode
            error_calls = [call for call in mock_csv_upload.message.reply_text.call_args_list
                          if "error" in call.args[0].lower() or "Error" in call.args[0]]
            assert len(error_calls) > 0, "No error message found"

            # Error message should not mention development mode
            error_message = error_calls[0].args[0]
            assert "Development Mode" not in error_message


class TestErrorHandlerBehavior:
    """Test error_handler provides useful error information."""

    @pytest.fixture
    def mock_update(self):
        """Create mock update."""
        update = AsyncMock()
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context_with_error(self):
        """Create mock context with error."""
        context = AsyncMock()
        context.error = Exception("Test error")
        return context

    @pytest.mark.asyncio
    async def test_error_handler_provides_helpful_message(self, mock_update, mock_context_with_error):
        """Test: error_handler should provide helpful error message."""
        await error_handler(mock_update, mock_context_with_error)

        sent_message = mock_update.effective_message.reply_text.call_args[0][0]

        # Should mention error occurred and provide guidance
        assert "error" in sent_message.lower() or "Error" in sent_message
        assert len(sent_message) > 20  # Should be substantial message
        assert "Development Mode" not in sent_message


class TestHandlerImportIntegrity:
    """Test that handlers properly import and use required modules."""

    def test_handlers_imports_dataloader_correctly(self):
        """Test: handlers.py should import DataLoader correctly."""
        try:
            # This import should work without errors
            from src.bot.handlers import document_handler
            from src.processors.data_loader import DataLoader

            # DataLoader should be accessible
            loader = DataLoader()
            assert hasattr(loader, 'load_from_telegram')

        except ImportError as e:
            pytest.fail(f"Handler imports failed: {e}")

    def test_handlers_imports_exceptions_correctly(self):
        """Test: handlers.py should import exceptions correctly."""
        try:
            from src.bot.handlers import document_handler
            from src.utils.exceptions import ValidationError, DataError

            # Should be able to create exceptions
            assert issubclass(ValidationError, Exception)
            assert issubclass(DataError, Exception)

        except ImportError as e:
            pytest.fail(f"Exception imports failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output to see which ones fail
    pytest.main([__file__, "-v", "--tb=short"])