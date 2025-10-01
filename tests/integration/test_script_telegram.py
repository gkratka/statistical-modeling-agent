"""
Integration tests for script generation and execution via Telegram.

Tests the complete pipeline from Telegram commands to script execution results.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.script_handler import ScriptHandler
from src.core.parser import RequestParser
from src.core.orchestrator import TaskOrchestrator
from src.bot.handlers import message_handler
from src.utils.exceptions import ParseError


class TestScriptTelegramIntegration:
    """Test script generation and execution integration with Telegram."""

    def setup_method(self):
        """Set up test environment."""
        self.parser = RequestParser()
        self.orchestrator = TaskOrchestrator()
        self.script_handler = ScriptHandler(self.parser, self.orchestrator)

        # Create mock Telegram objects
        self.mock_user = User(id=123, is_bot=False, first_name="Test")
        self.mock_chat = Chat(id=456, type="private")

    def _create_mock_update(self, message_text: str) -> Update:
        """Create a mock Telegram update with message text."""
        mock_message = MagicMock(spec=Message)
        mock_message.text = message_text
        mock_message.reply_text = AsyncMock()

        mock_update = MagicMock(spec=Update)
        mock_update.message = mock_message
        mock_update.effective_user = self.mock_user
        mock_update.effective_chat = self.mock_chat

        return mock_update

    def _create_mock_context(self, has_data: bool = True) -> ContextTypes.DEFAULT_TYPE:
        """Create a mock Telegram context."""
        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        mock_context.args = []
        mock_context.bot_data = {'script_handler': self.script_handler}

        if has_data:
            # Mock user data with sample dataframe
            sample_data = pd.DataFrame({
                'sales': [100, 200, 300, 400, 500],
                'profit': [10, 25, 35, 50, 75],
                'quantity': [5, 10, 15, 20, 25]
            })
            mock_context.user_data = {
                'dataframe': sample_data,
                'metadata': {
                    'columns': ['sales', 'profit', 'quantity'],
                    'shape': (5, 3),
                    'numeric_columns': ['sales', 'profit', 'quantity']
                },
                'file_name': 'test_data.csv'
            }
        else:
            mock_context.user_data = {}

        return mock_context

    async def test_script_command_basic(self):
        """Test basic /script command showing template listing."""
        update = self._create_mock_update("/script")
        context = self._create_mock_context()

        await self.script_handler.script_command_handler(update, context)

        # Verify template listing was sent
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        message_sent = call_args[0][0]

        assert "Available Script Templates" in message_sent
        assert "/script descriptive" in message_sent
        assert "/script correlation" in message_sent
        assert "/script train_classifier" in message_sent

    @patch('src.core.orchestrator.TaskOrchestrator.execute_task')
    async def test_script_command_with_operation(self, mock_execute):
        """Test /script command with specific operation."""
        # Mock successful execution
        mock_execute.return_value = {
            'success': True,
            'output': '{"mean": 260.0, "std": 158.11}',
            'execution_time': 0.123,
            'memory_usage': 45,
            'metadata': {
                'operation': 'descriptive',
                'template_used': 'descriptive.j2',
                'security_validated': True
            }
        }

        update = self._create_mock_update("/script descriptive")
        context = self._create_mock_context()
        context.args = ['descriptive']

        await self.script_handler.script_command_handler(update, context)

        # Verify script was executed
        mock_execute.assert_called_once()
        task = mock_execute.call_args[0][0]
        assert task.task_type == "script"
        assert task.operation == "descriptive"

        # Verify response was formatted and sent
        update.message.reply_text.assert_called()
        response_message = update.message.reply_text.call_args[0][0]
        assert "Script Executed Successfully" in response_message
        assert "descriptive" in response_message

    async def test_script_command_no_data(self):
        """Test /script command when user has no data uploaded."""
        update = self._create_mock_update("/script descriptive")
        context = self._create_mock_context(has_data=False)
        context.args = ['descriptive']

        await self.script_handler.script_command_handler(update, context)

        # Verify error message about missing data
        update.message.reply_text.assert_called_once()
        response_message = update.message.reply_text.call_args[0][0]
        assert "upload a data file first" in response_message.lower()

    @patch('src.core.orchestrator.TaskOrchestrator.execute_task')
    async def test_natural_language_script_generation(self, mock_execute):
        """Test natural language script generation through message handler."""
        # Mock successful execution
        mock_execute.return_value = {
            'success': True,
            'output': '{"correlation_matrix": {"sales": {"profit": 0.95}}}',
            'execution_time': 0.156,
            'memory_usage': 52,
            'metadata': {
                'operation': 'correlation',
                'template_used': 'correlation.j2',
                'security_validated': True
            }
        }

        update = self._create_mock_update("Generate a script for correlation analysis")
        context = self._create_mock_context()

        # Mock the imports within message_handler
        with patch('src.bot.handlers.RequestParser') as mock_parser_class, \
             patch('src.bot.handlers.TaskOrchestrator') as mock_orchestrator_class, \
             patch('src.bot.handlers.TelegramResultFormatter') as mock_formatter_class, \
             patch('src.bot.handlers.ScriptHandler') as mock_script_handler_class:

            # Set up mocks
            mock_parser = mock_parser_class.return_value
            mock_orchestrator = mock_orchestrator_class.return_value
            mock_formatter = mock_formatter_class.return_value
            mock_script_handler = mock_script_handler_class.return_value

            # Mock parser to return script task
            mock_task = MagicMock()
            mock_task.task_type = "script"
            mock_task.operation = "correlation"
            mock_parser.parse_request.return_value = mock_task

            # Mock orchestrator execution
            mock_orchestrator.execute_task.return_value = mock_execute.return_value

            # Mock script handler formatting
            mock_script_handler.format_script_results.return_value = "✅ Script executed successfully"

            # Mock the safe_get_user_data function
            with patch('src.bot.handlers.safe_get_user_data') as mock_safe_get:
                mock_safe_get.return_value = context.user_data

                # Execute message handler
                await message_handler(update, context)

                # Verify script was executed
                mock_orchestrator.execute_task.assert_called_once()
                mock_script_handler.format_script_results.assert_called_once()

                # Verify response was sent
                update.message.reply_text.assert_called()

    @patch('src.core.orchestrator.TaskOrchestrator.execute_task')
    async def test_script_with_parameters(self, mock_execute):
        """Test script generation with specific parameters."""
        # Mock successful execution
        mock_execute.return_value = {
            'success': True,
            'output': '{"correlation": {"sales": {"profit": 0.95}}}',
            'execution_time': 0.089,
            'memory_usage': 38,
            'metadata': {
                'operation': 'correlation',
                'template_used': 'correlation.j2',
                'security_validated': True
            }
        }

        update = self._create_mock_update("/script correlation for sales and profit")
        context = self._create_mock_context()
        context.args = ['correlation', 'for', 'sales', 'and', 'profit']

        await self.script_handler.script_command_handler(update, context)

        # Verify task was parsed with parameters
        mock_execute.assert_called_once()
        task = mock_execute.call_args[0][0]
        assert task.task_type == "script"
        assert task.operation == "correlation"
        assert "columns" in task.parameters
        assert "sales" in task.parameters["columns"]
        assert "profit" in task.parameters["columns"]

    async def test_script_execution_failure(self):
        """Test handling of script execution failures."""
        with patch('src.core.orchestrator.TaskOrchestrator.execute_task') as mock_execute:
            # Mock execution failure
            mock_execute.return_value = {
                'success': False,
                'error': 'Script execution failed: Invalid column name',
                'execution_time': 0.045,
                'memory_usage': 20
            }

            update = self._create_mock_update("/script descriptive")
            context = self._create_mock_context()
            context.args = ['descriptive']

            await self.script_handler.script_command_handler(update, context)

            # Verify error message was sent
            update.message.reply_text.assert_called()
            response_message = update.message.reply_text.call_args[0][0]
            assert "Script Execution Failed" in response_message
            assert "Invalid column name" in response_message

    async def test_parse_error_handling(self):
        """Test handling of parse errors in script requests."""
        update = self._create_mock_update("/script invalid_operation")
        context = self._create_mock_context()
        context.args = ['invalid_operation']

        await self.script_handler.script_command_handler(update, context)

        # Verify parse error message was sent
        update.message.reply_text.assert_called()
        response_message = update.message.reply_text.call_args[0][0]
        assert "Parse Error" in response_message or "Invalid script command" in response_message

    async def test_script_result_formatting(self):
        """Test formatting of script execution results."""
        # Test successful result formatting
        result = {
            'success': True,
            'output': '{"mean": 260.0, "std": 158.11, "correlation_matrix": {"sales": {"profit": 0.95}}}',
            'execution_time': 0.123,
            'memory_usage': 45,
            'script_hash': 'abc123',
            'metadata': {
                'operation': 'descriptive',
                'template_used': 'descriptive.j2',
                'security_validated': True,
                'resource_limits': {'timeout': 30, 'memory_limit': 2048}
            }
        }

        formatted_message = self.script_handler.format_script_results(result)

        assert "Script Executed Successfully" in formatted_message
        assert "**Operation**: descriptive" in formatted_message
        assert "**Template**: descriptive.j2" in formatted_message
        assert "260.0" in formatted_message  # mean value
        assert "Security Validated: True" in formatted_message

    async def test_script_ml_operations(self):
        """Test script generation for ML operations."""
        with patch('src.core.orchestrator.TaskOrchestrator.execute_task') as mock_execute:
            # Mock successful ML script execution
            mock_execute.return_value = {
                'success': True,
                'output': '{"model_accuracy": 0.85, "features_used": ["sales", "quantity"]}',
                'execution_time': 2.456,
                'memory_usage': 128,
                'metadata': {
                    'operation': 'train_classifier',
                    'template_used': 'train_classifier.j2',
                    'security_validated': True
                }
            }

            update = self._create_mock_update("Generate script to train a classifier")
            context = self._create_mock_context()

            # Test through script generation handler
            await self.script_handler.script_generation_handler(update, context)

            # Verify ML script was executed
            mock_execute.assert_called_once()
            task = mock_execute.call_args[0][0]
            assert task.task_type == "script"
            assert task.operation == "train_classifier"

    def test_template_listing_format(self):
        """Test format of template listing."""
        template_listing = self.script_handler._get_template_listing()

        assert "Available Script Templates" in template_listing
        assert "Statistical Analysis" in template_listing
        assert "Machine Learning" in template_listing
        assert "/script descriptive" in template_listing
        assert "/script correlation" in template_listing
        assert "/script train_classifier" in template_listing
        assert "Usage Examples" in template_listing
        assert "natural language" in template_listing

    async def test_integration_error_handling(self):
        """Test comprehensive error handling in integration."""
        test_cases = [
            ("/script", "no operation", "Available Script Templates"),
            ("/script unknown", "unknown operation", "Invalid script command"),
            ("Generate script", "no data", "upload a data file first")
        ]

        for command, description, expected_response in test_cases:
            update = self._create_mock_update(command)
            context = self._create_mock_context(has_data="no data" not in description)

            if "no operation" in description:
                context.args = []
            elif "unknown operation" in description:
                context.args = ["unknown"]

            if "Generate script" in command:
                await self.script_handler.script_generation_handler(update, context)
            else:
                await self.script_handler.script_command_handler(update, context)

            # Verify appropriate response
            assert update.message.reply_text.called
            response = update.message.reply_text.call_args[0][0]
            assert expected_response.lower() in response.lower()

            # Reset mock for next test
            update.message.reply_text.reset_mock()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])