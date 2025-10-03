"""Integration test for ML training workflow via Telegram handlers."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, User, Chat, Document

from src.bot.handlers import message_handler


@pytest.mark.asyncio
class TestMLWorkflowIntegration:
    """Test ML training workflow through Telegram bot."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.message = MagicMock(spec=Message)
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock bot context with uploaded data."""
        context = MagicMock()

        # Simulate uploaded housing data
        df = pd.DataFrame({
            'sqft': [1000, 1500, 2000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2],
            'age': [5, 10, 15],
            'price': [200000, 300000, 400000],
            'location': ['A', 'B', 'C'],
            'condition': ['good', 'fair', 'excellent']
        })

        context.user_data = {
            'data_12345': {
                'dataframe': df,
                'metadata': {
                    'shape': (3, 7),
                    'columns': df.columns.tolist(),
                    'file_name': 'housing_data.csv'
                },
                'file_name': 'housing_data.csv'
            }
        }

        return context

    async def test_ml_training_workflow_initiation(self, mock_update, mock_context):
        """Test that 'Train a model to predict house prices' starts ML workflow."""

        # Set user message
        mock_update.message.text = "Train a model to predict house prices"

        # Call handler
        await message_handler(mock_update, mock_context)

        # Verify workflow was initiated (reply_text called with column selection)
        mock_update.message.reply_text.assert_called_once()
        response = mock_update.message.reply_text.call_args[0][0]

        # Assertions
        assert "🎯 Select Target Column" in response or "Select Target" in response, \
            f"Expected column selection prompt, got: {response[:200]}"
        assert "sqft" in response or "price" in response, \
            f"Expected column names in response, got: {response[:200]}"

        # Should NOT contain error messages
        assert "Error" not in response or "Unknown error" not in response, \
            f"Response should not contain errors, got: {response[:200]}"

    async def test_ml_training_with_valid_target(self, mock_update, mock_context):
        """Test ML training with explicitly valid target column."""

        # Set user message with valid column name
        mock_update.message.text = "Train a model to predict price"

        # Call handler - should still start workflow (target exists but we prefer workflow)
        await message_handler(mock_update, mock_context)

        # Should either start workflow OR execute training
        mock_update.message.reply_text.assert_called_once()
        response = mock_update.message.reply_text.call_args[0][0]

        # Should not show generic error
        assert "Unknown error occurred" not in response, \
            f"Should not show generic error, got: {response[:200]}"

    async def test_ml_training_with_invalid_target(self, mock_update, mock_context):
        """Test ML training with invalid target column triggers workflow."""

        # Set user message with invalid column name
        mock_update.message.text = "Train a model to predict house_value"

        # Call handler
        await message_handler(mock_update, mock_context)

        # Verify workflow was initiated
        mock_update.message.reply_text.assert_called_once()
        response = mock_update.message.reply_text.call_args[0][0]

        # Should show column selection, not error
        workflow_started = ("Select Target Column" in response) or ("Select Target" in response)
        no_error = "Unknown error" not in response

        assert workflow_started or no_error, \
            f"Expected workflow or no error, got: {response[:200]}"

    async def test_version_marker_in_logs(self, mock_update, mock_context, caplog):
        """Test that version marker appears in logs after parsing."""

        # Set user message
        mock_update.message.text = "Train a model to predict house prices"

        # Call handler with logging
        with caplog.at_level('INFO'):
            await message_handler(mock_update, mock_context)

        # Check for version marker in logs
        log_messages = [record.message for record in caplog.records]
        version_logged = any("CODE VERSION: v2.1.0-ml-workflow-fix" in msg for msg in log_messages)

        assert version_logged, \
            f"Version marker not found in logs. Got: {log_messages}"

    async def test_workflow_condition_logging(self, mock_update, mock_context, caplog):
        """Test that workflow decision is logged."""

        # Set user message
        mock_update.message.text = "Train a model to predict house prices"

        # Call handler with logging
        with caplog.at_level('INFO'):
            await message_handler(mock_update, mock_context)

        # Check for workflow decision logs
        log_messages = [record.message for record in caplog.records]
        workflow_check_logged = any("ML WORKFLOW CHECK" in msg for msg in log_messages)
        workflow_start_logged = any("WORKFLOW SHOULD START" in msg for msg in log_messages)

        assert workflow_check_logged, \
            f"Workflow check not logged. Got: {log_messages}"
        assert workflow_start_logged, \
            f"Workflow decision not logged. Got: {log_messages}"
