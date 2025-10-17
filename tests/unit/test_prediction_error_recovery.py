"""
TDD Tests for Prediction Workflow Error Recovery.

These tests verify that users can recover from path validation errors without
losing their workflow progress.

Background:
Bug discovered where users got stuck in AWAITING_FILE_PATH state after validation
failures. The bot would misinterpret subsequent input (like feature lists) as file
paths, causing confusing error messages.

Bug Fix: Add recovery buttons (Try Again, Different Data Source, Cancel) to path
validation errors, allowing users to retry, go back, or start over.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery, InlineKeyboardMarkup, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.ml_handlers.prediction_handlers import PredictionHandler
from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = AsyncMock(spec=StateManager)

    # Mock session object
    mock_session = MagicMock()
    mock_session.user_id = 12345
    mock_session.current_state = MLPredictionState.AWAITING_FILE_PATH.value
    mock_session.workflow_type = WorkflowType.ML_PREDICTION
    mock_session.uploaded_data = None
    mock_session.file_path = None
    mock_session.selections = {}

    manager.get_session.return_value = mock_session
    manager.get_or_create_session.return_value = mock_session
    manager.transition_state = AsyncMock()
    manager.reset_session = AsyncMock()

    return manager


@pytest.fixture
def mock_data_loader():
    """Create mock data loader."""
    loader = MagicMock(spec=DataLoader)
    loader.allowed_directories = ["/Users/test/data", "/tmp/datasets"]
    loader.local_max_size_mb = 1000
    loader.local_extensions = [".csv", ".xlsx", ".parquet"]
    return loader


@pytest.fixture
def mock_path_validator():
    """Create mock path validator."""
    validator = MagicMock(spec=PathValidator)
    return validator


@pytest.fixture
def prediction_handler(mock_state_manager, mock_data_loader, mock_path_validator):
    """Create prediction handler with mocked dependencies."""
    return PredictionHandler(
        state_manager=mock_state_manager,
        data_loader=mock_data_loader,
        path_validator=mock_path_validator
    )


@pytest.fixture
def mock_update_with_message():
    """Create mock update with message."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = AsyncMock(spec=Message)
    update.message.text = "/tmp/invalid/path.csv"
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_update_with_callback():
    """Create mock update with callback query."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.callback_query = AsyncMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


class TestPathValidationErrorRecovery:
    """Test recovery buttons appear on path validation errors."""

    @pytest.mark.asyncio
    async def test_path_validation_failure_shows_recovery_buttons(
        self,
        prediction_handler,
        mock_update_with_message,
        mock_context,
        mock_path_validator
    ):
        """
        Test that path validation failure shows error with recovery buttons.

        This is the core bug fix - when validation fails, user should see:
        1. Clear error message
        2. Recovery buttons (Try Again, Different Data Source, Cancel)
        3. Session stays in AWAITING_FILE_PATH state
        """
        # Mock path validation failure
        mock_path_validator.validate_path.return_value = {
            'is_valid': False,
            'error': 'Path not in allowed directories'
        }

        # Mock the validating message that gets created
        validating_msg = AsyncMock()
        mock_update_with_message.message.reply_text.side_effect = [
            validating_msg,  # First call: "Validating path..."
            AsyncMock()      # Second call: Error message with buttons
        ]

        # Handle file path input
        await prediction_handler.handle_file_path_input(
            mock_update_with_message,
            mock_context
        )

        # Verify error message was sent
        assert mock_update_with_message.message.reply_text.call_count == 2
        error_call = mock_update_with_message.message.reply_text.call_args_list[1]

        # Verify error message contains expected text
        error_message = error_call[0][0]
        assert "File Loading Error" in error_message
        assert "Path not in allowed directories" in error_message

        # Verify recovery buttons were included
        assert 'reply_markup' in error_call[1]
        reply_markup = error_call[1]['reply_markup']
        assert isinstance(reply_markup, InlineKeyboardMarkup)

        # Verify buttons exist
        buttons = reply_markup.inline_keyboard
        assert len(buttons) == 3  # Three rows of buttons

        # Verify button callback data
        assert buttons[0][0].callback_data == "pred_retry_path"
        assert buttons[1][0].callback_data == "pred_back_to_source"
        assert buttons[2][0].callback_data == "pred_cancel"

    @pytest.mark.asyncio
    async def test_retry_button_reshows_path_prompt(
        self,
        prediction_handler,
        mock_update_with_callback,
        mock_context,
        mock_state_manager
    ):
        """
        Test that clicking 'Try Again' re-shows the path input prompt.

        User should be able to try entering path again without losing state.
        """
        # Set callback data for retry
        mock_update_with_callback.callback_query.data = "pred_retry_path"

        # Handle retry callback
        await prediction_handler.handle_retry_path(
            mock_update_with_callback,
            mock_context
        )

        # Verify callback was answered
        mock_update_with_callback.callback_query.answer.assert_called_once()

        # Verify path prompt was shown again
        mock_update_with_callback.callback_query.edit_message_text.assert_called_once()
        prompt_call = mock_update_with_callback.callback_query.edit_message_text.call_args

        prompt_text = prompt_call[0][0]
        assert "Local File Path" in prompt_text
        assert "Allowed directories" in prompt_text
        assert "Type or paste your file path" in prompt_text

    @pytest.mark.asyncio
    async def test_back_button_returns_to_data_source(
        self,
        prediction_handler,
        mock_update_with_callback,
        mock_context,
        mock_state_manager
    ):
        """
        Test that clicking 'Different Data Source' returns to data source selection.

        User should be able to choose Telegram upload instead of local path.
        """
        # Set callback data for back
        mock_update_with_callback.callback_query.data = "pred_back_to_source"

        # Get the mock session
        mock_session = await mock_state_manager.get_session(12345, "chat_67890")

        # Handle back callback
        await prediction_handler.handle_back_to_source(
            mock_update_with_callback,
            mock_context
        )

        # Verify callback was answered
        mock_update_with_callback.callback_query.answer.assert_called_once()

        # Verify state transition to CHOOSING_DATA_SOURCE
        mock_state_manager.transition_state.assert_called_once_with(
            mock_session,
            MLPredictionState.CHOOSING_DATA_SOURCE.value
        )

        # Verify session data was cleared
        assert mock_session.uploaded_data is None
        assert mock_session.file_path is None

        # Verify data source prompt was shown
        mock_update_with_callback.callback_query.edit_message_text.assert_called_once()
        prompt_call = mock_update_with_callback.callback_query.edit_message_text.call_args

        prompt_text = prompt_call[0][0]
        assert "Load Prediction Data" in prompt_text
        assert "Upload File" in prompt_text
        assert "Use Local Path" in prompt_text

    @pytest.mark.asyncio
    async def test_cancel_button_resets_session(
        self,
        prediction_handler,
        mock_update_with_callback,
        mock_context,
        mock_state_manager
    ):
        """
        Test that clicking 'Cancel Workflow' resets the session.

        User should be able to exit workflow completely and start fresh.
        """
        # Set callback data for cancel
        mock_update_with_callback.callback_query.data = "pred_cancel"

        # Get the mock session
        mock_session = await mock_state_manager.get_session(12345, "chat_67890")

        # Handle cancel callback
        await prediction_handler.handle_cancel_workflow(
            mock_update_with_callback,
            mock_context
        )

        # Verify callback was answered
        mock_update_with_callback.callback_query.answer.assert_called_once()

        # Verify session was reset
        mock_state_manager.reset_session.assert_called_once_with(mock_session)

        # Verify cancellation message was shown
        mock_update_with_callback.callback_query.edit_message_text.assert_called_once()
        cancel_call = mock_update_with_callback.callback_query.edit_message_text.call_args

        cancel_text = cancel_call[0][0]
        assert "Workflow Canceled" in cancel_text
        assert "/predict" in cancel_text
        assert "/train" in cancel_text

    @pytest.mark.asyncio
    async def test_recovery_from_multiple_failed_attempts(
        self,
        prediction_handler,
        mock_update_with_message,
        mock_context,
        mock_path_validator
    ):
        """
        Test that user can retry multiple times after validation failures.

        This verifies the session doesn't get corrupted after multiple errors.
        """
        # Mock path validation failure
        mock_path_validator.validate_path.return_value = {
            'is_valid': False,
            'error': 'Path not in allowed directories'
        }

        # Mock validating messages
        validating_msg = AsyncMock()
        mock_update_with_message.message.reply_text.side_effect = [
            validating_msg, AsyncMock(),  # First attempt
            validating_msg, AsyncMock(),  # Second attempt
            validating_msg, AsyncMock()   # Third attempt
        ]

        # Attempt 1
        await prediction_handler.handle_file_path_input(
            mock_update_with_message,
            mock_context
        )

        # Attempt 2
        mock_update_with_message.message.text = "/tmp/another_invalid.csv"
        await prediction_handler.handle_file_path_input(
            mock_update_with_message,
            mock_context
        )

        # Attempt 3
        mock_update_with_message.message.text = "/Users/wrong/path.csv"
        await prediction_handler.handle_file_path_input(
            mock_update_with_message,
            mock_context
        )

        # Verify all three attempts showed error messages with buttons
        assert mock_update_with_message.message.reply_text.call_count == 6  # 3 validating + 3 errors

        # Verify all error messages had recovery buttons
        error_calls = [
            mock_update_with_message.message.reply_text.call_args_list[1],
            mock_update_with_message.message.reply_text.call_args_list[3],
            mock_update_with_message.message.reply_text.call_args_list[5]
        ]

        for error_call in error_calls:
            assert 'reply_markup' in error_call[1]
            reply_markup = error_call[1]['reply_markup']
            assert isinstance(reply_markup, InlineKeyboardMarkup)
            assert len(reply_markup.inline_keyboard) == 3  # Three recovery buttons


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
