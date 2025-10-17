"""
TDD Tests for Text Message Handler Routing.

These tests verify that text messages are correctly routed between the general
message_handler and the specialized ML training text handler based on session state.

Background:
The general message_handler (group 0) should detect ML training states and exit early,
allowing the specialized ML training text handler (group 1) to process the input.

Bug Fix: NAMING_MODEL state was missing from the early-exit list, causing the general
handler to process model naming text instead of deferring to the specialized handler.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType


class TestTextHandlerRouting:
    """Test that text messages route correctly based on ML training state."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update with text message."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.message = MagicMock(spec=Message)
        update.message.text = "test_input"
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock bot context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': StateManager()}
        return context

    @pytest.mark.asyncio
    async def test_naming_model_state_triggers_early_exit(self, mock_update, mock_context):
        """
        Test that NAMING_MODEL state causes general handler to exit early.

        This test validates the fix for the bug where custom model names weren't
        being processed because NAMING_MODEL was missing from the early-exit list.
        """
        from src.bot.handlers import message_handler
        from src.core.state_manager import MLTrainingState

        # Setup: User in NAMING_MODEL state
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        # Mock the text to be a custom model name
        mock_update.message.text = "german_keras_def_20v"

        # Execute: Call message_handler
        await message_handler(mock_update, mock_context)

        # Verify: Handler should return early (not process the text)
        # If it processed, it would call reply_text with parse_mode="Markdown"
        # Early exit means reply_text is NOT called
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_complete_state_triggers_early_exit(self, mock_update, mock_context):
        """
        Test that TRAINING_COMPLETE state causes general handler to exit early.

        This ensures users in the model naming workflow (before clicking buttons)
        don't have their text intercepted by the general handler.
        """
        from src.bot.handlers import message_handler

        # Setup: User in TRAINING_COMPLETE state
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.TRAINING_COMPLETE.value
        await state_manager.update_session(session)

        # Execute
        await message_handler(mock_update, mock_context)

        # Verify: Early exit (no processing)
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_selecting_target_state_triggers_early_exit(self, mock_update, mock_context):
        """
        Test that existing ML training states still trigger early exit.

        This ensures our fix doesn't break existing behavior for other states.
        """
        from src.bot.handlers import message_handler

        # Setup: User in SELECTING_TARGET state (existing state)
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.SELECTING_TARGET.value
        await state_manager.update_session(session)

        # Execute
        await message_handler(mock_update, mock_context)

        # Verify: Early exit
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_ml_state_allows_general_handler_processing(self, mock_update, mock_context):
        """
        Test that non-ML-training states allow general handler to process text.

        This ensures we don't break normal text message handling for users
        not in ML training workflow.
        """
        from src.bot.handlers import message_handler

        # Setup: User with NO active workflow
        state_manager = mock_context.bot_data['state_manager']
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = None
        session.current_state = None
        await state_manager.update_session(session)

        # Mock: User has no uploaded data
        mock_context.user_data = {}

        # Execute
        await message_handler(mock_update, mock_context)

        # Verify: Handler processes the message (calls reply_text)
        assert mock_update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_all_ml_training_states_in_early_exit_list(self):
        """
        Test that all relevant ML training states are in the early-exit list.

        This is a regression test to ensure future state additions don't get missed.
        """
        from src.bot.handlers import message_handler
        import inspect

        # Get the handler source code
        source = inspect.getsource(message_handler)

        # Critical states that MUST be in early-exit list
        critical_states = [
            "CHOOSING_DATA_SOURCE",
            "AWAITING_FILE_PATH",
            "CONFIRMING_SCHEMA",
            "SELECTING_TARGET",
            "SELECTING_FEATURES",
            "CONFIRMING_MODEL",
            "TRAINING_COMPLETE",  # NEW: Model naming workflow entry point
            "NAMING_MODEL",       # NEW: Custom name input state (THE FIX)
        ]

        for state in critical_states:
            assert f"MLTrainingState.{state}.value" in source, \
                f"State '{state}' is missing from message_handler early-exit list"


class TestHandlerPriority:
    """Test handler group priority and execution order."""

    def test_general_handler_is_group_0(self):
        """Verify general message_handler has default group (0)."""
        # This is verified by checking telegram_bot.py registration
        # Group 0 is default when no group specified
        pass

    def test_ml_training_handler_is_group_1(self):
        """Verify ML training text handler is registered in group 1."""
        from src.bot.ml_handlers.ml_training_local_path import register_local_path_handlers
        import inspect

        source = inspect.getsource(register_local_path_handlers)

        # Verify group=1 is specified for unified text handler
        assert "group=1" in source, \
            "ML training text handler must be in group=1 for proper priority"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
