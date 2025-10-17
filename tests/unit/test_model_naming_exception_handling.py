"""
TDD Tests for Model Naming Exception Handling.

These tests verify that ApplicationHandlerStop exceptions are properly handled
in the model naming workflow, preventing spurious error messages when naming
succeeds.

Background:
After fixing the state transition bug, a new bug was discovered where the model
naming succeeded but showed an error message. This was because ApplicationHandlerStop
(raised for control flow) was caught by the generic except Exception handler.

Bug Fix: Add except ApplicationHandlerStop handler BEFORE except Exception to
re-raise it immediately, preventing it from being treated as an error.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User
from telegram.ext import ContextTypes, ApplicationHandlerStop

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.utils.exceptions import ValidationError


class TestModelNamingExceptionHandling:
    """Test exception handling in model naming workflow."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update with message."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.message = MagicMock(spec=Message)
        update.message.text = "test_model_name"
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock bot context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': StateManager()}
        return context

    @pytest.mark.asyncio
    async def test_successful_naming_sends_only_success_message(self, mock_update, mock_context):
        """
        Test that successful model naming sends only success message, not error.

        This validates the fix - before the fix, ApplicationHandlerStop was caught
        by the generic Exception handler, causing both success and error messages.
        """
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        # Setup handler
        ml_engine = MLEngine(MLEngineConfig.get_default())
        state_manager = mock_context.bot_data['state_manager']
        handler = LocalPathMLTrainingHandler(ml_engine, state_manager)

        # Setup: User in NAMING_MODEL state with pending model
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        # Mock ML engine methods
        ml_engine.set_model_name = MagicMock()
        ml_engine.get_model_info = MagicMock(return_value={
            'model_type': 'keras_binary_classification',
            'task_type': 'neural_network'
        })

        # Execute: Handle model name input
        with pytest.raises(ApplicationHandlerStop):
            await handler.handle_model_name_input(mock_update, mock_context)

        # Verify: Only success message sent, no error message
        assert mock_update.message.reply_text.call_count == 1
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Model Named Successfully" in call_args
        assert "Failed to set model name" not in call_args
        assert "Error" not in call_args or "Error" not in call_args.split('\n')[0]

    @pytest.mark.asyncio
    async def test_application_handler_stop_propagates_correctly(self, mock_update, mock_context):
        """
        Test that ApplicationHandlerStop is re-raised correctly.

        The exception should propagate to stop handler chain, not be caught
        as a generic error.
        """
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        # Setup handler
        ml_engine = MLEngine(MLEngineConfig.get_default())
        state_manager = mock_context.bot_data['state_manager']
        handler = LocalPathMLTrainingHandler(ml_engine, state_manager)

        # Setup: User in NAMING_MODEL state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        # Mock ML engine
        ml_engine.set_model_name = MagicMock()
        ml_engine.get_model_info = MagicMock(return_value={'model_type': 'test'})

        # Execute and verify: ApplicationHandlerStop is raised
        with pytest.raises(ApplicationHandlerStop):
            await handler.handle_model_name_input(mock_update, mock_context)

    @pytest.mark.asyncio
    async def test_validation_error_shows_validation_message(self, mock_update, mock_context):
        """
        Test that validation errors show appropriate validation message.

        ValidationError should be caught separately and show validation-specific
        error message, not generic error message.
        """
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        # Setup handler
        ml_engine = MLEngine(MLEngineConfig.get_default())
        state_manager = mock_context.bot_data['state_manager']
        handler = LocalPathMLTrainingHandler(ml_engine, state_manager)

        # Setup: User in NAMING_MODEL state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        # Mock ML engine to raise ValidationError
        ml_engine.set_model_name = MagicMock(
            side_effect=ValidationError("Name must be 3-100 characters")
        )

        # Execute: Handle model name input
        await handler.handle_model_name_input(mock_update, mock_context)

        # Verify: Validation error message shown
        assert mock_update.message.reply_text.call_count == 1
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Invalid Name" in call_args
        assert "3-100 characters" in call_args
        assert "Failed to set model name" not in call_args  # Not generic error

    @pytest.mark.asyncio
    async def test_generic_exception_shows_error_message(self, mock_update, mock_context):
        """
        Test that generic exceptions show error message.

        Actual exceptions (not ApplicationHandlerStop) should be caught and
        show user-friendly error message.
        """
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        # Setup handler
        ml_engine = MLEngine(MLEngineConfig.get_default())
        state_manager = mock_context.bot_data['state_manager']
        handler = LocalPathMLTrainingHandler(ml_engine, state_manager)

        # Setup: User in NAMING_MODEL state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        # Mock ML engine to raise generic exception
        ml_engine.set_model_name = MagicMock(
            side_effect=RuntimeError("Database connection failed")
        )

        # Execute: Handle model name input
        await handler.handle_model_name_input(mock_update, mock_context)

        # Verify: Error message shown
        assert mock_update.message.reply_text.call_count == 1
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Error" in call_args
        assert "Failed to set model name" in call_args


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
