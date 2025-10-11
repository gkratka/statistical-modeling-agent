"""
Unit tests for Keras neural network configuration handlers.

Tests defensive error handling, session management, and configuration flow.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, CallbackQuery, User, Chat

from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager, MLTrainingState, WorkflowType, UserSession
from src.processors.data_loader import DataLoader


@pytest.fixture
def state_manager():
    return StateManager()


@pytest.fixture
def data_loader():
    loader = DataLoader()
    loader.local_enabled = True
    return loader


@pytest.fixture
def handler(state_manager, data_loader):
    return LocalPathMLTrainingHandler(state_manager, data_loader)


@pytest.fixture
def mock_callback_query():
    """Create mock callback query for button presses."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.callback_query = MagicMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = "keras_epochs:100"
    return update


@pytest.fixture
def mock_context():
    return MagicMock()


@pytest.mark.asyncio
class TestKerasEpochsHandler:
    """Test handle_keras_epochs with various error conditions."""

    async def test_valid_session_with_keras_config(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test normal flow with valid session and initialized keras_config."""
        # Create session with keras_config
        session = await state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        session.selections['keras_config'] = {}
        await state_manager.update_session(session)

        # Call handler
        await handler.handle_keras_epochs(mock_callback_query, mock_context)

        # Verify epoch was stored
        session = await state_manager.get_session(12345, "67890")
        assert session is not None
        assert 'keras_config' in session.selections
        assert session.selections['keras_config']['epochs'] == 100

        # Verify message was edited
        mock_callback_query.callback_query.edit_message_text.assert_called_once()

    async def test_none_session_handles_gracefully(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test handler when session doesn't exist (returns None)."""
        # Don't create a session - get_session will return None

        # Call handler - should not crash
        await handler.handle_keras_epochs(mock_callback_query, mock_context)

        # Verify error message sent
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args[0][0]
        assert "expired" in call_args.lower() or "error" in call_args.lower()

    async def test_session_without_keras_config(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test handler when session exists but keras_config not initialized."""
        # Create session WITHOUT keras_config
        session = await state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        # Deliberately don't set keras_config
        await state_manager.update_session(session)

        # Call handler - should handle gracefully
        await handler.handle_keras_epochs(mock_callback_query, mock_context)

        # Should still work - initializes keras_config if missing
        session = await state_manager.get_session(12345, "67890")
        assert 'keras_config' in session.selections
        assert session.selections['keras_config']['epochs'] == 100


@pytest.mark.asyncio
class TestKerasInitializerHandler:
    """Test handle_keras_initializer with missing previous configuration."""

    async def test_valid_session_with_complete_config(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test normal flow with all previous config present."""
        # Create session with epochs and batch_size
        session = await state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.selections['keras_config'] = {
            'epochs': 100,
            'batch_size': 32
        }
        await state_manager.update_session(session)

        mock_callback_query.callback_query.data = "keras_init:glorot_uniform"

        # Call handler
        await handler.handle_keras_initializer(mock_callback_query, mock_context)

        # Verify initializer was stored
        session = await state_manager.get_session(12345, "67890")
        assert session.selections['keras_config']['kernel_initializer'] == "glorot_uniform"

        # Verify message shows previous config
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args[0][0]
        assert "100" in call_args  # epochs
        assert "32" in call_args   # batch_size

    async def test_missing_epochs_key(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test handler when epochs key is missing from keras_config."""
        # Create session with batch_size but NO epochs
        session = await state_manager.get_or_create_session(12345, "67890")
        session.selections['keras_config'] = {
            'batch_size': 32
            # epochs is missing!
        }
        await state_manager.update_session(session)

        mock_callback_query.callback_query.data = "keras_init:glorot_uniform"

        # Call handler - should use default for missing epochs
        await handler.handle_keras_initializer(mock_callback_query, mock_context)

        # Should not crash - uses default value
        mock_callback_query.callback_query.edit_message_text.assert_called_once()

    async def test_missing_batch_size_key(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test handler when batch_size key is missing from keras_config."""
        # Create session with epochs but NO batch_size
        session = await state_manager.get_or_create_session(12345, "67890")
        session.selections['keras_config'] = {
            'epochs': 100
            # batch_size is missing!
        }
        await state_manager.update_session(session)

        mock_callback_query.callback_query.data = "keras_init:glorot_uniform"

        # Call handler - should use default for missing batch_size
        await handler.handle_keras_initializer(mock_callback_query, mock_context)

        # Should not crash - uses default value
        mock_callback_query.callback_query.edit_message_text.assert_called_once()

    async def test_none_session_handles_gracefully(
        self, handler, mock_callback_query, mock_context
    ):
        """Test handler when session is None (expired)."""
        # Don't create session - get_session returns None
        mock_callback_query.callback_query.data = "keras_init:glorot_uniform"

        # Call handler - should not crash
        await handler.handle_keras_initializer(mock_callback_query, mock_context)

        # Should show error message
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args[0][0]
        assert "expired" in call_args.lower() or "error" in call_args.lower()


@pytest.mark.asyncio
class TestKerasValidationHandler:
    """Test handle_keras_validation (final step)."""

    async def test_complete_config_flow(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test final handler with all configuration present."""
        # Create session with complete keras_config
        session = await state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.selections['keras_config'] = {
            'epochs': 100,
            'batch_size': 32,
            'kernel_initializer': 'glorot_uniform',
            'verbose': 1
        }
        session.selections['target_column'] = 'price'
        session.selections['feature_columns'] = ['sqft', 'bedrooms']
        session.selections['model_type'] = 'keras_binary_classification'  # Required for architecture generation
        session.file_path = '/tmp/test.csv'  # Required for training
        await state_manager.update_session(session)

        # Mock ML Engine training to avoid actual training call
        handler.ml_engine.train_model = MagicMock(return_value={
            'success': True,
            'model_id': 'test_model_123',
            'metrics': {'loss': 0.5, 'accuracy': 0.85}
        })

        # Mock the reply_text for success message
        mock_callback_query.effective_message = MagicMock()
        mock_callback_query.effective_message.reply_text = AsyncMock()

        mock_callback_query.callback_query.data = "keras_val:0.2"

        # Call handler
        await handler.handle_keras_validation(mock_callback_query, mock_context)

        # Verify validation_split was stored
        session = await state_manager.get_session(12345, "67890")
        assert session.selections['keras_config']['validation_split'] == 0.2

        # Verify completion message
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args[0][0]
        assert "Configuration Complete" in call_args or "complete" in call_args.lower()

    async def test_none_session_handles_gracefully(
        self, handler, mock_callback_query, mock_context
    ):
        """Test final handler when session is None."""
        mock_callback_query.callback_query.data = "keras_val:0.2"

        # Call handler with no session
        await handler.handle_keras_validation(mock_callback_query, mock_context)

        # Should show error
        mock_callback_query.callback_query.edit_message_text.assert_called_once()


@pytest.mark.asyncio
class TestKerasBatchHandler:
    """Test handle_keras_batch error handling."""

    async def test_valid_session(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test normal batch size selection."""
        session = await state_manager.get_or_create_session(12345, "67890")
        session.selections['keras_config'] = {'epochs': 100}
        await state_manager.update_session(session)

        mock_callback_query.callback_query.data = "keras_batch:32"

        await handler.handle_keras_batch(mock_callback_query, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.selections['keras_config']['batch_size'] == 32

    async def test_none_session(
        self, handler, mock_callback_query, mock_context
    ):
        """Test batch handler with None session."""
        mock_callback_query.callback_query.data = "keras_batch:32"

        await handler.handle_keras_batch(mock_callback_query, mock_context)

        # Should handle gracefully
        mock_callback_query.callback_query.edit_message_text.assert_called_once()


@pytest.mark.asyncio
class TestKerasVerboseHandler:
    """Test handle_keras_verbose error handling."""

    async def test_valid_session(
        self, handler, state_manager, mock_callback_query, mock_context
    ):
        """Test normal verbose selection."""
        session = await state_manager.get_or_create_session(12345, "67890")
        session.selections['keras_config'] = {
            'epochs': 100,
            'batch_size': 32,
            'kernel_initializer': 'glorot_uniform'
        }
        await state_manager.update_session(session)

        mock_callback_query.callback_query.data = "keras_verbose:1"

        await handler.handle_keras_verbose(mock_callback_query, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.selections['keras_config']['verbose'] == 1

    async def test_none_session(
        self, handler, mock_callback_query, mock_context
    ):
        """Test verbose handler with None session."""
        mock_callback_query.callback_query.data = "keras_verbose:1"

        await handler.handle_keras_verbose(mock_callback_query, mock_context)

        # Should handle gracefully
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
