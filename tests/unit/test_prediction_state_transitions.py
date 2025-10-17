"""
TDD Tests for Prediction Workflow State Transitions.

These tests verify that state transitions are properly validated and
errors are handled gracefully when transitions fail.

Background:
The bug occurs when transition_state() is called but the return value
is not checked. If a transition fails, the handler continues execution
as if it succeeded, leading to incorrect routing of user input.

Fix: Check transition_state() return values and handle failures appropriately.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.ml_handlers.prediction_handlers import PredictionHandler
from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
import pandas as pd


@pytest.fixture
def mock_state_manager():
    """Create mock state manager."""
    manager = AsyncMock(spec=StateManager)

    # Mock session object
    mock_session = MagicMock()
    mock_session.user_id = 12345
    mock_session.current_state = None
    mock_session.workflow_type = WorkflowType.ML_PREDICTION
    mock_session.uploaded_data = None
    mock_session.file_path = None
    mock_session.selections = {}
    mock_session.save_state_snapshot = MagicMock()

    manager.get_session.return_value = mock_session
    manager.get_or_create_session.return_value = mock_session

    # Default: transitions succeed
    manager.transition_state = AsyncMock(return_value=(True, None, []))

    return manager


@pytest.fixture
def mock_data_loader():
    """Create mock data loader."""
    loader = MagicMock(spec=DataLoader)
    loader.allowed_directories = ["/Users/test/data"]
    loader.local_max_size_mb = 1000
    loader.local_extensions = [".csv"]

    # Mock successful data loading
    test_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    loader.load_from_local_path = AsyncMock(return_value=test_df)

    return loader


@pytest.fixture
def mock_path_validator():
    """Create mock path validator."""
    validator = MagicMock(spec=PathValidator)

    # Default: paths are valid
    validator.validate_path.return_value = {
        'is_valid': True,
        'resolved_path': '/Users/test/data/test.csv',
        'error': None
    }

    return validator


@pytest.fixture
def prediction_handler(mock_state_manager, mock_data_loader, mock_path_validator):
    """Create prediction handler with mocked dependencies."""
    handler = PredictionHandler(
        state_manager=mock_state_manager,
        data_loader=mock_data_loader,
        path_validator=mock_path_validator
    )
    # Mock ML engine to avoid initialization issues
    handler.ml_engine = MagicMock()
    return handler


@pytest.fixture
def mock_update():
    """Create mock update with message."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = AsyncMock(spec=Message)
    update.message.text = ""
    update.message.reply_text = AsyncMock()
    update.effective_message = update.message
    update.callback_query = None
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


class TestFilePathTransitionValidation:
    """Test that handle_file_path_input validates state transitions."""

    @pytest.mark.asyncio
    async def test_successful_path_validation_and_transition(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When file path is valid and transition succeeds,
        schema confirmation UI should be shown.
        """
        # Setup: User in AWAITING_FILE_PATH state
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.AWAITING_FILE_PATH.value

        # User provides valid path
        mock_update.message.text = "/Users/test/data/test.csv"

        # Mock successful transition
        mock_state_manager.transition_state.return_value = (True, None, [])

        # Execute handler (expect ApplicationHandlerStop)
        try:
            await prediction_handler.handle_file_path_input(mock_update, mock_context)
        except Exception as e:
            # ApplicationHandlerStop is expected behavior
            if "ApplicationHandlerStop" not in str(type(e)):
                raise

        # Verify path was validated
        prediction_handler.path_validator.validate_path.assert_called_once_with(
            "/Users/test/data/test.csv"
        )

        # Verify data was loaded
        prediction_handler.data_loader.load_from_local_path.assert_called_once()

        # Verify transition was attempted
        mock_state_manager.transition_state.assert_called_once_with(
            session,
            MLPredictionState.CONFIRMING_SCHEMA.value
        )

        # Verify user received success message (schema UI shown)
        assert mock_update.message.reply_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_transition_failure_shows_error_message(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When state transition fails, user should see error message
        and schema UI should NOT be shown.

        This is the bug fix: Previously, transition failures were ignored
        and schema UI was shown anyway.
        """
        # Setup: User in AWAITING_FILE_PATH state
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.AWAITING_FILE_PATH.value

        # User provides valid path
        mock_update.message.text = "/Users/test/data/test.csv"

        # Mock FAILED transition (prerequisites not met)
        mock_state_manager.transition_state.return_value = (
            False,
            "Prerequisites not met for state 'confirming_schema'",
            ["uploaded_data"]
        )

        # Execute handler
        await prediction_handler.handle_file_path_input(mock_update, mock_context)

        # Verify error message shown to user
        error_call_found = False
        for call in mock_update.message.reply_text.call_args_list:
            args, kwargs = call
            message = args[0] if args else ""
            if "State Transition Failed" in message or "Transition Failed" in message:
                error_call_found = True
                # Verify error details included
                assert "Prerequisites not met" in message
                break

        assert error_call_found, "Expected error message about transition failure"

        # Verify state remains AWAITING_FILE_PATH (not changed)
        assert session.current_state == MLPredictionState.AWAITING_FILE_PATH.value


class TestSchemaConfirmationStateValidation:
    """Test that handle_schema_confirmation validates current state."""

    @pytest.mark.asyncio
    async def test_schema_confirmation_in_correct_state(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When user clicks "Continue" in CONFIRMING_SCHEMA state,
        transition to AWAITING_FEATURE_SELECTION should succeed.
        """
        # Setup: User in CONFIRMING_SCHEMA state with data loaded
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.CONFIRMING_SCHEMA.value
        session.uploaded_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Mock callback query (user clicked "Continue")
        mock_update.callback_query = AsyncMock()
        mock_update.callback_query.data = "pred_schema_accept"
        mock_update.callback_query.answer = AsyncMock()
        mock_update.callback_query.edit_message_text = AsyncMock()

        # Mock successful transition
        mock_state_manager.transition_state.return_value = (True, None, [])

        # Execute handler
        await prediction_handler.handle_schema_confirmation(mock_update, mock_context)

        # Verify transition was attempted
        mock_state_manager.transition_state.assert_called_once_with(
            session,
            MLPredictionState.AWAITING_FEATURE_SELECTION.value
        )

        # Verify success message shown
        assert mock_update.callback_query.edit_message_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_schema_confirmation_in_wrong_state(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When user clicks "Continue" but state is NOT CONFIRMING_SCHEMA,
        error message should be shown and no transition attempted.

        This prevents the bug where wrong state leads to incorrect routing.
        """
        # Setup: User in WRONG state (AWAITING_FILE_PATH instead of CONFIRMING_SCHEMA)
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.AWAITING_FILE_PATH.value  # WRONG!
        session.uploaded_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Mock callback query (user clicked "Continue")
        mock_update.callback_query = AsyncMock()
        mock_update.callback_query.data = "pred_schema_accept"
        mock_update.callback_query.answer = AsyncMock()
        mock_update.callback_query.edit_message_text = AsyncMock()

        # Execute handler
        await prediction_handler.handle_schema_confirmation(mock_update, mock_context)

        # Verify error message shown
        error_call = mock_update.callback_query.edit_message_text.call_args_list[0]
        error_message = error_call[0][0]
        assert "Invalid State" in error_message or "Wrong state" in error_message
        # State value is lowercase in enum
        assert "awaiting_file_path" in error_message.lower()  # Show current state

        # Verify NO transition attempted (guard blocked it)
        mock_state_manager.transition_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_schema_confirmation_transition_failure(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When transition from CONFIRMING_SCHEMA to AWAITING_FEATURE_SELECTION fails,
        error message should be shown to user.
        """
        # Setup: User in CONFIRMING_SCHEMA state
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.CONFIRMING_SCHEMA.value
        session.uploaded_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Mock callback query
        mock_update.callback_query = AsyncMock()
        mock_update.callback_query.data = "pred_schema_accept"
        mock_update.callback_query.answer = AsyncMock()
        mock_update.callback_query.edit_message_text = AsyncMock()

        # Mock FAILED transition
        mock_state_manager.transition_state.return_value = (
            False,
            "Invalid transition",
            []
        )

        # Execute handler
        await prediction_handler.handle_schema_confirmation(mock_update, mock_context)

        # Verify error message shown
        error_call = mock_update.callback_query.edit_message_text.call_args_list[0]
        error_message = error_call[0][0]
        assert "Transition Failed" in error_message or "Failed" in error_message


class TestFeatureInputRouting:
    """Test that feature input is routed correctly after successful transitions."""

    @pytest.mark.asyncio
    async def test_features_input_routed_correctly(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        When user types features in AWAITING_FEATURE_SELECTION state,
        input should be routed to handle_feature_selection_input().
        """
        # Setup: User in AWAITING_FEATURE_SELECTION state
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
        session.uploaded_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })

        # User types features (with "features:" prefix like in bug report)
        mock_update.message.text = "features: col1, col2, col3"

        # Execute unified text handler (router) - expect ApplicationHandlerStop
        try:
            await prediction_handler.handle_text_input(mock_update, mock_context)
        except Exception as e:
            # ApplicationHandlerStop is expected behavior
            if "ApplicationHandlerStop" not in str(type(e)):
                raise

        # Verify route to feature selection, not file path validation
        # (Feature selection handler shows format auto-correction message)
        success_call_found = False
        for call in mock_update.message.reply_text.call_args_list:
            args, kwargs = call
            message = args[0] if args else ""
            if "Format Auto-Corrected" in message or "features" in message.lower():
                success_call_found = True
                break

        assert success_call_found, "Expected feature selection processing, not path validation"

    @pytest.mark.asyncio
    async def test_features_input_not_validated_as_path(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Regression test: Verify features input is NOT validated as file path.

        This is the original bug - user typed "features: Attribute1,..." and
        got "Path not in allowed directories" error because state was stuck
        at AWAITING_FILE_PATH.
        """
        # Setup: User in AWAITING_FEATURE_SELECTION state (CORRECT state)
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
        session.uploaded_data = pd.DataFrame({
            'Attribute1': [1, 2],
            'Attribute2': [3, 4],
            'Attribute3': [5, 6]
        })

        # User types features (exact format from bug report)
        mock_update.message.text = "features: Attribute1,Attribute2,Attribute3"

        # Execute unified text handler - expect ApplicationHandlerStop
        try:
            await prediction_handler.handle_text_input(mock_update, mock_context)
        except Exception as e:
            # ApplicationHandlerStop is expected behavior
            if "ApplicationHandlerStop" not in str(type(e)):
                raise

        # Verify path validator was NOT called
        prediction_handler.path_validator.validate_path.assert_not_called()

        # Verify NO "Path not in allowed directories" error
        for call in mock_update.message.reply_text.call_args_list:
            args, kwargs = call
            message = args[0] if args else ""
            assert "Path not in allowed directories" not in message, \
                "BUG: Features input was incorrectly validated as file path!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
