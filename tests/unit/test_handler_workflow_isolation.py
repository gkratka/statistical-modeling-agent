"""
TDD Tests for Handler Workflow Isolation.

These tests verify that training and prediction text handlers only process
messages for their respective workflows, preventing handler collision errors.

Background:
Training and prediction handlers are both registered to process ALL text messages
(filters.TEXT) but in different groups (1 and 2). Without workflow type filtering,
both handlers would process every message, leading to validation errors and
unexpected behavior.

Fix: Each handler checks session.workflow_type before processing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.ml_handlers.prediction_handlers import PredictionHandler
from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager, MLPredictionState, MLTrainingState, WorkflowType
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
    mock_session.workflow_type = None
    mock_session.uploaded_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    mock_session.selections = {}

    manager.get_session.return_value = mock_session
    manager.get_or_create_session.return_value = mock_session
    manager.transition_state = AsyncMock()

    return manager


@pytest.fixture
def mock_data_loader():
    """Create mock data loader."""
    loader = MagicMock(spec=DataLoader)
    loader.allowed_directories = ["/Users/test/data"]
    loader.local_max_size_mb = 1000
    loader.local_extensions = [".csv"]
    return loader


@pytest.fixture
def mock_path_validator():
    """Create mock path validator."""
    return MagicMock(spec=PathValidator)


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
def training_handler(mock_state_manager, mock_data_loader, mock_path_validator):
    """Create training handler with mocked dependencies."""
    handler = LocalPathMLTrainingHandler(
        state_manager=mock_state_manager,
        data_loader=mock_data_loader,
        path_validator=mock_path_validator
    )
    # Mock ML engine to avoid initialization issues
    handler.ml_engine = MagicMock()
    # Mock template handlers
    handler.template_handlers = MagicMock()
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
    update.message.reply_text = AsyncMock()
    update.effective_message = update.message
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


class TestPredictionHandlerWorkflowIsolation:
    """Test prediction handler only processes prediction workflow messages."""

    @pytest.mark.asyncio
    async def test_prediction_handler_ignores_training_workflow(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Prediction handler should ignore messages when user is in training workflow.

        This prevents the handler from trying to validate training schema input
        as a file path, which caused "Path not in allowed directories" errors.
        """
        # Setup session in TRAINING workflow
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value

        # User types schema input (looks like file path to prediction handler)
        mock_update.message.text = "features: col1, col2, col3"

        # Prediction handler should return early (not process)
        await prediction_handler.handle_text_input(mock_update, mock_context)

        # Verify NO path validation attempted (handler returned early)
        prediction_handler.path_validator.validate_path.assert_not_called()
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_prediction_handler_processes_prediction_workflow(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Prediction handler should process messages when user is in prediction workflow.
        """
        # Setup session in PREDICTION workflow
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.AWAITING_FILE_PATH.value

        # User types file path
        mock_update.message.text = "/Users/test/data/test.csv"

        # Configure path validator to return valid result
        prediction_handler.path_validator.validate_path.return_value = {
            'is_valid': False,
            'error': 'File not found',
            'resolved_path': None
        }

        # Prediction handler should process the message
        await prediction_handler.handle_text_input(mock_update, mock_context)

        # Verify path validation WAS attempted
        prediction_handler.path_validator.validate_path.assert_called_once()

    @pytest.mark.asyncio
    async def test_prediction_handler_ignores_none_workflow(
        self,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Prediction handler should ignore messages when no workflow is active.
        """
        # Setup session with NO workflow
        session = await mock_state_manager.get_session(12345, "chat_67890")
        session.workflow_type = None
        session.current_state = None

        # User types random message
        mock_update.message.text = "hello"

        # Prediction handler should return early
        await prediction_handler.handle_text_input(mock_update, mock_context)

        # Verify no processing occurred
        prediction_handler.path_validator.validate_path.assert_not_called()
        mock_update.message.reply_text.assert_not_called()


class TestTrainingHandlerWorkflowIsolation:
    """Test training handler only processes training workflow messages."""

    @pytest.mark.asyncio
    async def test_training_handler_ignores_prediction_workflow(
        self,
        training_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Training handler should ignore messages when user is in prediction workflow.
        """
        # Setup session in PREDICTION workflow
        session = await mock_state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value

        # User types feature selection
        mock_update.message.text = "col1, col2, col3"

        # Training handler should return early (not process)
        await training_handler.handle_text_input(mock_update, mock_context)

        # Verify NO processing occurred
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_training_handler_processes_training_workflow(
        self,
        training_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Training handler should process messages when user is in training workflow.
        """
        # Setup session in TRAINING workflow
        session = await mock_state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.data_source = "local_path"

        # User types file path
        mock_update.message.text = "/Users/test/data/test.csv"

        # Configure data loader to return valid result
        training_handler.data_loader.validate_local_path = MagicMock(return_value=(
            False, "File not found", None
        ))

        # Training handler should process the message
        try:
            await training_handler.handle_text_input(mock_update, mock_context)
        except Exception:
            pass  # May raise ApplicationHandlerStop or validation errors

        # Verify processing occurred (validation message sent)
        assert mock_update.message.reply_text.call_count > 0

    @pytest.mark.asyncio
    async def test_training_handler_ignores_none_workflow(
        self,
        training_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Training handler should ignore messages when no workflow is active.
        """
        # Setup session with NO workflow
        session = await mock_state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = None
        session.current_state = None

        # User types random message
        mock_update.message.text = "hello"

        # Training handler should return early
        await training_handler.handle_text_input(mock_update, mock_context)

        # Verify no processing occurred (no messages sent)
        mock_update.message.reply_text.assert_not_called()


class TestCrossWorkflowNoCollision:
    """Test that both handlers can coexist without collision."""

    @pytest.mark.asyncio
    async def test_schema_input_no_path_validation_error(
        self,
        training_handler,
        prediction_handler,
        mock_update,
        mock_context,
        mock_state_manager
    ):
        """
        Reproduce original bug: User in training workflow types schema input,
        prediction handler should NOT try to validate it as a file path.

        This test reproduces the exact scenario from the user's report:
        - User is in TRAINING workflow at AWAITING_SCHEMA_INPUT state
        - User types: "features: Attribute1,Attribute2,...,Attribute20"
        - Prediction handler should ignore (not validate as path)
        - Training handler should process (accept schema)
        """
        # Setup: User in training workflow, awaiting schema input
        session = await mock_state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_SCHEMA_INPUT.value
        session.file_path = "/Users/test/data/german_credit_data.csv"

        # User types schema with "features:" prefix (exact user input from bug report)
        mock_update.message.text = "features: Attribute1,Attribute2,Attribute3,Attribute4,Attribute5"

        # PREDICTION handler processes first (should ignore due to workflow type)
        await prediction_handler.handle_text_input(mock_update, mock_context)

        # Verify prediction handler did NOT attempt path validation
        prediction_handler.path_validator.validate_path.assert_not_called()

        # TRAINING handler processes (should accept schema)
        # Note: We won't actually call it here as it would require full mock setup,
        # but the key assertion is that prediction handler didn't interfere


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
