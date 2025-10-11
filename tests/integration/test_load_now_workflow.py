"""Integration tests for Load Now workflow."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery
import pandas as pd


@pytest.fixture
def test_data_file(tmp_path):
    """Create temporary CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "feature1,feature2,target\n"
        "1.0,2.0,100\n"
        "2.0,3.0,200\n"
        "3.0,4.0,300\n"
    )
    return str(csv_file)


@pytest.fixture
def mock_update():
    """Create mock Telegram update with callback query."""
    update = MagicMock(spec=Update)
    update.effective_user.id = 12345
    update.callback_query = MagicMock(spec=CallbackQuery)
    update.callback_query.data = "load_option:immediate"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message.reply_text = AsyncMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    return update


@pytest.mark.asyncio
class TestLoadNowWorkflow:
    """TDD tests for Load Now workflow."""

    async def test_load_now_with_valid_path(self, mock_update, test_data_file, tmp_path):
        """Test 1: Load Now with valid CSV path completes successfully."""
        # Import here to avoid circular dependencies
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
        from src.processors.data_loader import DataLoader

        # ARRANGE - Configure DataLoader with local path enabled
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 1000,
                'allowed_extensions': ['.csv']
            }
        }
        state_manager = StateManager()
        data_loader = DataLoader(config=config)
        handler = LocalPathMLTrainingHandler(state_manager, data_loader)

        # Set up session as if user just validated path - WITH WORKFLOW ACTIVE
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="12345"
        )
        session.workflow_type = WorkflowType.ML_TRAINING  # CRITICAL: Activate workflow
        session.file_path = test_data_file
        session.current_state = MLTrainingState.CHOOSING_LOAD_OPTION.value
        await state_manager.update_session(session)

        # ACT
        await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        mock_update.callback_query.answer.assert_called_once()

        # Check state transitioned to CONFIRMING_SCHEMA
        session = await state_manager.get_session(12345, "12345")
        assert session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value

        # Check data was loaded
        assert session.uploaded_data is not None
        assert "feature1" in session.uploaded_data.columns
        assert len(session.uploaded_data) == 3

        # Check schema detection ran
        assert session.detected_schema is not None
        assert session.detected_schema.get('target') is not None

    async def test_load_now_with_missing_file(self, mock_update, tmp_path):
        """Test 2: Load Now with non-existent file shows proper error."""
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
        from src.processors.data_loader import DataLoader

        # ARRANGE - Configure DataLoader
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 1000,
                'allowed_extensions': ['.csv']
            }
        }
        state_manager = StateManager()
        data_loader = DataLoader(config=config)
        handler = LocalPathMLTrainingHandler(state_manager, data_loader)

        # Set up session with non-existent file
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="12345"
        )
        session.workflow_type = WorkflowType.ML_TRAINING  # CRITICAL: Activate workflow
        session.file_path = str(tmp_path / "nonexistent.csv")
        session.current_state = MLTrainingState.CHOOSING_LOAD_OPTION.value
        await state_manager.update_session(session)

        # ACT
        await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        # Should show error message
        mock_update.callback_query.edit_message_text.assert_called()
        args = mock_update.callback_query.edit_message_text.call_args
        error_message = args[0][0]
        assert "File Not Found" in error_message
        assert "nonexistent.csv" in error_message

        # State should revert to AWAITING_FILE_PATH
        session = await state_manager.get_session(12345, "12345")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

    async def test_load_now_exception_handling(self, mock_update, test_data_file, tmp_path):
        """Test 3: Unexpected exceptions are logged with full traceback."""
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
        from src.processors.data_loader import DataLoader

        # ARRANGE
        config = {
            'local_data': {
                'enabled': True,
                'allowed_directories': [str(tmp_path)],
                'max_file_size_mb': 1000,
                'allowed_extensions': ['.csv']
            }
        }
        state_manager = StateManager()
        data_loader = DataLoader(config=config)
        handler = LocalPathMLTrainingHandler(state_manager, data_loader)

        # Set up session
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="12345"
        )
        session.workflow_type = WorkflowType.ML_TRAINING  # CRITICAL: Activate workflow
        session.file_path = test_data_file
        session.current_state = MLTrainingState.CHOOSING_LOAD_OPTION.value
        await state_manager.update_session(session)

        # Force an unexpected exception by patching data loader
        with patch.object(
            handler.data_loader,
            "load_from_local_path",
            side_effect=RuntimeError("Simulated crash")
        ):
            # ACT
            await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        mock_update.callback_query.edit_message_text.assert_called()
        args = mock_update.callback_query.edit_message_text.call_args
        error_message = args[0][0]

        # Should show actual exception type and message
        assert "RuntimeError" in error_message
        assert "Simulated crash" in error_message
