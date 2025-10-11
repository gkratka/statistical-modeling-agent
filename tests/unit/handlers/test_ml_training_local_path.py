"""
Unit tests for ML training local path workflow handlers (Phase 5).

Tests the LocalPathMLTrainingHandler class including:
- Data source selection workflow
- File path input handling
- Schema confirmation flow
- Error handling
- State transitions
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery, Message, User, Chat, InlineKeyboardMarkup

from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.core.state_manager import StateManager, MLTrainingState, WorkflowType, UserSession
from src.processors.data_loader import DataLoader
from src.utils.exceptions import PathValidationError, DataError, ValidationError
from src.utils.schema_detector import DatasetSchema, ColumnSchema


@pytest.fixture
def data_loader_disabled(config_disabled):
    return DataLoader(config=config_disabled)


@pytest.fixture
def handler_enabled(state_manager, data_loader_enabled):
    return LocalPathMLTrainingHandler(state_manager, data_loader_enabled)


@pytest.fixture
def handler_disabled(state_manager, data_loader_disabled):
    return LocalPathMLTrainingHandler(state_manager, data_loader_disabled)


@pytest.fixture
def mock_update():
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = MagicMock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.message.text = "/train"
    return update


@pytest.fixture
def mock_callback_query():
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.callback_query = MagicMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = "data_source:telegram"
    return update


@pytest.mark.asyncio
class TestHandleStartTraining:

    async def test_shows_data_source_selection_when_enabled(
        self, handler_enabled, mock_update, mock_context
    ):
        await handler_enabled.handle_start_training(mock_update, mock_context)

        # Should create session
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session is not None
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Should send message with inline keyboard
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "ML Training Workflow" in call_args[0][0]
        assert "Upload File" in call_args[0][0]
        assert "Use Local Path" in call_args[0][0]
        assert "reply_markup" in call_args[1]

    async def test_goes_directly_to_upload_when_disabled(
        self, handler_disabled, mock_update, mock_context
    ):
        await handler_disabled.handle_start_training(mock_update, mock_context)

        # Should go directly to AWAITING_DATA
        session = await handler_disabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_DATA.value

        # Should send upload prompt
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "upload" in call_args[0][0].lower()


@pytest.mark.asyncio
class TestHandleDataSourceSelection:

    async def test_telegram_upload_selection(
        self, handler_enabled, mock_callback_query, mock_context
    ):
        # Setup: create session in CHOOSING_DATA_SOURCE state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await handler_enabled.state_manager.update_session(session)

        mock_callback_query.callback_query.data = "data_source:telegram"

        await handler_enabled.handle_data_source_selection(mock_callback_query, mock_context)

        # Should transition to AWAITING_DATA
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_DATA.value
        assert session.data_source == "telegram"

        # Should edit message
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args
        assert "Telegram Upload" in call_args[0][0]

    async def test_local_path_selection(
        self, handler_enabled, mock_callback_query, mock_context
    ):
        # Setup: create session in CHOOSING_DATA_SOURCE state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await handler_enabled.state_manager.update_session(session)

        mock_callback_query.callback_query.data = "data_source:local_path"

        await handler_enabled.handle_data_source_selection(mock_callback_query, mock_context)

        # Should transition to AWAITING_FILE_PATH
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert session.data_source == "local_path"

        # Should show allowed directories
        call_args = mock_callback_query.callback_query.edit_message_text.call_args
        assert "Local File Path" in call_args[0][0]
        assert "Allowed directories" in call_args[0][0]


@pytest.mark.asyncio
class TestHandleFilePathInput:

    async def test_valid_file_path_loads_data(
        self, handler_enabled, mock_update, mock_context, sample_csv
    ):
        # Setup: create session in AWAITING_FILE_PATH state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.data_source = "local_path"
        await handler_enabled.state_manager.update_session(session)

        mock_update.message.text = str(sample_csv)
        mock_update.message.reply_text = AsyncMock()

        # Mock loading message
        loading_msg = MagicMock()
        loading_msg.delete = AsyncMock()
        mock_update.message.reply_text.return_value = loading_msg

        await handler_enabled.handle_text_input(mock_update, mock_context)

        # Should transition to CHOOSING_LOAD_OPTION (not directly to schema confirmation)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CHOOSING_LOAD_OPTION.value
        assert session.file_path == str(sample_csv)

        # Should show load option selection
        assert mock_update.message.reply_text.call_count >= 1
        final_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Choose Loading Strategy" in final_call[0][0] or "Load Now" in final_call[0][0]

    async def test_invalid_path_shows_error(
        self, handler_enabled, mock_update, mock_context, tmp_path
    ):
        # Setup: create session in AWAITING_FILE_PATH state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.data_source = "local_path"
        await handler_enabled.state_manager.update_session(session)

        mock_update.message.text = str(tmp_path / "nonexistent.csv")
        mock_update.message.reply_text = AsyncMock()

        # Mock loading message
        loading_msg = MagicMock()
        loading_msg.delete = AsyncMock()
        mock_update.message.reply_text.return_value = loading_msg

        await handler_enabled.handle_text_input(mock_update, mock_context)

        # Should stay in AWAITING_FILE_PATH state
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

        # Should show error message (polished message format)
        assert mock_update.message.reply_text.call_count >= 2
        error_call = mock_update.message.reply_text.call_args_list[-1]
        # Check for either "File Not Found" or other error message
        assert any(text in error_call[0][0] for text in ["File Not Found", "Error", "not found"])

    async def test_ignores_input_in_wrong_state(
        self, handler_enabled, mock_update, mock_context
    ):
        # Setup: create session in different state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        await handler_enabled.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        # Stay in initial state, don't transition to AWAITING_FILE_PATH

        mock_update.message.text = "/some/path/to/file.csv"
        mock_update.message.reply_text = AsyncMock()

        await handler_enabled.handle_text_input(mock_update, mock_context)

        # Should not process input
        mock_update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
class TestHandleSchemaConfirmation:

    async def test_accept_schema_continues_workflow(
        self, handler_enabled, mock_callback_query, mock_context
    ):
        # Setup: create session in CONFIRMING_SCHEMA state with detected schema
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value
        session.data_source = "local_path"

        # IMPORTANT: uploaded_data is required for model selection
        session.uploaded_data = pd.DataFrame({
            'sqft': [1000, 1500, 2000],
            'bedrooms': [2, 3, 4],
            'price': [200000, 300000, 400000]
        })

        session.detected_schema = {
            'task_type': 'regression',
            'target': 'price',
            'features': ['sqft', 'bedrooms']
        }
        await handler_enabled.state_manager.update_session(session)

        # Mock the update object with message attribute for _show_model_selection
        mock_callback_query.message = MagicMock(spec=Message)
        mock_callback_query.message.reply_text = AsyncMock()

        mock_callback_query.callback_query.data = "schema:accept"

        await handler_enabled.handle_schema_confirmation(mock_callback_query, mock_context)

        # Should transition to CONFIRMING_MODEL (our fix skips target selection)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value

        # Should have transferred schema to selections
        assert session.selections['target_column'] == 'price'
        assert session.selections['feature_columns'] == ['sqft', 'bedrooms']
        assert session.selections['detected_task_type'] == 'regression'

        # Should show acceptance message
        mock_callback_query.callback_query.edit_message_text.assert_called_once()
        call_args = mock_callback_query.callback_query.edit_message_text.call_args
        assert "Schema Accepted" in call_args[0][0]

    async def test_reject_schema_goes_back_to_file_path(
        self, handler_enabled, mock_callback_query, mock_context
    ):
        # Setup: create session in CONFIRMING_SCHEMA state
        session = await handler_enabled.state_manager.get_or_create_session(12345, "67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value
        session.data_source = "local_path"

        session.uploaded_data = pd.DataFrame({'a': [1, 2, 3]})
        session.file_path = "/path/to/data.csv"
        session.detected_schema = {'task_type': 'regression'}
        await handler_enabled.state_manager.update_session(session)

        mock_callback_query.callback_query.data = "schema:reject"

        await handler_enabled.handle_schema_confirmation(mock_callback_query, mock_context)

        # Should transition back to AWAITING_FILE_PATH
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

        # Should clear stored data
        assert session.uploaded_data is None
        assert session.file_path is None
        assert session.detected_schema is None

        # Should show rejection message
        call_args = mock_callback_query.callback_query.edit_message_text.call_args
        assert "Schema Rejected" in call_args[0][0]


@pytest.mark.asyncio
class TestCompleteWorkflow:

    async def test_complete_local_path_workflow(
        self, handler_enabled, mock_update, mock_callback_query, mock_context, sample_csv
    ):
        # Step 1: Start training
        await handler_enabled.handle_start_training(mock_update, mock_context)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Step 2: Choose local path
        mock_callback_query.callback_query.data = "data_source:local_path"
        await handler_enabled.handle_data_source_selection(mock_callback_query, mock_context)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert session.data_source == "local_path"

        # Step 3: Provide file path
        mock_update.message.text = str(sample_csv)
        loading_msg = MagicMock()
        loading_msg.delete = AsyncMock()
        mock_update.message.reply_text = AsyncMock(return_value=loading_msg)

        await handler_enabled.handle_text_input(mock_update, mock_context)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CHOOSING_LOAD_OPTION.value
        assert session.file_path == str(sample_csv)

        # Step 4: Select "Load Now"
        mock_callback_query.callback_query.data = "load_option:immediate"
        await handler_enabled.handle_load_option_selection(mock_callback_query, mock_context)
        session = await handler_enabled.state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value
        assert session.uploaded_data is not None

        # Step 5: Accept schema
        # Mock the update object with message attribute for _show_model_selection
        mock_callback_query.message = MagicMock(spec=Message)
        mock_callback_query.message.reply_text = AsyncMock()

        mock_callback_query.callback_query.data = "schema:accept"
        await handler_enabled.handle_schema_confirmation(mock_callback_query, mock_context)
        session = await handler_enabled.state_manager.get_session(12345, "67890")

        # Should transition to CONFIRMING_MODEL (our fix skips target selection)
        assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value

        # Verify data persisted through workflow
        assert session.data_source == "local_path"
        assert session.file_path == str(sample_csv)
        assert session.uploaded_data is not None

        # Verify schema was transferred to selections
        assert 'target_column' in session.selections
        assert 'feature_columns' in session.selections


@pytest.fixture
def mock_context():
    return MagicMock()
