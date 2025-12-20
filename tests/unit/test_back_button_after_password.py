"""Test back button specifically after password authentication.

This test reproduces the exact bug reported:
- After "Access Granted" for directory → Back shows "at beginning"
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from telegram import Update, CallbackQuery, User, Chat, Message
from telegram.ext import ContextTypes
from pathlib import Path

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.processors.data_loader import DataLoader
from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler


@pytest.fixture
def state_manager():
    """Create StateManager instance for testing."""
    return StateManager()


@pytest.fixture
def data_loader():
    """Create DataLoader instance for testing."""
    loader = MagicMock(spec=DataLoader)
    loader.local_enabled = True
    loader.allowed_directories = ["/allowed/dir"]  # Restricted whitelist to force password flow
    loader.local_max_size_mb = 100
    loader.local_extensions = [".csv"]
    return loader


@pytest.fixture
def handler(state_manager, data_loader):
    """Create LocalPathMLTrainingHandler instance."""
    with patch('src.bot.ml_handlers.ml_training_local_path.PasswordValidator'):
        handler = LocalPathMLTrainingHandler(
            state_manager=state_manager,
            data_loader=data_loader
        )
        # Mock password validator to always succeed
        handler.password_validator.validate_password = MagicMock(return_value=(True, None))
        return handler


@pytest.fixture
def mock_update():
    """Create mock Update object."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = AsyncMock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.message.text = ""
    update.callback_query = AsyncMock(spec=CallbackQuery)
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.data = ""
    return update


@pytest.fixture
def mock_context():
    """Create mock Context object."""
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot_data = {}
    return context


class TestBackButtonAfterPasswordAuth:
    """Test back button after password authentication workflow."""

    @pytest.mark.asyncio
    async def test_back_button_after_access_granted(
        self,
        state_manager,
        handler,
        mock_update,
        mock_context
    ):
        """
        Reproduce the exact bug:
        1. Start /train
        2. Select "Local Path"
        3. Enter non-whitelisted path (triggers password flow)
        4. Enter correct password → "Access Granted" message shown
        5. Click Back button → Should work, but shows "at beginning"
        """
        # Step 1: Start workflow
        await handler.handle_start_training(mock_update, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value
        print(f"📍 After /train: state={session.current_state}, history_depth={session.state_history.get_depth()}")

        # Step 2: Select local path
        mock_update.callback_query.data = "data_source:local_path"
        await handler.handle_data_source_selection(mock_update, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value
        depth_after_selection = session.state_history.get_depth()
        print(f"📍 After local path selection: state={session.current_state}, history_depth={depth_after_selection}")
        assert depth_after_selection >= 1, "Should have history after first transition"

        # Step 3: Enter non-whitelisted path (triggers password flow)
        non_whitelisted_path = "/users/data/test.csv"  # Not in allowed_directories
        mock_update.message.text = non_whitelisted_path

        # Mock path validator to validate but require password
        with patch('src.utils.path_validator.PathValidator.validate_path') as mock_validate:
            mock_validate.return_value = (True, str(Path(non_whitelisted_path).resolve()), None)

            with patch('src.utils.path_validator.PathValidator.is_path_whitelisted') as mock_whitelist:
                mock_whitelist.return_value = False  # Force password requirement

                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.is_file', return_value=True):
                        with patch('src.utils.path_validator.get_file_size_mb', return_value=10.5):
                            await handler._process_file_path_input(
                                mock_update,
                                mock_context,
                                session,
                                non_whitelisted_path
                            )

        session = await state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.AWAITING_PASSWORD.value
        depth_after_path = session.state_history.get_depth()
        print(f"📍 After password prompt: state={session.current_state}, history_depth={depth_after_path}")
        assert depth_after_path > depth_after_selection, "History should grow after password prompt"

        # Step 4: Enter correct password
        mock_update.message.text = "correct_password"
        session.pending_auth_path = non_whitelisted_path  # Simulate pending auth

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_file', return_value=True):
                with patch('src.utils.path_validator.get_file_size_mb', return_value=10.5):
                    await handler.handle_password_input(mock_update, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CHOOSING_LOAD_OPTION.value
        depth_after_password = session.state_history.get_depth()
        print(f"📍 After password success (Access Granted): state={session.current_state}, history_depth={depth_after_password}")

        # BUG CHECK: At this point, user sees "Access Granted" message and back button
        # The history should contain previous states
        assert session.can_go_back(), \
            f"BUG REPRODUCED: cannot go back after 'Access Granted'! History depth={depth_after_password}"
        assert depth_after_password > 0, \
            f"History should not be empty after multiple transitions, got depth={depth_after_password}"

        # Step 5: Click back button
        success = session.restore_previous_state()

        # Should successfully go back to password state
        assert success, "restore_previous_state() should return True"
        assert session.current_state == MLTrainingState.AWAITING_PASSWORD.value, \
            f"Expected to restore to AWAITING_PASSWORD, got {session.current_state}"

        # Verify we can continue going back
        assert session.can_go_back(), "Should still be able to go back further"
        success2 = session.restore_previous_state()
        assert success2
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

    @pytest.mark.asyncio
    async def test_history_depth_accumulates_correctly(
        self,
        state_manager,
        handler,
        mock_update,
        mock_context
    ):
        """Test that history depth increases with each state transition."""
        # Start workflow
        await handler.handle_start_training(mock_update, mock_context)
        session = await state_manager.get_session(12345, "67890")
        depths = [session.state_history.get_depth()]

        # Transition 1: Select local path
        mock_update.callback_query.data = "data_source:local_path"
        await handler.handle_data_source_selection(mock_update, mock_context)
        session = await state_manager.get_session(12345, "67890")
        depths.append(session.state_history.get_depth())

        # Transition 2: Simulate direct state change with snapshot
        session.save_state_snapshot()
        await state_manager.transition_state(session, MLTrainingState.CHOOSING_LOAD_OPTION.value)
        session = await state_manager.get_session(12345, "67890")
        depths.append(session.state_history.get_depth())

        # Verify monotonic increase
        print(f"History depths across transitions: {depths}")
        for i in range(len(depths) - 1):
            assert depths[i+1] > depths[i], \
                f"History should grow: depth[{i}]={depths[i]} should be < depth[{i+1}]={depths[i+1]}"

    @pytest.mark.asyncio
    async def test_schema_accepted_back_button(
        self,
        state_manager,
        handler,
        mock_update,
        mock_context
    ):
        """
        Test the second reported bug:
        - After "Schema Accepted" → Back shows "at beginning"
        """
        # Start and select local path
        await handler.handle_start_training(mock_update, mock_context)
        mock_update.callback_query.data = "data_source:local_path"
        await handler.handle_data_source_selection(mock_update, mock_context)

        # Manually transition to schema confirmation state (simulating file load)
        session = await state_manager.get_session(12345, "67890")
        session.save_state_snapshot()
        await state_manager.transition_state(session, MLTrainingState.CONFIRMING_SCHEMA.value)

        # Mock schema detection
        session.detected_schema = {
            'target': 'price',
            'features': ['sqft', 'bedrooms'],
            'task_type': 'regression'
        }
        await state_manager.update_session(session)

        depth_before_accept = session.state_history.get_depth()
        print(f"📍 Before schema accept: history_depth={depth_before_accept}")

        # User accepts schema
        mock_update.callback_query.data = "schema:accept"
        await handler.handle_schema_confirmation(mock_update, mock_context)

        session = await state_manager.get_session(12345, "67890")
        assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value

        depth_after_accept = session.state_history.get_depth()
        print(f"📍 After schema accept: state={session.current_state}, history_depth={depth_after_accept}")

        # BUG CHECK: Can we go back after "Schema Accepted"?
        assert session.can_go_back(), \
            f"BUG REPRODUCED: cannot go back after 'Schema Accepted'! History depth={depth_after_accept}"

        # Click back
        success = session.restore_previous_state()
        assert success, "restore_previous_state() should return True"
        assert session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value, \
            f"Expected to restore to CONFIRMING_SCHEMA, got {session.current_state}"
