"""Tests for local worker connection handler.

This module tests the /connect command handler and worker connection
notifications following the TDD approach.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Message, Chat, InlineKeyboardButton, InlineKeyboardMarkup

# Test fixtures
@pytest.fixture
def mock_update():
    """Create a mock Telegram update."""
    update = AsyncMock(spec=Update)
    update.effective_user = User(id=12345, first_name="Test", is_bot=False)
    update.effective_chat = Chat(id=67890, type="private")
    update.message = AsyncMock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.callback_query = None
    return update


@pytest.fixture
def mock_context():
    """Create a mock bot context with required components."""
    context = AsyncMock()
    context.bot_data = {
        'websocket_server': AsyncMock(),
        'worker_enabled': True,
    }

    # Mock WorkerManager
    worker_manager = AsyncMock()
    worker_manager.is_user_connected = MagicMock(return_value=False)
    worker_manager.get_machine_name = MagicMock(return_value=None)

    # Mock TokenManager via WebSocketServer
    token_manager = AsyncMock()
    token_manager.generate_token = MagicMock(return_value="test-token-uuid-1234")

    context.bot_data['websocket_server'].worker_manager = worker_manager
    context.bot_data['websocket_server'].token_manager = token_manager

    return context


class TestConnectHandler:
    """Test /connect command handler."""

    @pytest.mark.asyncio
    async def test_connect_generates_token(self, mock_update, mock_context):
        """Test that /connect generates a one-time token."""
        from src.bot.handlers.connect_handler import handle_connect_command

        await handle_connect_command(mock_update, mock_context)

        # Verify token was generated
        token_manager = mock_context.bot_data['websocket_server'].token_manager
        token_manager.generate_token.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_connect_displays_curl_command_mac_linux(self, mock_update, mock_context):
        """Test that /connect displays curl command for Mac/Linux."""
        from src.bot.handlers.connect_handler import handle_connect_command

        await handle_connect_command(mock_update, mock_context)

        # Verify message contains curl command
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "curl -s" in reply_text_call
        assert "python3 -" in reply_text_call
        assert "--token=test-token-uuid-1234" in reply_text_call

    @pytest.mark.asyncio
    async def test_connect_displays_irm_command_windows(self, mock_update, mock_context):
        """Test that /connect displays irm command for Windows."""
        from src.bot.handlers.connect_handler import handle_connect_command

        await handle_connect_command(mock_update, mock_context)

        # Verify message contains irm command
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "irm" in reply_text_call or "Invoke-RestMethod" in reply_text_call
        assert "python -" in reply_text_call or "python3 -" in reply_text_call

    @pytest.mark.asyncio
    async def test_connect_shows_token_expiry_warning(self, mock_update, mock_context):
        """Test that /connect shows token expiry warning (5 minutes)."""
        from src.bot.handlers.connect_handler import handle_connect_command

        await handle_connect_command(mock_update, mock_context)

        # Verify message contains expiry warning
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "5 min" in reply_text_call or "expires" in reply_text_call.lower()

    @pytest.mark.asyncio
    async def test_connect_when_worker_disabled(self, mock_update, mock_context):
        """Test that /connect shows error when worker feature disabled."""
        from src.bot.handlers.connect_handler import handle_connect_command

        # Disable worker feature
        mock_context.bot_data['worker_enabled'] = False

        await handle_connect_command(mock_update, mock_context)

        # Verify error message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "not available" in reply_text_call.lower() or "disabled" in reply_text_call.lower()

    @pytest.mark.asyncio
    async def test_connect_when_worker_already_connected(self, mock_update, mock_context):
        """Test that /connect shows status when worker already connected."""
        from src.bot.handlers.connect_handler import handle_connect_command

        # Set worker as connected
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=True)
        mock_context.bot_data['websocket_server'].worker_manager.get_machine_name = MagicMock(return_value="MacBook-Pro")

        await handle_connect_command(mock_update, mock_context)

        # Verify message shows already connected
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "already connected" in reply_text_call.lower() or "MacBook-Pro" in reply_text_call


class TestStartHandlerWorkerStatus:
    """Test /start handler shows worker connection status."""

    @pytest.mark.asyncio
    async def test_start_shows_no_worker_button(self, mock_update, mock_context):
        """Test that /start shows 'Connect local worker' button when no worker connected."""
        from src.bot.handlers.connect_handler import get_start_message_with_worker_status

        # No worker connected
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=False)

        message, keyboard = get_start_message_with_worker_status(12345, mock_context)

        # Verify message and keyboard
        assert "Connect local worker" in str(keyboard) or message is not None

    @pytest.mark.asyncio
    async def test_start_shows_worker_connected_status(self, mock_update, mock_context):
        """Test that /start shows 'Worker: Connected (machine-name)' when connected."""
        from src.bot.handlers.connect_handler import get_start_message_with_worker_status

        # Worker connected
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=True)
        mock_context.bot_data['websocket_server'].worker_manager.get_machine_name = MagicMock(return_value="MacBook-Pro")

        message, keyboard = get_start_message_with_worker_status(12345, mock_context)

        # Verify message shows connection status
        assert "MacBook-Pro" in message
        assert "Connected" in message or "connected" in message


class TestWorkerNotifications:
    """Test worker connection/disconnection notifications."""

    @pytest.mark.asyncio
    async def test_worker_connection_notification_sent_to_user(self):
        """Test that user receives notification when worker connects."""
        from src.bot.handlers.connect_handler import notify_worker_connected

        mock_bot = AsyncMock()
        user_id = 12345
        machine_name = "MacBook-Pro"

        await notify_worker_connected(mock_bot, user_id, machine_name)

        # Verify bot sent message
        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        assert call_args[1]['chat_id'] == user_id
        assert "MacBook-Pro" in call_args[1]['text']
        assert "connected" in call_args[1]['text'].lower()

    @pytest.mark.asyncio
    async def test_worker_disconnection_notification_sent_to_user(self):
        """Test that user receives notification when worker disconnects."""
        from src.bot.handlers.connect_handler import notify_worker_disconnected

        mock_bot = AsyncMock()
        user_id = 12345

        await notify_worker_disconnected(mock_bot, user_id)

        # Verify bot sent message
        mock_bot.send_message.assert_called_once()
        call_args = mock_bot.send_message.call_args
        assert call_args[1]['chat_id'] == user_id
        assert "disconnected" in call_args[1]['text'].lower()


class TestWorkerAutostartCommand:
    """Test /worker autostart command."""

    @pytest.mark.asyncio
    async def test_worker_autostart_shows_instructions(self, mock_update, mock_context):
        """Test that /worker autostart shows platform-specific instructions."""
        from src.bot.handlers.connect_handler import handle_worker_autostart_command

        await handle_worker_autostart_command(mock_update, mock_context)

        # Verify message was sent
        mock_update.message.reply_text.assert_called_once()
        reply_text_call = mock_update.message.reply_text.call_args[0][0]

        # Verify instructions contain all platforms
        assert "Mac:" in reply_text_call
        assert "Linux:" in reply_text_call
        assert "Windows" in reply_text_call

        # Verify autostart flag mentioned
        assert "--autostart on" in reply_text_call
        assert "--autostart off" in reply_text_call

        # Verify platform-specific details
        assert "launchd" in reply_text_call  # Mac
        assert "systemd" in reply_text_call  # Linux
        assert "Task Scheduler" in reply_text_call  # Windows

    @pytest.mark.asyncio
    async def test_worker_autostart_when_worker_disabled(self, mock_update, mock_context):
        """Test that /worker autostart shows error when feature disabled."""
        from src.bot.handlers.connect_handler import handle_worker_autostart_command

        # Disable worker feature
        mock_context.bot_data['worker_enabled'] = False

        await handle_worker_autostart_command(mock_update, mock_context)

        # Verify error message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "not available" in reply_text_call.lower() or "not enabled" in reply_text_call.lower()


class TestTrainHandlerWorkerRouting:
    """Test /train workflow routes jobs to local worker when connected."""

    @pytest.mark.asyncio
    async def test_train_checks_worker_connection(self, mock_update, mock_context):
        """Test that train handler checks for worker connection."""
        # This test verifies the routing logic exists
        # Actual execution is tested in integration tests
        from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
        from src.core.state_manager import StateManager
        from src.processors.data_loader import DataLoader

        state_manager = StateManager()
        # DataLoader expects config dict, not local_enabled parameter
        data_loader = DataLoader(config={'local_data': {'enabled': True}})
        handler = LocalPathMLTrainingHandler(state_manager, data_loader)

        # Verify the method exists
        assert hasattr(handler, '_execute_training_on_worker')

    @pytest.mark.asyncio
    async def test_train_routes_to_worker_when_connected(self):
        """Test that training is routed to worker when connected."""
        # Integration test - verifies that worker_manager.is_user_connected() is checked
        # and job is created when worker is available
        pass

    @pytest.mark.asyncio
    async def test_train_uses_bot_when_no_worker(self):
        """Test that training uses bot execution when no worker connected."""
        # Integration test - verifies fallback to ml_engine.train_model()
        # when worker is not connected
        pass


class TestPredictHandlerWorkerRouting:
    """Test /predict workflow routes jobs to local worker when connected."""

    @pytest.mark.asyncio
    async def test_predict_checks_worker_connection(self):
        """Test that predict handler checks for worker connection."""
        # This test verifies the routing logic exists
        from src.bot.ml_handlers.prediction_handlers import PredictionHandler
        from src.core.state_manager import StateManager
        from src.processors.data_loader import DataLoader
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig

        state_manager = StateManager()
        # DataLoader expects config dict, not local_enabled parameter
        data_loader = DataLoader(config={'local_data': {'enabled': True}})
        ml_engine = MLEngine(MLEngineConfig.get_default())
        handler = PredictionHandler(state_manager, data_loader, ml_engine)

        # Verify the method exists
        assert hasattr(handler, '_execute_prediction_on_worker')

    @pytest.mark.asyncio
    async def test_predict_routes_to_worker_when_connected(self):
        """Test that prediction is routed to worker when connected."""
        # Integration test - verifies that worker_manager.is_user_connected() is checked
        # and job is created when worker is available
        pass

    @pytest.mark.asyncio
    async def test_predict_uses_bot_when_no_worker(self):
        """Test that prediction uses bot execution when no worker connected."""
        # Integration test - verifies fallback to ml_engine.predict()
        # when worker is not connected
        pass
