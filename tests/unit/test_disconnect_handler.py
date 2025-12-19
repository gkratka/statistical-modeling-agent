"""Tests for /disconnect command handler.

This module tests the /disconnect command handler following TDD approach.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, User, Message, Chat


@pytest.fixture
def mock_update():
    """Create a mock Telegram update."""
    update = AsyncMock(spec=Update)
    update.effective_user = User(id=12345, first_name="Test", is_bot=False)
    update.effective_chat = Chat(id=67890, type="private")
    update.message = AsyncMock(spec=Message)
    update.message.reply_text = AsyncMock()
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
    worker_manager.disconnect_user = AsyncMock(return_value=True)

    context.bot_data['websocket_server'].worker_manager = worker_manager

    return context


class TestDisconnectHandler:
    """Test /disconnect command handler."""

    @pytest.mark.asyncio
    async def test_disconnect_when_no_worker_connected(self, mock_update, mock_context):
        """Test that /disconnect shows info message when no worker connected."""
        from src.bot.handlers.connect_handler import handle_disconnect_command

        # No worker connected
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=False)

        await handle_disconnect_command(mock_update, mock_context)

        # Verify info message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "no worker" in reply_text_call.lower() or "not connected" in reply_text_call.lower()
        assert "/connect" in reply_text_call

    @pytest.mark.asyncio
    async def test_disconnect_when_worker_connected(self, mock_update, mock_context):
        """Test that /disconnect disconnects worker and shows success message."""
        from src.bot.handlers.connect_handler import handle_disconnect_command

        # Worker connected
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=True)
        mock_context.bot_data['websocket_server'].worker_manager.disconnect_user = AsyncMock(return_value=True)

        await handle_disconnect_command(mock_update, mock_context)

        # Verify disconnect was called
        mock_context.bot_data['websocket_server'].worker_manager.disconnect_user.assert_called_once_with(12345)

        # Verify success message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "disconnected" in reply_text_call.lower()
        assert "/connect" in reply_text_call

    @pytest.mark.asyncio
    async def test_disconnect_when_worker_feature_disabled(self, mock_update, mock_context):
        """Test that /disconnect shows error when worker feature disabled."""
        from src.bot.handlers.connect_handler import handle_disconnect_command

        # Disable worker feature by removing websocket_server
        mock_context.bot_data['websocket_server'] = None

        await handle_disconnect_command(mock_update, mock_context)

        # Verify error message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "not available" in reply_text_call.lower() or "not enabled" in reply_text_call.lower()

    @pytest.mark.asyncio
    async def test_disconnect_handles_disconnect_failure(self, mock_update, mock_context):
        """Test that /disconnect handles failure gracefully."""
        from src.bot.handlers.connect_handler import handle_disconnect_command

        # Worker connected but disconnect fails
        mock_context.bot_data['websocket_server'].worker_manager.is_user_connected = MagicMock(return_value=True)
        mock_context.bot_data['websocket_server'].worker_manager.disconnect_user = AsyncMock(return_value=False)

        await handle_disconnect_command(mock_update, mock_context)

        # Verify error message
        reply_text_call = mock_update.message.reply_text.call_args[0][0]
        assert "failed" in reply_text_call.lower() or "error" in reply_text_call.lower()
