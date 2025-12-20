"""Unit tests for WebSocket server."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.worker.websocket_server import WebSocketServer


class TestWebSocketServer:
    """Tests for WebSocketServer class."""

    @pytest.fixture
    def server(self):
        """Create WebSocket server instance."""
        return WebSocketServer(host="localhost", port=8765)

    def test_server_initialization(self, server):
        """Test server initializes with correct config."""
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.connections == {}
        assert server._server is None

    def test_server_default_port(self):
        """Test server uses default port when not specified."""
        server = WebSocketServer()
        assert server.port == 8765

    @pytest.mark.asyncio
    async def test_start_server(self, server):
        """Test server starts correctly."""
        with patch('websockets.serve', new_callable=AsyncMock) as mock_serve:
            mock_server = AsyncMock()
            mock_serve.return_value = mock_server

            await server.start()

            mock_serve.assert_called_once()
            assert server._server == mock_server

    @pytest.mark.asyncio
    async def test_stop_server(self, server):
        """Test server stops correctly."""
        server._server = AsyncMock()
        server._server.close = MagicMock()
        server._server.wait_closed = AsyncMock()

        await server.stop()

        server._server.close.assert_called_once()
        server._server.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_server_when_not_running(self, server):
        """Test stopping server that isn't running doesn't raise."""
        await server.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_register_connection(self, server):
        """Test registering a worker connection."""
        mock_websocket = AsyncMock()
        user_id = 12345

        server.register_connection(user_id, mock_websocket)

        assert user_id in server.connections
        assert server.connections[user_id] == mock_websocket

    @pytest.mark.asyncio
    async def test_unregister_connection(self, server):
        """Test unregistering a worker connection."""
        mock_websocket = AsyncMock()
        user_id = 12345
        server.connections[user_id] = mock_websocket

        server.unregister_connection(user_id)

        assert user_id not in server.connections

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_connection(self, server):
        """Test unregistering connection that doesn't exist doesn't raise."""
        server.unregister_connection(99999)  # Should not raise

    def test_is_connected(self, server):
        """Test checking if user has connected worker."""
        mock_websocket = AsyncMock()
        user_id = 12345

        assert not server.is_connected(user_id)

        server.connections[user_id] = mock_websocket

        assert server.is_connected(user_id)

    def test_get_connection(self, server):
        """Test getting worker connection for user."""
        mock_websocket = AsyncMock()
        user_id = 12345
        server.connections[user_id] = mock_websocket

        result = server.get_connection(user_id)

        assert result == mock_websocket

    def test_get_connection_not_found(self, server):
        """Test getting connection for user without worker returns None."""
        result = server.get_connection(99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_to_worker(self, server):
        """Test sending message to worker."""
        mock_websocket = AsyncMock()
        user_id = 12345
        server.connections[user_id] = mock_websocket

        message = {"type": "job", "action": "train"}
        await server.send_to_worker(user_id, message)

        mock_websocket.send.assert_awaited_once_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_send_to_worker_not_connected(self, server):
        """Test sending to disconnected worker raises error."""
        with pytest.raises(ConnectionError, match="No worker connected"):
            await server.send_to_worker(99999, {"type": "job"})

    @pytest.mark.asyncio
    async def test_broadcast_to_all_workers(self, server):
        """Test broadcasting message to all workers."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        server.connections[1] = mock_ws1
        server.connections[2] = mock_ws2

        message = {"type": "broadcast", "data": "test"}
        await server.broadcast(message)

        mock_ws1.send.assert_awaited_once()
        mock_ws2.send.assert_awaited_once()

    def test_connection_count(self, server):
        """Test getting number of connected workers."""
        assert server.connection_count == 0

        server.connections[1] = AsyncMock()
        server.connections[2] = AsyncMock()

        assert server.connection_count == 2


class TestWebSocketServerHandlers:
    """Tests for WebSocket connection handlers."""

    @pytest.fixture
    def server(self):
        """Create WebSocket server instance."""
        return WebSocketServer(host="localhost", port=8765)

    @pytest.mark.asyncio
    async def test_handle_auth_message_valid(self, server):
        """Test handling valid authentication message."""
        mock_websocket = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "auth",
            "token": "valid-token-123"
        }))

        with patch.object(server, '_validate_token', return_value=12345):
            user_id = await server._handle_auth(mock_websocket)
            assert user_id == 12345

    @pytest.mark.asyncio
    async def test_handle_auth_message_invalid(self, server):
        """Test handling invalid authentication message."""
        mock_websocket = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "auth",
            "token": "invalid-token"
        }))

        with patch.object(server, '_validate_token', return_value=None):
            user_id = await server._handle_auth(mock_websocket)
            assert user_id is None
