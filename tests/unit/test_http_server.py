"""Unit tests for HTTP server."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from src.worker.http_server import HTTPServer


class TestHTTPServer:
    """Tests for HTTPServer class."""

    @pytest.fixture
    def server(self, tmp_path):
        """Create HTTP server instance with temp worker script."""
        script_path = tmp_path / "worker.py"
        script_path.write_text("print('Hello from worker')")
        return HTTPServer(host="localhost", port=8080, worker_script_path=script_path)

    def test_server_initialization(self, server):
        """Test server initializes with correct config."""
        assert server.host == "localhost"
        assert server.port == 8080
        assert server._app is None

    def test_server_default_values(self):
        """Test server uses default values."""
        server = HTTPServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8080
        assert server.worker_script_path == Path("worker/statsbot_worker.py")

    @pytest.mark.asyncio
    async def test_handle_worker_script_success(self, server):
        """Test serving worker script successfully."""
        mock_request = MagicMock()

        response = await server._handle_worker_script(mock_request)

        assert response.status == 200
        assert response.content_type == "text/x-python"
        assert "Hello from worker" in response.text

    @pytest.mark.asyncio
    async def test_handle_worker_script_not_found(self):
        """Test handling missing worker script."""
        server = HTTPServer(worker_script_path=Path("/nonexistent/path.py"))
        mock_request = MagicMock()

        response = await server._handle_worker_script(mock_request)

        assert response.status == 404
        assert "not found" in response.text.lower()

    @pytest.mark.asyncio
    async def test_handle_health(self, server):
        """Test health check endpoint."""
        mock_request = MagicMock()

        response = await server._handle_health(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
