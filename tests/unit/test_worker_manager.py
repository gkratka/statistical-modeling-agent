"""Unit tests for worker manager."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from src.worker.worker_manager import WorkerManager, ConnectedWorker


class TestConnectedWorker:
    """Tests for ConnectedWorker dataclass."""

    def test_worker_creation(self):
        """Test creating a connected worker."""
        mock_ws = MagicMock()
        worker = ConnectedWorker(
            user_id=12345,
            websocket=mock_ws,
            machine_name="test-machine",
            connected_at=datetime.now()
        )
        assert worker.user_id == 12345
        assert worker.websocket == mock_ws
        assert worker.machine_name == "test-machine"
        assert worker.is_busy is False

    def test_worker_default_values(self):
        """Test default values for connected worker."""
        mock_ws = MagicMock()
        worker = ConnectedWorker(
            user_id=12345,
            websocket=mock_ws,
            machine_name="test-machine",
            connected_at=datetime.now()
        )
        assert worker.is_busy is False
        assert worker.current_job_id is None

    def test_worker_with_busy_status(self):
        """Test worker with busy status."""
        mock_ws = MagicMock()
        worker = ConnectedWorker(
            user_id=12345,
            websocket=mock_ws,
            machine_name="test-machine",
            connected_at=datetime.now(),
            is_busy=True,
            current_job_id="job-123"
        )
        assert worker.is_busy is True
        assert worker.current_job_id == "job-123"


class TestWorkerManager:
    """Tests for WorkerManager class."""

    @pytest.fixture
    def manager(self):
        """Create worker manager instance."""
        return WorkerManager(max_workers_per_user=1)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = MagicMock()
        ws.close = AsyncMock()
        ws.send = AsyncMock()
        return ws

    def test_manager_initialization(self, manager):
        """Test manager initializes with correct config."""
        assert manager.max_workers_per_user == 1
        assert len(manager._workers) == 0
        assert len(manager._user_workers) == 0

    def test_manager_default_values(self):
        """Test manager uses default values."""
        manager = WorkerManager()
        assert manager.max_workers_per_user == 1

    def test_register_worker(self, manager, mock_websocket):
        """Test registering a new worker."""
        result = manager.register_worker(
            user_id=12345,
            websocket=mock_websocket,
            machine_name="test-machine"
        )

        assert result is True
        assert 12345 in manager._user_workers
        assert len(manager._workers) == 1

    def test_register_worker_creates_connection(self, manager, mock_websocket):
        """Test registered worker has correct connection info."""
        manager.register_worker(
            user_id=12345,
            websocket=mock_websocket,
            machine_name="test-machine"
        )

        worker = manager._workers[mock_websocket]
        assert worker.user_id == 12345
        assert worker.machine_name == "test-machine"
        assert worker.is_busy is False

    def test_register_worker_replaces_existing(self, manager):
        """Test registering new worker replaces existing for same user."""
        ws1 = MagicMock()
        ws1.close = AsyncMock()
        ws2 = MagicMock()
        ws2.close = AsyncMock()

        manager.register_worker(12345, ws1, "machine-1")
        manager.register_worker(12345, ws2, "machine-2")

        assert ws1 not in manager._workers
        assert ws2 in manager._workers
        assert manager._user_workers[12345] == ws2

    def test_unregister_worker(self, manager, mock_websocket):
        """Test unregistering a worker."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        result = manager.unregister_worker(mock_websocket)

        assert result is True
        assert mock_websocket not in manager._workers
        assert 12345 not in manager._user_workers

    def test_unregister_worker_returns_false_for_unknown(self, manager):
        """Test unregistering unknown worker returns False."""
        mock_ws = MagicMock()
        result = manager.unregister_worker(mock_ws)

        assert result is False

    def test_get_worker_for_user(self, manager, mock_websocket):
        """Test getting worker for user."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        worker = manager.get_worker_for_user(12345)

        assert worker is not None
        assert worker.user_id == 12345
        assert worker.websocket == mock_websocket

    def test_get_worker_for_user_returns_none_if_no_worker(self, manager):
        """Test getting worker for user with no worker returns None."""
        worker = manager.get_worker_for_user(99999)

        assert worker is None

    def test_get_user_for_worker(self, manager, mock_websocket):
        """Test getting user for worker websocket."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        user_id = manager.get_user_for_worker(mock_websocket)

        assert user_id == 12345

    def test_get_user_for_worker_returns_none_for_unknown(self, manager):
        """Test getting user for unknown worker returns None."""
        mock_ws = MagicMock()
        user_id = manager.get_user_for_worker(mock_ws)

        assert user_id is None

    def test_is_user_connected(self, manager, mock_websocket):
        """Test checking if user has connected worker."""
        assert manager.is_user_connected(12345) is False

        manager.register_worker(12345, mock_websocket, "test-machine")

        assert manager.is_user_connected(12345) is True

    def test_get_machine_name(self, manager, mock_websocket):
        """Test getting machine name for user."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        name = manager.get_machine_name(12345)

        assert name == "test-machine"

    def test_get_machine_name_returns_none_if_not_connected(self, manager):
        """Test getting machine name for unconnected user returns None."""
        name = manager.get_machine_name(99999)

        assert name is None

    def test_set_worker_busy(self, manager, mock_websocket):
        """Test setting worker as busy."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        result = manager.set_worker_busy(12345, job_id="job-123")

        assert result is True
        worker = manager.get_worker_for_user(12345)
        assert worker.is_busy is True
        assert worker.current_job_id == "job-123"

    def test_set_worker_busy_returns_false_if_not_connected(self, manager):
        """Test setting busy for unconnected user returns False."""
        result = manager.set_worker_busy(99999, job_id="job-123")

        assert result is False

    def test_set_worker_idle(self, manager, mock_websocket):
        """Test setting worker as idle."""
        manager.register_worker(12345, mock_websocket, "test-machine")
        manager.set_worker_busy(12345, job_id="job-123")

        result = manager.set_worker_idle(12345)

        assert result is True
        worker = manager.get_worker_for_user(12345)
        assert worker.is_busy is False
        assert worker.current_job_id is None

    def test_set_worker_idle_returns_false_if_not_connected(self, manager):
        """Test setting idle for unconnected user returns False."""
        result = manager.set_worker_idle(99999)

        assert result is False

    def test_is_worker_available(self, manager, mock_websocket):
        """Test checking if worker is available."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        assert manager.is_worker_available(12345) is True

        manager.set_worker_busy(12345, "job-123")

        assert manager.is_worker_available(12345) is False

    def test_is_worker_available_false_if_not_connected(self, manager):
        """Test worker availability is False if not connected."""
        assert manager.is_worker_available(99999) is False

    def test_get_all_workers(self, manager):
        """Test getting all connected workers."""
        ws1 = MagicMock()
        ws1.close = AsyncMock()
        ws2 = MagicMock()
        ws2.close = AsyncMock()

        manager.register_worker(12345, ws1, "machine-1")
        manager.register_worker(67890, ws2, "machine-2")

        workers = manager.get_all_workers()

        assert len(workers) == 2
        user_ids = {w.user_id for w in workers}
        assert user_ids == {12345, 67890}

    def test_get_stats(self, manager):
        """Test getting worker statistics."""
        ws1 = MagicMock()
        ws1.close = AsyncMock()
        ws2 = MagicMock()
        ws2.close = AsyncMock()

        manager.register_worker(12345, ws1, "machine-1")
        manager.register_worker(67890, ws2, "machine-2")
        manager.set_worker_busy(12345, "job-123")

        stats = manager.get_stats()

        assert stats["total_workers"] == 2
        assert stats["busy_workers"] == 1
        assert stats["idle_workers"] == 1

    @pytest.mark.asyncio
    async def test_disconnect_user(self, manager, mock_websocket):
        """Test disconnecting a user's worker."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        result = await manager.disconnect_user(12345)

        assert result is True
        assert 12345 not in manager._user_workers
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_user_returns_false_if_not_connected(self, manager):
        """Test disconnecting unconnected user returns False."""
        result = await manager.disconnect_user(99999)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_worker(self, manager, mock_websocket):
        """Test sending message to worker."""
        manager.register_worker(12345, mock_websocket, "test-machine")

        result = await manager.send_to_worker(12345, {"type": "test", "data": "hello"})

        assert result is True
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_worker_returns_false_if_not_connected(self, manager):
        """Test sending to unconnected user returns False."""
        result = await manager.send_to_worker(99999, {"type": "test"})

        assert result is False


class TestWorkerManagerCallbacks:
    """Tests for WorkerManager callback functionality."""

    @pytest.fixture
    def manager(self):
        """Create worker manager instance."""
        return WorkerManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket."""
        ws = MagicMock()
        ws.close = AsyncMock()
        ws.send = AsyncMock()
        return ws

    def test_set_on_connect_callback(self, manager, mock_websocket):
        """Test setting connection callback."""
        callback = MagicMock()
        manager.on_connect(callback)

        manager.register_worker(12345, mock_websocket, "test-machine")

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == 12345
        assert call_args[1] == "test-machine"

    def test_set_on_disconnect_callback(self, manager, mock_websocket):
        """Test setting disconnection callback."""
        callback = MagicMock()
        manager.on_disconnect(callback)

        manager.register_worker(12345, mock_websocket, "test-machine")
        manager.unregister_worker(mock_websocket)

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == 12345
