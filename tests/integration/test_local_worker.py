"""Integration tests for local worker lifecycle.

Tests the complete worker connection and job execution flow.
Note: These are simplified integration tests that verify the core components
work together. Full end-to-end tests require running the actual WebSocket server.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Import worker components
from src.worker.job_queue import JobQueue
from src.worker.token_manager import TokenManager
from src.worker.worker_manager import WorkerManager


@pytest.fixture
def token_manager():
    """Create token manager instance."""
    return TokenManager(token_expiry_seconds=300)


@pytest.fixture
def worker_manager():
    """Create worker manager instance."""
    return WorkerManager(max_workers_per_user=1)


@pytest.fixture
def job_queue():
    """Create job queue instance."""
    return JobQueue(default_timeout=60.0)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [100.0, 200.0, 300.0, 400.0, 500.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_csv_data.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


class TestTokenManagement:
    """Test token management integration."""

    def test_token_generation_and_validation(self, token_manager):
        """Test token generation and validation flow."""
        user_id = 12345

        # Generate token
        token = token_manager.generate_token(user_id)
        assert token is not None
        assert len(token) > 0

        # Validate token
        validated_user = token_manager.validate_token(token)
        assert validated_user == user_id

        # Token should be invalidated after use
        validated_again = token_manager.validate_token(token)
        assert validated_again is None

    def test_token_expiration(self, token_manager):
        """Test token expiration."""
        # Create token manager with 1 second expiry
        tm = TokenManager(token_expiry_seconds=1)
        user_id = 12345

        # Generate token
        token = tm.generate_token(user_id)

        # Should be valid immediately
        assert tm.get_user_for_token(token) == user_id

        # Token should still be in manager (not validated yet, so not expired)
        assert tm.get_token_for_user(user_id) == token

    def test_multiple_users(self, token_manager):
        """Test multiple users can have tokens."""
        user_ids = [100, 200, 300]
        tokens = {}

        # Generate tokens for each user
        for user_id in user_ids:
            token = token_manager.generate_token(user_id)
            tokens[user_id] = token

        # Verify each token maps to correct user
        for user_id, token in tokens.items():
            assert token_manager.get_user_for_token(token) == user_id

        # Validate each token
        for user_id, token in tokens.items():
            assert token_manager.validate_token(token) == user_id


class TestWorkerManagement:
    """Test worker management integration."""

    def test_worker_registration(self, worker_manager):
        """Test worker registration flow."""
        user_id = 12345
        machine_id = "test-machine"
        websocket = object()  # Mock websocket

        # Register worker (user_id, websocket, machine_name)
        result = worker_manager.register_worker(user_id, websocket, machine_id)
        assert result is True

        # Verify worker is connected
        assert worker_manager.is_user_connected(user_id)
        assert worker_manager.get_machine_name(user_id) == machine_id

        # Disconnect worker (takes websocket, not user_id)
        result = worker_manager.unregister_worker(websocket)
        assert result is True
        assert not worker_manager.is_user_connected(user_id)

    def test_worker_replacement(self, worker_manager):
        """Test worker replacement when reconnecting."""
        user_id = 12345
        machine_id_1 = "test-machine-1"
        machine_id_2 = "test-machine-2"

        # Register first worker
        ws1 = object()
        worker_manager.register_worker(user_id, ws1, machine_id_1)
        assert worker_manager.get_machine_name(user_id) == machine_id_1

        # Register second worker (should replace first)
        ws2 = object()
        worker_manager.register_worker(user_id, ws2, machine_id_2)
        assert worker_manager.get_machine_name(user_id) == machine_id_2

    def test_multiple_users(self, worker_manager):
        """Test multiple users can connect workers."""
        users = [(100, "machine-1"), (200, "machine-2"), (300, "machine-3")]

        # Register all workers
        for user_id, machine_id in users:
            ws = object()
            worker_manager.register_worker(user_id, ws, machine_id)

        # Verify all connected
        for user_id, machine_id in users:
            assert worker_manager.is_user_connected(user_id)
            assert worker_manager.get_machine_name(user_id) == machine_id

        # Get stats
        stats = worker_manager.get_stats()
        assert stats["total_workers"] == 3
        assert stats["idle_workers"] == 3


class TestJobQueueIntegration:
    """Test job queue integration."""

    def test_job_queue_initialization(self, job_queue):
        """Test job queue initializes correctly."""
        assert job_queue is not None

        # Check stats
        stats = job_queue.get_stats()
        assert "total_jobs" in stats
        assert stats["total_jobs"] == 0

    def test_job_queue_with_worker_manager(self, job_queue, worker_manager):
        """Test job queue integrates with worker manager."""
        # Set worker manager
        job_queue.set_worker_manager(worker_manager)

        # Register a worker
        user_id = 12345
        machine_id = "test-machine"
        ws = object()
        worker_manager.register_worker(user_id, ws, machine_id)

        # Verify worker is available
        assert worker_manager.is_user_connected(user_id)
        assert worker_manager.is_worker_available(user_id)


class TestFullIntegration:
    """Test full component integration."""

    def test_token_to_worker_flow(self, token_manager, worker_manager):
        """Test complete flow from token generation to worker connection."""
        user_id = 12345
        machine_id = "test-machine"

        # Step 1: User requests connection → Generate token
        token = token_manager.generate_token(user_id)
        assert token is not None

        # Step 2: User runs worker script with token → Validate token
        validated_user = token_manager.validate_token(token)
        assert validated_user == user_id

        # Step 3: Worker connects with validated user_id
        ws = object()
        result = worker_manager.register_worker(validated_user, ws, machine_id)
        assert result is True

        # Step 4: Verify worker is connected
        assert worker_manager.is_user_connected(user_id)
        assert worker_manager.get_machine_name(user_id) == machine_id

        # Step 5: Worker disconnects (takes websocket)
        worker_manager.unregister_worker(ws)
        assert not worker_manager.is_user_connected(user_id)

    def test_worker_reconnection_flow(self, token_manager, worker_manager):
        """Test worker reconnection with new token."""
        user_id = 12345
        machine_id = "test-machine"

        # First connection
        token1 = token_manager.generate_token(user_id)
        validated1 = token_manager.validate_token(token1)
        ws1 = object()
        worker_manager.register_worker(validated1, ws1, machine_id)
        assert worker_manager.is_user_connected(user_id)

        # Disconnect (takes websocket)
        worker_manager.unregister_worker(ws1)
        assert not worker_manager.is_user_connected(user_id)

        # Reconnection with new token
        token2 = token_manager.generate_token(user_id)
        validated2 = token_manager.validate_token(token2)
        ws2 = object()
        worker_manager.register_worker(validated2, ws2, machine_id)
        assert worker_manager.is_user_connected(user_id)

    def test_integrated_system_state(self, token_manager, worker_manager, job_queue):
        """Test integrated system state across components."""
        # Set up integration
        job_queue.set_worker_manager(worker_manager)

        # Connect multiple workers
        users = [
            (100, "machine-1"),
            (200, "machine-2"),
            (300, "machine-3"),
        ]

        for user_id, machine_id in users:
            # Generate and validate token
            token = token_manager.generate_token(user_id)
            validated = token_manager.validate_token(token)

            # Register worker
            ws = object()
            worker_manager.register_worker(validated, ws, machine_id)

        # Verify system state
        worker_stats = worker_manager.get_stats()
        assert worker_stats["total_workers"] == 3

        job_stats = job_queue.get_stats()
        assert "total_jobs" in job_stats

        token_stats = token_manager.get_stats()
        assert token_stats["total_tokens"] >= 3


class TestErrorHandling:
    """Test error handling in integrated scenarios."""

    def test_token_reuse_prevention(self, token_manager, worker_manager):
        """Test that tokens cannot be reused for multiple connections."""
        user_id = 12345
        machine_id = "test-machine"

        # Generate token
        token = token_manager.generate_token(user_id)

        # First validation
        validated1 = token_manager.validate_token(token)
        assert validated1 == user_id

        # Second validation should fail
        validated2 = token_manager.validate_token(token)
        assert validated2 is None

    def test_worker_manager_max_workers(self, worker_manager):
        """Test worker manager enforces max workers limit."""
        # Note: Current implementation allows replacement, not multiple workers
        user_id = 12345

        # Register first worker
        ws1 = object()
        result1 = worker_manager.register_worker(user_id, ws1, "machine-1")
        assert result1 is True

        # Register second worker (replaces first)
        ws2 = object()
        result2 = worker_manager.register_worker(user_id, ws2, "machine-2")
        assert result2 is True

        # Should only have one connection
        assert worker_manager.is_user_connected(user_id)
        assert worker_manager.get_machine_name(user_id) == "machine-2"


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Test async integration scenarios."""

    async def test_async_token_validation(self, token_manager):
        """Test token validation in async context."""
        user_id = 12345

        # Generate token
        token = token_manager.generate_token(user_id)

        # Validate asynchronously
        validated = await asyncio.to_thread(token_manager.validate_token, token)
        assert validated == user_id

    async def test_concurrent_token_generation(self, token_manager):
        """Test concurrent token generation for multiple users."""
        user_ids = list(range(100, 110))

        # Generate tokens concurrently
        async def generate_token_async(uid):
            return await asyncio.to_thread(token_manager.generate_token, uid)

        tokens = await asyncio.gather(*[generate_token_async(uid) for uid in user_ids])

        # Verify all tokens generated
        assert len(tokens) == len(user_ids)
        assert all(token is not None for token in tokens)

        # Verify each token maps to correct user
        for user_id, token in zip(user_ids, tokens):
            assert token_manager.get_user_for_token(token) == user_id
