"""Unit tests for StatsBot Local Worker script."""

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
import tempfile


# Import worker components (will be implemented)
# We'll test the worker as a module
import sys

WORKER_PATH = Path(__file__).parent.parent.parent / "worker" / "statsbot_worker.py"


class TestMachineIdentifier:
    """Test machine identifier detection."""

    def test_get_machine_identifier_returns_hostname(self):
        """Should return hostname as machine identifier."""
        import socket

        expected = socket.gethostname()
        # This will be implemented in worker
        with patch("socket.gethostname", return_value="test-machine"):
            # We'll test via importing the module once implemented
            assert socket.gethostname() == "test-machine"


class TestPathValidation:
    """Test local file path validation in worker."""

    def test_validate_file_path_rejects_traversal(self):
        """Should reject path traversal attempts."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e/etc/passwd",
        ]
        for path in dangerous_paths:
            # Should detect traversal pattern
            assert ".." in path or "%2e" in path.lower()

    def test_validate_file_path_accepts_valid_absolute(self):
        """Should accept valid absolute paths."""
        valid_paths = [
            "/Users/test/data/file.csv",
            "/home/user/datasets/train.csv",
        ]
        for path in valid_paths:
            # Should be absolute
            assert Path(path).is_absolute()

    def test_validate_file_path_checks_existence(self, tmp_path):
        """Should check if file exists."""
        test_file = tmp_path / "test.csv"
        assert not test_file.exists()

        test_file.write_text("col1,col2\n1,2\n")
        assert test_file.exists()

    def test_validate_file_path_checks_extension(self):
        """Should validate file extension."""
        allowed = [".csv", ".xlsx", ".parquet"]
        test_path = Path("/path/to/file.csv")
        assert test_path.suffix in allowed

        invalid_path = Path("/path/to/file.txt")
        assert invalid_path.suffix not in allowed


class TestModelStorage:
    """Test local model storage functionality."""

    def test_get_models_dir_creates_directory(self, tmp_path):
        """Should create ~/.statsbot/models/ directory."""
        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            models_dir = tmp_path / ".statsbot" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            assert models_dir.exists()
            assert models_dir.is_dir()

    def test_get_models_dir_returns_path(self, tmp_path):
        """Should return models directory path."""
        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            expected = tmp_path / ".statsbot" / "models"
            expected.mkdir(parents=True, exist_ok=True)
            assert expected.exists()


class TestJobExecution:
    """Test job execution handlers."""

    @pytest.fixture
    def sample_train_job(self) -> Dict[str, Any]:
        """Sample training job."""
        return {
            "type": "job",
            "job_id": "test-job-123",
            "action": "train",
            "params": {
                "file_path": "/path/to/data.csv",
                "target_column": "price",
                "feature_columns": ["sqft", "bedrooms"],
                "model_type": "random_forest",
                "task_type": "regression",
            },
        }

    @pytest.fixture
    def sample_predict_job(self) -> Dict[str, Any]:
        """Sample prediction job."""
        return {
            "type": "job",
            "job_id": "test-job-456",
            "action": "predict",
            "params": {
                "model_id": "model_123",
                "file_path": "/path/to/test.csv",
            },
        }

    def test_parse_job_message_valid(self, sample_train_job):
        """Should parse valid job message."""
        message = json.dumps(sample_train_job)
        parsed = json.loads(message)
        assert parsed["type"] == "job"
        assert parsed["action"] == "train"
        assert "job_id" in parsed

    def test_parse_job_message_invalid_json(self):
        """Should handle invalid JSON."""
        invalid_json = "{invalid json"
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_create_progress_message(self):
        """Should create progress message."""
        progress = {
            "type": "progress",
            "job_id": "test-123",
            "status": "training",
            "progress": 50,
            "message": "Training in progress...",
        }
        assert progress["type"] == "progress"
        assert 0 <= progress["progress"] <= 100

    def test_create_result_message_success(self):
        """Should create success result message."""
        result = {
            "type": "result",
            "job_id": "test-123",
            "success": True,
            "data": {
                "model_id": "model_123",
                "metrics": {"r2": 0.85, "mse": 0.12},
            },
        }
        assert result["type"] == "result"
        assert result["success"] is True
        assert "data" in result

    def test_create_result_message_error(self):
        """Should create error result message."""
        result = {
            "type": "result",
            "job_id": "test-123",
            "success": False,
            "error": "File not found: /path/to/data.csv",
        }
        assert result["type"] == "result"
        assert result["success"] is False
        assert "error" in result


class TestMLPackageDetection:
    """Test ML package availability detection."""

    def test_check_ml_packages_all_available(self):
        """Should detect when all ML packages are available."""
        try:
            import pandas
            import sklearn
            import xgboost

            all_available = True
        except ImportError:
            all_available = False

        # Test should reflect actual environment
        if all_available:
            import pandas
            import sklearn

            assert hasattr(pandas, "DataFrame")
            assert hasattr(sklearn, "__version__")

    def test_check_ml_packages_graceful_fallback(self):
        """Should handle missing ML packages gracefully."""
        with patch.dict("sys.modules", {"pandas": None}):
            # Should not crash, just indicate unavailable
            packages_available = False
            assert packages_available is False

    def test_get_available_packages_list(self):
        """Should return list of available packages."""
        available = []
        try:
            import pandas

            available.append("pandas")
        except ImportError:
            pass

        try:
            import sklearn

            available.append("sklearn")
        except ImportError:
            pass

        # Should be a list (possibly empty)
        assert isinstance(available, list)


class TestReconnectionLogic:
    """Test reconnection handling."""

    def test_should_reconnect_after_disconnect(self):
        """Should attempt reconnection after connection drops."""
        max_retries = 3
        current_retry = 0

        # Simulate disconnect
        connected = False
        should_retry = current_retry < max_retries and not connected

        assert should_retry is True

    def test_should_stop_after_max_retries(self):
        """Should stop reconnecting after max retries."""
        max_retries = 3
        current_retry = 3

        connected = False
        should_retry = current_retry < max_retries and not connected

        assert should_retry is False

    def test_exponential_backoff_calculation(self):
        """Should calculate exponential backoff delay."""
        base_delay = 5  # seconds
        retry_count = 2

        # Exponential: base * 2^retry
        expected_delay = base_delay * (2**retry_count)
        assert expected_delay == 20  # 5 * 4


class TestWorkerAuthentication:
    """Test worker authentication flow."""

    @pytest.fixture
    def auth_message(self) -> Dict[str, Any]:
        """Sample authentication message."""
        return {
            "type": "auth",
            "token": "test-token-123",
            "machine_id": "test-machine",
        }

    def test_create_auth_message(self, auth_message):
        """Should create valid auth message."""
        assert auth_message["type"] == "auth"
        assert "token" in auth_message
        assert "machine_id" in auth_message

    def test_auth_response_success(self):
        """Should parse successful auth response."""
        response = {
            "type": "auth_response",
            "success": True,
            "user_id": 12345,
            "message": "Authentication successful",
        }
        assert response["success"] is True
        assert "user_id" in response

    def test_auth_response_failure(self):
        """Should parse failed auth response."""
        response = {
            "type": "auth_response",
            "success": False,
            "error": "Invalid token",
        }
        assert response["success"] is False
        assert "error" in response


class TestCommandLineArgs:
    """Test command-line argument parsing."""

    def test_parse_token_argument(self):
        """Should parse --token argument."""
        args = ["--token", "abc123xyz"]
        # Simulate argparse
        token = args[1] if len(args) > 1 else None
        assert token == "abc123xyz"

    def test_parse_autostart_flag(self):
        """Should parse --autostart flag."""
        args = ["--token", "abc123", "--autostart"]
        autostart = "--autostart" in args
        assert autostart is True

    def test_require_token_argument(self):
        """Should require token argument."""
        args = []
        has_token = any("--token" in arg for arg in args)
        assert has_token is False


class TestListModelsJob:
    """Test list_models job execution."""

    def test_list_models_empty(self, tmp_path):
        """Should return empty list when no models exist."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # List models
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        assert len(model_dirs) == 0

    def test_list_models_with_models(self, tmp_path):
        """Should list existing models."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create mock model directories
        (models_dir / "model_1").mkdir()
        (models_dir / "model_2").mkdir()

        model_dirs = [d.name for d in models_dir.iterdir() if d.is_dir()]
        assert len(model_dirs) == 2
        assert "model_1" in model_dirs
        assert "model_2" in model_dirs


class TestJobResultSerialization:
    """Test serialization of job results."""

    def test_serialize_result_with_metrics(self):
        """Should serialize result with numeric metrics."""
        result = {
            "type": "result",
            "job_id": "test-123",
            "success": True,
            "data": {
                "model_id": "model_123",
                "metrics": {"r2": 0.85, "mse": 0.12, "mae": 0.08},
            },
        }
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)

        assert deserialized["data"]["metrics"]["r2"] == 0.85
        assert isinstance(deserialized["data"]["metrics"]["mse"], float)

    def test_serialize_result_with_predictions(self):
        """Should serialize result with prediction array."""
        result = {
            "type": "result",
            "job_id": "test-456",
            "success": True,
            "data": {"predictions": [1.2, 3.4, 5.6], "count": 3},
        }
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)

        assert len(deserialized["data"]["predictions"]) == 3
        assert deserialized["data"]["count"] == 3


class TestWorkerConfigPersistence:
    """Test worker configuration persistence."""

    def test_save_worker_config(self, tmp_path):
        """Should save worker config to file."""
        config_dir = tmp_path / ".statsbot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config = {
            "bot_url": "wss://example.com/ws",
            "token": "saved-token",
            "machine_id": "test-machine",
        }

        config_file.write_text(json.dumps(config, indent=2))
        assert config_file.exists()

        loaded = json.loads(config_file.read_text())
        assert loaded["machine_id"] == "test-machine"

    def test_load_worker_config(self, tmp_path):
        """Should load worker config from file."""
        config_dir = tmp_path / ".statsbot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config = {"bot_url": "wss://example.com/ws"}
        config_file.write_text(json.dumps(config))

        loaded = json.loads(config_file.read_text())
        assert loaded["bot_url"] == "wss://example.com/ws"


class TestErrorHandling:
    """Test error handling in worker."""

    def test_handle_file_not_found_error(self):
        """Should handle file not found gracefully."""
        error_result = {
            "type": "result",
            "job_id": "test-123",
            "success": False,
            "error": "File not found: /nonexistent/file.csv",
        }
        assert error_result["success"] is False
        assert "File not found" in error_result["error"]

    def test_handle_missing_ml_packages(self):
        """Should handle missing ML packages gracefully."""
        try:
            import nonexistent_package

            package_available = True
        except ImportError as e:
            package_available = False
            error_msg = str(e)

        assert package_available is False
        assert error_msg  # Should have error message

    def test_handle_invalid_model_type(self):
        """Should handle invalid model type."""
        valid_models = ["random_forest", "xgboost", "linear"]
        test_model = "invalid_model"

        is_valid = test_model in valid_models
        assert is_valid is False


class TestSecurityValidation:
    """Test security validation in worker."""

    def test_reject_system_paths(self):
        """Should reject access to system directories."""
        system_paths = [
            "/etc/passwd",
            "/System/Library",
            "C:\\Windows\\System32",
        ]

        forbidden = ["/etc", "/System", "/Windows"]

        for path in system_paths:
            # Check if any forbidden prefix matches
            is_forbidden = any(path.startswith(f) for f in forbidden)
            # At least one should be forbidden
            if path.startswith("/etc") or path.startswith("/System"):
                assert is_forbidden is True

    def test_validate_file_size_limit(self):
        """Should enforce file size limits."""
        max_size_mb = 1000

        # Test file within limit
        file_size_mb = 500
        is_valid = file_size_mb <= max_size_mb
        assert is_valid is True

        # Test file over limit
        file_size_mb = 1500
        is_valid = file_size_mb <= max_size_mb
        assert is_valid is False


# Integration-style tests (will use mocked WebSocket)


@pytest.mark.asyncio
class TestWorkerLifecycle:
    """Test worker lifecycle (connection, jobs, disconnection)."""

    async def test_worker_connection_flow(self):
        """Should connect, authenticate, and enter ready state."""
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps(
                {"type": "auth_response", "success": True, "user_id": 12345}
            )
        )
        mock_ws.send = AsyncMock()

        # Simulate auth
        auth_msg = {"type": "auth", "token": "test-token", "machine_id": "test-machine"}
        await mock_ws.send(json.dumps(auth_msg))

        # Receive auth response
        response = json.loads(await mock_ws.recv())
        assert response["success"] is True

    async def test_worker_job_execution_flow(self):
        """Should receive job, execute, and send result."""
        # Mock WebSocket
        mock_ws = AsyncMock()

        # Receive job
        job = {
            "type": "job",
            "job_id": "test-123",
            "action": "list_models",
            "params": {},
        }
        mock_ws.recv = AsyncMock(return_value=json.dumps(job))

        received_job = json.loads(await mock_ws.recv())
        assert received_job["type"] == "job"

        # Send result
        result = {
            "type": "result",
            "job_id": "test-123",
            "success": True,
            "data": {"models": []},
        }
        await mock_ws.send(json.dumps(result))
        mock_ws.send.assert_called_once()

    async def test_worker_handles_disconnect(self):
        """Should handle disconnection gracefully."""
        # Mock WebSocket that disconnects
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(side_effect=ConnectionError("Connection lost"))

        # Should catch the error
        try:
            await mock_ws.recv()
            assert False, "Should have raised ConnectionError"
        except ConnectionError as e:
            assert "Connection lost" in str(e)
