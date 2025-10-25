"""
Tests for RunPodSecurityManager.

This module tests the RunPod security manager including user isolation
validation and audit logging.

Author: Statistical Modeling Agent
Created: 2025-10-24 (Task 4.6: RunPod Configuration Tests)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.cloud.security import RunPodSecurityManager
from src.cloud.runpod_config import RunPodConfig


@pytest.fixture
def runpod_config():
    """Create valid RunPod configuration for testing."""
    return RunPodConfig(
        runpod_api_key="test-api-key",
        storage_endpoint="https://storage.runpod.io",
        network_volume_id="vol-123",
        default_gpu_type="NVIDIA RTX A5000",
        cloud_type="COMMUNITY",
        storage_access_key="test-access-key",
        storage_secret_key="test-secret-key",
        data_prefix="datasets",
        models_prefix="models",
        results_prefix="results"
    )


@pytest.fixture
def security_manager(runpod_config):
    """Create RunPodSecurityManager instance."""
    return RunPodSecurityManager(runpod_config)


class TestRunPodSecurityManagerInitialization:
    """Test RunPodSecurityManager initialization."""

    def test_init_stores_config(self, runpod_config):
        """Initialization should store config."""
        manager = RunPodSecurityManager(runpod_config)

        assert manager.config == runpod_config


class TestRunPodUserStorageAccessValidation:
    """Test RunPod user storage access validation."""

    def test_validate_write_access_valid_dataset_path(self, security_manager):
        """Valid dataset write path should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="datasets/user_12345/data.csv",
            user_id=12345,
            operation="write"
        )

        assert result is True

    def test_validate_write_access_valid_model_path(self, security_manager):
        """Valid model write path should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="models/user_12345/model_abc/model.pkl",
            user_id=12345,
            operation="write"
        )

        assert result is True

    def test_validate_write_access_valid_predictions_path(self, security_manager):
        """Valid predictions write path should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="predictions/user_12345/predictions.csv",
            user_id=12345,
            operation="write"
        )

        assert result is True

    def test_validate_write_access_valid_results_path(self, security_manager):
        """Valid results write path should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="results/user_12345/output.csv",
            user_id=12345,
            operation="write"
        )

        assert result is True

    def test_validate_write_access_wrong_user_raises_error(self, security_manager):
        """Write to another user's path should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            security_manager.validate_user_storage_access(
                storage_key="datasets/user_99999/data.csv",
                user_id=12345,
                operation="write"
            )

        assert "write access denied" in str(exc_info.value).lower()
        assert "12345" in str(exc_info.value)

    def test_validate_write_access_unauthorized_prefix_raises_error(self, security_manager):
        """Write to unauthorized prefix should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            security_manager.validate_user_storage_access(
                storage_key="unauthorized/user_12345/data.csv",
                user_id=12345,
                operation="write"
            )

        assert "write access denied" in str(exc_info.value).lower()

    def test_validate_read_access_own_data_allowed(self, security_manager):
        """Read from own data should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="datasets/user_12345/data.csv",
            user_id=12345,
            operation="read"
        )

        assert result is True

    def test_validate_read_access_other_user_raises_error(self, security_manager):
        """Read from another user's data should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            security_manager.validate_user_storage_access(
                storage_key="datasets/user_99999/data.csv",
                user_id=12345,
                operation="read"
            )

        assert "read access denied" in str(exc_info.value).lower()
        assert "99999" in str(exc_info.value)
        assert "12345" in str(exc_info.value)

    def test_validate_read_access_public_data_allowed(self, security_manager):
        """Read from public data (no user prefix) should be allowed."""
        result = security_manager.validate_user_storage_access(
            storage_key="public/shared_data.csv",
            user_id=12345,
            operation="read"
        )

        assert result is True


class TestRunPodAuditLogging:
    """Test RunPod audit logging functionality."""

    def test_audit_log_operation_creates_log_entry(self, security_manager, tmp_path, monkeypatch):
        """audit_log_operation should create log entry with RunPod provider tag."""
        # Use temp directory for audit logs
        audit_log_file = tmp_path / "cloud_audit.json"
        monkeypatch.setattr(Path, '__new__', lambda *args: audit_log_file if 'cloud_audit.json' in str(args[1] if len(args) > 1 else '') else Path(*args[1:]))

        # Create temporary audit log directory
        audit_dir = tmp_path / "data" / "logs"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_log_file = audit_dir / "cloud_audit.json"

        # Patch the audit log file path
        import src.cloud.security as security_module
        original_path = security_module.Path
        monkeypatch.setattr(security_module, 'Path', lambda x: audit_log_file if 'cloud_audit.json' in x else original_path(x))

        # Log operation
        security_manager.audit_log_operation(
            user_id=12345,
            operation="runpod_training",
            resource="pod-abc123",
            success=True,
            gpu_type="NVIDIA RTX A5000",
            cost=0.145,
            duration=1800
        )

        # Verify log file was created
        assert audit_log_file.exists()

        # Read log entry
        with open(audit_log_file, 'r') as f:
            logs = json.load(f)

        assert len(logs) == 1
        entry = logs[0]

        assert entry['user_id'] == 12345
        assert entry['operation'] == "runpod_training"
        assert entry['resource'] == "pod-abc123"
        assert entry['success'] is True
        assert entry['provider'] == 'runpod'  # RunPod provider tag
        assert entry['gpu_type'] == "NVIDIA RTX A5000"
        assert entry['cost'] == 0.145
        assert entry['duration'] == 1800
        assert 'timestamp' in entry

    def test_audit_log_operation_appends_to_existing_logs(self, security_manager, tmp_path, monkeypatch):
        """audit_log_operation should append to existing log file."""
        # Create temporary audit log directory
        audit_dir = tmp_path / "data" / "logs"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_log_file = audit_dir / "cloud_audit.json"

        # Create existing log with one entry
        existing_logs = [
            {
                'timestamp': '2025-10-24T10:00:00',
                'user_id': 11111,
                'operation': 'runpod_prediction',
                'resource': 'endpoint-xyz',
                'success': True,
                'provider': 'runpod'
            }
        ]

        with open(audit_log_file, 'w') as f:
            json.dump(existing_logs, f)

        # Patch the audit log file path
        import src.cloud.security as security_module
        original_path = security_module.Path
        monkeypatch.setattr(security_module, 'Path', lambda x: audit_log_file if 'cloud_audit.json' in x else original_path(x))

        # Log new operation
        security_manager.audit_log_operation(
            user_id=22222,
            operation="storage_upload",
            resource="datasets/user_22222/data.csv",
            success=True
        )

        # Read all logs
        with open(audit_log_file, 'r') as f:
            logs = json.load(f)

        # Should have 2 entries now
        assert len(logs) == 2

        # First entry unchanged
        assert logs[0]['user_id'] == 11111

        # Second entry is new
        assert logs[1]['user_id'] == 22222
        assert logs[1]['operation'] == "storage_upload"
        assert logs[1]['provider'] == 'runpod'

    def test_audit_log_operation_failure(self, security_manager, tmp_path, monkeypatch):
        """audit_log_operation should log failed operations."""
        # Create temporary audit log directory
        audit_dir = tmp_path / "data" / "logs"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_log_file = audit_dir / "cloud_audit.json"

        # Patch the audit log file path
        import src.cloud.security as security_module
        original_path = security_module.Path
        monkeypatch.setattr(security_module, 'Path', lambda x: audit_log_file if 'cloud_audit.json' in x else original_path(x))

        # Log failed operation
        security_manager.audit_log_operation(
            user_id=12345,
            operation="runpod_training",
            resource="pod-failed",
            success=False,
            error="GPU not available"
        )

        # Read log entry
        with open(audit_log_file, 'r') as f:
            logs = json.load(f)

        entry = logs[0]
        assert entry['success'] is False
        assert entry['error'] == "GPU not available"
