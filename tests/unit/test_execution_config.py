"""
Test suite for execution configuration system.
"""

import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Test first, then implement
def test_sandbox_config_creation():
    """Test SandboxConfig creation with defaults."""
    from src.execution.config import SandboxConfig

    config = SandboxConfig()

    assert config.timeout == 30
    assert config.memory_limit == 2 * 1024 * 1024 * 1024  # 2GB
    assert config.cpu_limit is None
    assert config.allow_network is False
    assert config.temp_dir is None


def test_sandbox_config_custom_values():
    """Test SandboxConfig with custom values."""
    from src.execution.config import SandboxConfig

    config = SandboxConfig(
        timeout=60,
        memory_limit=1024 * 1024 * 1024,  # 1GB
        cpu_limit=45.0,
        allow_network=True,
        temp_dir=Path("/tmp/custom")
    )

    assert config.timeout == 60
    assert config.memory_limit == 1024 * 1024 * 1024
    assert config.cpu_limit == 45.0
    assert config.allow_network is True
    assert config.temp_dir == Path("/tmp/custom")


def test_sandbox_config_validation():
    """Test SandboxConfig validation."""
    from src.execution.config import SandboxConfig

    # Test invalid timeout
    with pytest.raises(ValueError, match="Timeout must be positive"):
        SandboxConfig(timeout=0)

    # Test invalid memory limit
    with pytest.raises(ValueError, match="Memory limit must be positive"):
        SandboxConfig(memory_limit=0)


def test_script_result_creation():
    """Test ScriptResult creation."""
    from src.execution.config import ScriptResult

    result = ScriptResult(
        success=True,
        output='{"result": "test"}',
        error=None,
        execution_time=1.5,
        memory_usage=1024,
        exit_code=0,
        script_hash="abc123"
    )

    assert result.success is True
    assert result.output == '{"result": "test"}'
    assert result.error is None
    assert result.execution_time == 1.5
    assert result.memory_usage == 1024
    assert result.exit_code == 0
    assert result.script_hash == "abc123"


def test_script_result_failure():
    """Test ScriptResult for failed execution."""
    from src.execution.config import ScriptResult

    result = ScriptResult(
        success=False,
        output=None,
        error="Runtime error occurred",
        execution_time=0.5,
        memory_usage=512,
        exit_code=1,
        script_hash="def456"
    )

    assert result.success is False
    assert result.output is None
    assert result.error == "Runtime error occurred"
    assert result.exit_code == 1