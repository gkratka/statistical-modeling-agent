"""
Configuration system for script execution.

This module defines configuration classes and data structures
for the sandboxed execution environment.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SandboxConfig:
    """Configuration for sandboxed script execution.

    Security Features:
        - sandbox_enabled: Controls whether resource limits and env isolation are applied
        - memory_limit: Maximum memory in MB (default: 2048)
        - cpu_limit: Maximum CPU time in seconds
        - timeout: Maximum execution time in seconds (default: 30)
    """

    timeout: int = 30  # seconds
    memory_limit: Optional[int] = 2048  # MB, None to disable
    cpu_limit: Optional[float] = None
    allow_network: bool = False
    temp_dir: Optional[Path] = None
    sandbox_enabled: bool = field(default_factory=lambda: os.getenv('SANDBOX_DISABLED', '').lower() != 'true')

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")

        if self.memory_limit is not None and self.memory_limit <= 0:
            raise ValueError("Memory limit must be positive")

        if self.cpu_limit is not None and self.cpu_limit <= 0:
            raise ValueError("CPU limit must be positive")


@dataclass
class ScriptResult:
    """Result of script execution."""

    success: bool
    output: Optional[str]
    error: Optional[str]
    execution_time: float
    memory_usage: int
    exit_code: int
    script_hash: str

    @classmethod
    def create_hash(cls, script: str) -> str:
        """Create hash for script caching."""
        return hashlib.sha256(script.encode()).hexdigest()[:16]

    @classmethod
    def success_result(
        cls,
        output: str,
        execution_time: float,
        memory_usage: int,
        script: str
    ) -> "ScriptResult":
        """Create successful result."""
        return cls(
            success=True,
            output=output,
            error=None,
            execution_time=execution_time,
            memory_usage=memory_usage,
            exit_code=0,
            script_hash=cls.create_hash(script)
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        execution_time: float,
        memory_usage: int,
        exit_code: int,
        script: str
    ) -> "ScriptResult":
        """Create failure result."""
        return cls(
            success=False,
            output=None,
            error=error,
            execution_time=execution_time,
            memory_usage=memory_usage,
            exit_code=exit_code,
            script_hash=cls.create_hash(script)
        )