"""
Script executor for running generated Python scripts in sandboxed environments.

This module provides secure execution of generated scripts with resource
monitoring, timeout control, and comprehensive cleanup.
"""

import asyncio
import json
import os
import resource
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import psutil

from src.execution.config import SandboxConfig, ScriptResult
from src.generators.validator import ScriptValidator
from src.utils.exceptions import (
    ExecutionError,
    SecurityViolationError,
    ResourceLimitError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessManager:
    """Manages subprocess lifecycle and monitoring."""

    @staticmethod
    def set_resource_limits(config: SandboxConfig) -> None:
        """Set resource limits for the subprocess."""
        try:
            # Set memory limit more safely
            if config.memory_limit:
                # Convert MB to bytes
                memory_bytes = config.memory_limit * 1024 * 1024
                # Get current limits
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)

                # Use the smaller of requested limit and system hard limit
                if hard_limit != resource.RLIM_INFINITY:
                    memory_bytes = min(memory_bytes, hard_limit)

                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # Set CPU time limit
            if config.cpu_limit:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(config.cpu_limit), int(config.cpu_limit))
                )

            # Disable core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        except Exception as e:
            logger.warning(f"Failed to set resource limits: {e}")

    @staticmethod
    def create_sandbox_env() -> Dict[str, str]:
        """Create isolated environment variables."""
        # Start with current environment to preserve Python package access
        env = dict(os.environ)

        # Override security-sensitive variables
        env.update({
            'HOME': '/tmp',
            'TMPDIR': '/tmp',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
        })

        # Remove potentially dangerous variables
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH']
        for var in dangerous_vars:
            env.pop(var, None)

        return env

    @staticmethod
    async def monitor_process(process: asyncio.subprocess.Process) -> Dict[str, Any]:
        """Monitor process resource usage."""
        try:
            if process.pid:
                proc = psutil.Process(process.pid)
                memory_info = proc.memory_info()

                return {
                    'memory_mb': memory_info.rss / (1024 * 1024),
                    'cpu_percent': proc.cpu_percent(),
                    'num_threads': proc.num_threads() if hasattr(proc, 'num_threads') else 1
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return {'memory_mb': 0, 'cpu_percent': 0, 'num_threads': 0}


class ScriptExecutor:
    """Executes Python scripts in sandboxed environments."""

    def __init__(self) -> None:
        """Initialize the script executor."""
        self.validator = ScriptValidator()
        self.process_manager = ProcessManager()
        logger.info("Script executor initialized")

    async def run_sandboxed(
        self,
        script: str,
        data: Dict[str, Any],
        config: SandboxConfig
    ) -> ScriptResult:
        """
        Execute script in sandboxed environment.

        Args:
            script: Python script to execute
            data: Input data to pass to script via stdin
            config: Sandbox configuration

        Returns:
            ScriptResult with execution details

        Raises:
            SecurityViolationError: If script fails security validation
            ResourceLimitError: If execution exceeds resource limits
            ExecutionError: If execution fails for other reasons
        """
        logger.info("Starting sandboxed script execution")

        # Validate script security
        await self._validate_script_security(script)

        # Execute with monitoring
        start_time = time.time()

        try:
            async with self._create_execution_context(config) as temp_dir:
                result = await self._execute_script_in_sandbox(
                    script, data, config, temp_dir, start_time
                )

                logger.info(f"Script execution completed in {result.execution_time:.2f}s")
                return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Script execution timed out after {execution_time:.2f}s"
            logger.error(error_msg)
            raise ResourceLimitError(error_msg, "timeout", str(config.timeout))

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Script execution failed: {str(e)}"
            logger.error(error_msg)
            raise ExecutionError(error_msg)

    async def _validate_script_security(self, script: str) -> None:
        """Validate script for security violations."""
        is_valid, violations = self.validator.validate_script(script)

        if not is_valid:
            error_msg = f"Script failed security validation: {violations}"
            logger.error(error_msg)
            raise SecurityViolationError(error_msg, violations)

    @asynccontextmanager
    async def _create_execution_context(self, config: SandboxConfig):
        """Create and manage execution context with cleanup."""
        temp_dir = None

        try:
            # Create temporary directory
            if config.temp_dir:
                config.temp_dir.mkdir(parents=True, exist_ok=True)
                temp_dir = config.temp_dir
            else:
                temp_dir = Path(tempfile.mkdtemp(prefix="script_exec_"))

            logger.debug(f"Created execution context in {temp_dir}")
            yield temp_dir

        finally:
            # Cleanup temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up execution context: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    async def _execute_script_in_sandbox(
        self,
        script: str,
        data: Dict[str, Any],
        config: SandboxConfig,
        temp_dir: Path,
        start_time: float
    ) -> ScriptResult:
        """Execute script in isolated subprocess."""
        # Write script to temporary file
        script_file = temp_dir / "script.py"
        script_file.write_text(script)

        # Prepare input data
        input_json = json.dumps(data).encode()

        # Create subprocess with isolation
        # Prepare subprocess kwargs based on sandbox configuration
        subprocess_kwargs = {
            'stdin': asyncio.subprocess.PIPE,
            'stdout': asyncio.subprocess.PIPE,
            'stderr': asyncio.subprocess.PIPE,
            'cwd': str(temp_dir),
        }

        # Apply sandbox isolation when enabled (default: True)
        if config.sandbox_enabled:
            subprocess_kwargs['env'] = self.process_manager.create_sandbox_env()
            subprocess_kwargs['preexec_fn'] = lambda: self.process_manager.set_resource_limits(config)
            logger.debug("Sandbox enabled: resource limits and env isolation active")
        else:
            logger.warning("Sandbox DISABLED: running without resource limits (not recommended)")

        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_file),
            **subprocess_kwargs
        )

        try:
            # Execute with timeout and monitoring
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_json),
                timeout=config.timeout
            )

            # Get resource usage
            resource_info = await self.process_manager.monitor_process(process)
            execution_time = time.time() - start_time

            # Create result
            if process.returncode == 0:
                return ScriptResult.success_result(
                    output=stdout.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    memory_usage=int(resource_info.get('memory_mb', 0)),
                    script=script
                )
            else:
                return ScriptResult.failure_result(
                    error=stderr.decode('utf-8', errors='replace'),
                    execution_time=execution_time,
                    memory_usage=int(resource_info.get('memory_mb', 0)),
                    exit_code=process.returncode,
                    script=script
                )

        except asyncio.TimeoutError:
            # Kill the process if it's still running
            if process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.warning(f"Failed to kill timed out process: {e}")

            raise  # Re-raise the timeout error

    async def validate_script_before_execution(self, script: str) -> bool:
        """
        Validate script without executing it.

        Args:
            script: Python script to validate

        Returns:
            True if script is safe to execute
        """
        try:
            await self._validate_script_security(script)
            return True
        except SecurityViolationError:
            return False

    def get_executor_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics and configuration.

        Returns:
            Dictionary with executor statistics
        """
        return {
            'validator_patterns': len(self.validator.forbidden_patterns),
            'allowed_imports': len(self.validator.allowed_imports),
            'python_executable': sys.executable,
            'platform': sys.platform,
        }


class SecurityValidator:
    """Additional security validation for execution context."""

    @staticmethod
    def validate_execution_environment() -> bool:
        """Validate that execution environment is safe."""
        # Check if running as root (dangerous)
        if os.getuid() == 0:
            logger.warning("Running as root - potential security risk")
            return False

        # Check available resource limits
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if soft_limit == resource.RLIM_INFINITY:
                logger.warning("No memory limits set - potential resource exhaustion risk")

        except Exception as e:
            logger.warning(f"Could not check resource limits: {e}")

        return True

    @staticmethod
    def sanitize_error_output(error_output: str) -> str:
        """Sanitize error output to remove sensitive information."""
        # Remove absolute paths
        import re
        sanitized = re.sub(r'/[^/\s]+/[^/\s]+/[^\s]*', '<path>', error_output)

        # Remove potential user information
        sanitized = re.sub(r'User:\s+\w+', 'User: <user>', sanitized)

        # Limit length to prevent log flooding
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "... [truncated]"

        return sanitized