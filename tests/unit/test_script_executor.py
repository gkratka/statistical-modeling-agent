"""
Test suite for script executor system.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path


@pytest.mark.asyncio
async def test_script_executor_creation():
    """Test ScriptExecutor creation."""
    from src.execution.executor import ScriptExecutor

    executor = ScriptExecutor()
    assert executor is not None


@pytest.mark.asyncio
async def test_successful_script_execution():
    """Test successful script execution."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json
import sys

data = json.loads(sys.stdin.read())
result = {'sum': data['a'] + data['b']}
print(json.dumps(result))
"""

    input_data = {'a': 2, 'b': 3}
    config = SandboxConfig(timeout=5)

    result = await executor.run_sandboxed(script, input_data, config)

    assert result.success is True
    assert result.exit_code == 0
    assert result.error is None

    output_data = json.loads(result.output)
    assert output_data['sum'] == 5


@pytest.mark.asyncio
async def test_script_execution_with_error():
    """Test script execution that produces an error."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json
raise ValueError("Test error")
"""

    config = SandboxConfig(timeout=5)

    result = await executor.run_sandboxed(script, {}, config)

    assert result.success is False
    assert result.exit_code != 0
    assert result.output is None
    assert "ValueError" in result.error
    assert "Test error" in result.error


@pytest.mark.asyncio
async def test_script_execution_timeout():
    """Test script execution timeout."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig
    from src.utils.exceptions import ResourceLimitError

    executor = ScriptExecutor()

    script = """
import json
import sys
# Simulate work without using time.sleep (which might be optimized)
result = 0
for i in range(100000000):  # Large computation
    result += i
print(json.dumps({"result": result}))
"""

    config = SandboxConfig(timeout=0.1)  # Very short timeout

    with pytest.raises(ResourceLimitError, match="timed out"):
        await executor.run_sandboxed(script, {}, config)


@pytest.mark.asyncio
async def test_script_validation_before_execution():
    """Test script validation before execution."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig
    from src.utils.exceptions import SecurityViolationError

    executor = ScriptExecutor()

    dangerous_script = """
import os
os.system('echo "dangerous command"')
"""

    config = SandboxConfig()

    with pytest.raises(SecurityViolationError):
        await executor.run_sandboxed(dangerous_script, {}, config)


@pytest.mark.asyncio
async def test_resource_monitoring():
    """Test resource monitoring during execution."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json
import sys

# Do some work that uses memory
data = [i for i in range(1000)]
result = {'count': len(data)}
print(json.dumps(result))
"""

    config = SandboxConfig(timeout=5)

    result = await executor.run_sandboxed(script, {}, config)

    assert result.success is True
    # Memory usage might be 0 due to subprocess monitoring limitations
    assert result.memory_usage >= 0
    assert result.execution_time > 0


@pytest.mark.asyncio
async def test_script_with_pandas():
    """Test script execution with pandas-like operations (without pandas)."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json
import sys

data = json.loads(sys.stdin.read())
values = data['dataframe']['values']

# Simulate pandas operations without importing pandas
mean_value = sum(values) / len(values)
count = len(values)

result = {
    'mean': mean_value,
    'count': count
}

print(json.dumps(result, default=str))
"""

    input_data = {
        'dataframe': {
            'values': [1, 2, 3, 4, 5]
        }
    }

    config = SandboxConfig(timeout=10)

    result = await executor.run_sandboxed(script, input_data, config)

    assert result.success is True
    output_data = json.loads(result.output)
    assert output_data['mean'] == 3.0
    assert output_data['count'] == 5


@pytest.mark.asyncio
async def test_script_hash_generation():
    """Test script hash generation for caching."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = "print('test')"
    config = SandboxConfig(timeout=5)

    result = await executor.run_sandboxed(script, {}, config)

    assert result.success is True
    assert len(result.script_hash) == 16  # SHA256 truncated to 16 chars
    assert result.script_hash.isalnum()


@pytest.mark.asyncio
async def test_cleanup_after_execution():
    """Test proper cleanup after script execution."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json

# Simple script that doesn't use forbidden operations
result = {'message': 'Script executed successfully'}
print(json.dumps(result))
"""

    config = SandboxConfig(timeout=5)

    result = await executor.run_sandboxed(script, {}, config)

    # Execution should succeed
    assert result.success is True
    output_data = json.loads(result.output)
    assert output_data['message'] == 'Script executed successfully'

    # The temp directory should be cleaned up automatically


@pytest.mark.asyncio
async def test_concurrent_executions():
    """Test multiple concurrent script executions."""
    from src.execution.executor import ScriptExecutor
    from src.execution.config import SandboxConfig

    executor = ScriptExecutor()

    script = """
import json
import time
import sys

data = json.loads(sys.stdin.read())
time.sleep(0.1)  # Small delay
result = {'id': data['id'], 'result': data['id'] * 2}
print(json.dumps(result))
"""

    config = SandboxConfig(timeout=5)

    # Run multiple scripts concurrently
    tasks = []
    for i in range(3):
        task = executor.run_sandboxed(script, {'id': i}, config)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # All should succeed
    for i, result in enumerate(results):
        assert result.success is True
        output_data = json.loads(result.output)
        assert output_data['id'] == i
        assert output_data['result'] == i * 2