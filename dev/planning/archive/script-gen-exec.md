# Script Generator and Executor Implementation Plan

## Executive Summary

This document outlines the design and implementation plan for a secure, sandboxed script generation and execution system that processes statistical and machine learning operations from user requests while maintaining strict security boundaries.

## System Architecture

### Overview
The script generator and executor form a critical security boundary in the statistical modeling agent. They enable dynamic Python script creation and safe execution for user-driven analyses while preventing malicious code execution.

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│ Script Generator │───▶│    Executor     │
│  (TaskDefinition) │    │   (Templates)   │    │  (Sandboxed)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Template System │    │ Security Layer  │
                       │ (Jinja2 + AST)  │    │ (Resource Limits)│
                       └─────────────────┘    └─────────────────┘
```

## Component Specifications

### 1. Script Generator (`src/generators/script_generator.py`)

#### Purpose
Converts TaskDefinition objects into secure, executable Python scripts using template-based generation.

#### Core Classes

**ScriptGenerator**
```python
class ScriptGenerator:
    async def generate(self, task: TaskDefinition, data_context: dict) -> str
    def _get_template(self, operation: str) -> str
    def _validate_parameters(self, params: dict) -> dict
    def _sanitize_column_names(self, columns: list[str]) -> list[str]
```

**TemplateRegistry**
```python
class TemplateRegistry:
    def __init__(self, template_dir: Path)
    def get_template(self, category: str, operation: str) -> jinja2.Template
    def _cache_template(self, path: str) -> jinja2.Template
```

**ScriptValidator**
```python
class ScriptValidator:
    def validate_syntax(self, script: str) -> bool
    def check_forbidden_patterns(self, script: str) -> list[str]
    def validate_imports(self, ast_tree: ast.AST) -> bool
```

#### Template Categories

**Statistical Operations (`templates/stats/`)**
- `descriptive.j2` - Mean, median, mode, std dev calculations
- `correlation.j2` - Correlation matrix and analysis
- `hypothesis.j2` - T-tests, chi-square, ANOVA
- `distribution.j2` - Histograms, normality tests
- `regression.j2` - Linear/polynomial regression analysis

**Machine Learning (`templates/ml/`)**
- `train_classifier.j2` - Classification model training
- `train_regressor.j2` - Regression model training
- `predict.j2` - Model prediction and scoring
- `feature_selection.j2` - Feature importance analysis
- `model_evaluation.j2` - Performance metrics

**Utilities (`templates/utils/`)**
- `data_validation.j2` - Input data validation
- `error_handling.j2` - Exception management
- `json_io.j2` - Input/output handling
- `memory_utils.j2` - Memory optimization

#### Security Features

**Input Sanitization**
```python
def sanitize_column_name(name: str) -> str:
    """Sanitize column names for safe script generation."""
    # Only allow alphanumeric and underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    return sanitized
```

**Forbidden Pattern Detection**
```python
FORBIDDEN_PATTERNS = [
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'open\s*\(',
    r'subprocess',
    r'os\.system',
    r'\.\./',  # Path traversal
    r'file\s*\(',
    r'input\s*\(',
]
```

### 2. Executor (`src/execution/executor.py`)

#### Purpose
Safely executes generated Python scripts in isolated environments with resource controls and comprehensive error handling.

#### Core Classes

**ScriptExecutor**
```python
class ScriptExecutor:
    async def run_sandboxed(self, script: str, data: dict, config: SandboxConfig) -> ScriptResult
    def _create_sandbox_env(self) -> dict[str, str]
    def _monitor_resources(self, process: asyncio.subprocess.Process) -> None
    async def _cleanup_resources(self, temp_dir: Path, process: Optional[asyncio.subprocess.Process]) -> None
```

**SandboxConfig**
```python
@dataclass
class SandboxConfig:
    timeout: int = 30  # seconds
    memory_limit: int = 2 * 1024 * 1024 * 1024  # 2GB
    cpu_limit: Optional[float] = None
    allow_network: bool = False
    temp_dir: Optional[Path] = None
```

**ScriptResult**
```python
@dataclass
class ScriptResult:
    success: bool
    output: Optional[str]
    error: Optional[str]
    execution_time: float
    memory_usage: int
    exit_code: int
    script_hash: str  # For caching
```

#### Security Implementation

**Resource Limits**
```python
import resource

def set_resource_limits(memory_limit: int) -> None:
    """Set memory limits for script execution."""
    resource.setrlimit(
        resource.RLIMIT_AS,
        (memory_limit, -1)
    )
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (30, -1)  # 30 seconds CPU time
    )
```

**Sandbox Environment**
```python
def create_sandbox_env() -> dict[str, str]:
    """Create isolated environment for script execution."""
    return {
        'PYTHONPATH': '',  # No external modules
        'HOME': '/tmp',
        'PATH': '/usr/bin:/bin',  # Minimal PATH
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUNBUFFERED': '1',
    }
```

**Process Isolation**
```python
async def run_isolated_script(script_path: Path, data: dict, config: SandboxConfig) -> ScriptResult:
    """Execute script in completely isolated subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_file = Path(tmpdir) / "script.py"
        script_file.write_text(script)

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
            env=create_sandbox_env(),
            preexec_fn=lambda: set_resource_limits(config.memory_limit)
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(data).encode()),
                timeout=config.timeout
            )
            return ScriptResult(
                success=proc.returncode == 0,
                output=stdout.decode() if proc.returncode == 0 else None,
                error=stderr.decode() if proc.returncode != 0 else None,
                execution_time=time.time() - start_time,
                exit_code=proc.returncode or 0
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ExecutionError("Script execution timed out")
```

## Template Design Patterns

### Base Template Structure

All templates follow this standard pattern:

```python
#!/usr/bin/env python3
"""
Generated script for {{ operation }} operation.
Generated at: {{ timestamp }}
"""

import json
import sys
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

def main() -> None:
    """Main execution function."""
    try:
        # Read input data from stdin
        input_data = json.loads(sys.stdin.read())
        df = pd.DataFrame(input_data['dataframe'])
        params = input_data.get('parameters', {})

        # Validate input data
        validate_input(df, params)

        # Execute operation
        result = execute_{{ operation }}(df, params)

        # Output results as JSON
        output = {
            'success': True,
            'result': result,
            'metadata': {
                'operation': '{{ operation }}',
                'timestamp': '{{ timestamp }}',
                'rows_processed': len(df)
            }
        }

        print(json.dumps(output, default=str))

    except Exception as e:
        error_output = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)

def validate_input(df: pd.DataFrame, params: dict) -> None:
    """Validate input data and parameters."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    # Operation-specific validation
    {{ validation_code }}

def execute_{{ operation }}(df: pd.DataFrame, params: dict) -> dict:
    """Execute the {{ operation }} operation."""
    {{ operation_code }}

if __name__ == "__main__":
    main()
```

### Example: Correlation Analysis Template

```python
def execute_correlation_analysis(df: pd.DataFrame, params: dict) -> dict:
    """Execute correlation analysis."""
    columns = params.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
    method = params.get('method', 'pearson')

    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)

    # Find strongest correlations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_pairs = corr_matrix.where(mask).stack().reset_index()
    corr_pairs.columns = ['var1', 'var2', 'correlation']
    corr_pairs = corr_pairs.sort_values('correlation', key=abs, ascending=False)

    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strongest_correlations': corr_pairs.head(10).to_dict('records'),
        'summary_stats': {
            'mean_correlation': corr_matrix.values[mask].mean(),
            'max_correlation': corr_matrix.values[mask].max(),
            'min_correlation': corr_matrix.values[mask].min(),
        }
    }
```

## Integration Patterns

### Orchestrator Integration

```python
# In orchestrator.py
async def execute_task(self, task: TaskDefinition, data: pd.DataFrame) -> dict:
    """Execute a task using script generation and execution."""

    # Generate script
    script = await self.script_generator.generate(task, {'dataframe': data.to_dict()})

    # Execute script
    result = await self.executor.run_sandboxed(
        script,
        {'dataframe': data.to_dict(), 'parameters': task.parameters},
        SandboxConfig(timeout=30, memory_limit=2*1024*1024*1024)
    )

    if not result.success:
        raise ExecutionError(f"Script execution failed: {result.error}")

    # Parse and return results
    return json.loads(result.output)
```

### Error Handling Chain

```python
class ScriptGenerationError(AgentError):
    """Script generation failures."""
    pass

class ExecutionError(AgentError):
    """Script execution failures."""
    pass

class SecurityViolationError(ExecutionError):
    """Security constraint violations."""
    pass

class ResourceLimitError(ExecutionError):
    """Resource limit exceeded."""
    pass
```

## Security Considerations

### Threat Model

**Threats Mitigated:**
- Malicious code injection through user input
- Resource exhaustion attacks (CPU, memory)
- File system access outside sandbox
- Network access and data exfiltration
- Import of unauthorized modules

**Attack Vectors:**
- Column names containing Python code
- Statistical parameters with executable content
- Large datasets causing memory exhaustion
- Infinite loops in user-provided formulas

### Security Controls

**Input Validation:**
- AST parsing to detect dangerous constructs
- Regular expression filtering for known patterns
- Parameter type validation and bounds checking
- Column name sanitization

**Runtime Protection:**
- Subprocess isolation with restricted environment
- Resource limits (memory, CPU, time)
- No network access
- Temporary directory confinement
- Process monitoring and forced termination

**Output Sanitization:**
- JSON-only output format
- Error message filtering
- No direct file output capabilities

## Testing Strategy

### Unit Tests

**Script Generator Tests:**
```python
class TestScriptGenerator:
    def test_generate_correlation_script(self):
        """Test correlation script generation."""
        task = TaskDefinition(
            task_type="stats",
            operation="correlation_analysis",
            parameters={"columns": ["age", "income"]},
            user_id=123,
            conversation_id="test"
        )

        script = self.generator.generate(task, {})

        # Validate script syntax
        assert self.validator.validate_syntax(script)

        # Check for required components
        assert "correlation_analysis" in script
        assert "age" in script
        assert "income" in script

        # Ensure no forbidden patterns
        violations = self.validator.check_forbidden_patterns(script)
        assert len(violations) == 0
```

**Executor Tests:**
```python
class TestExecutor:
    async def test_successful_execution(self):
        """Test successful script execution."""
        script = """
import json
import sys
result = {'sum': 2 + 2}
print(json.dumps(result))
"""

        result = await self.executor.run_sandboxed(script, {}, SandboxConfig())

        assert result.success
        assert json.loads(result.output)['sum'] == 4
        assert result.execution_time < 1.0

    async def test_timeout_handling(self):
        """Test timeout enforcement."""
        script = """
import time
time.sleep(60)  # Longer than timeout
"""

        with pytest.raises(ExecutionError, match="timed out"):
            await self.executor.run_sandboxed(script, {}, SandboxConfig(timeout=1))
```

### Security Tests

```python
class TestSecurity:
    def test_forbidden_import_detection(self):
        """Test detection of dangerous imports."""
        dangerous_script = """
import os
os.system('rm -rf /')
"""

        violations = self.validator.check_forbidden_patterns(dangerous_script)
        assert len(violations) > 0
        assert any('os.system' in v for v in violations)

    def test_code_injection_prevention(self):
        """Test prevention of code injection through parameters."""
        malicious_params = {
            'columns': ['age"; os.system("rm -rf /"); "']
        }

        sanitized = self.generator._sanitize_parameters(malicious_params)
        assert 'os.system' not in str(sanitized)
        assert 'rm -rf' not in str(sanitized)
```

### Integration Tests

```python
class TestFullPipeline:
    async def test_stats_workflow_end_to_end(self):
        """Test complete statistics workflow."""
        # Create task
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"statistics": ["mean", "std"]},
            user_id=123,
            conversation_id="test"
        )

        # Sample data
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [30000, 45000, 55000, 65000]
        })

        # Execute through orchestrator
        result = await self.orchestrator.execute_task(task, data)

        # Validate results
        assert 'mean' in result
        assert 'std' in result
        assert result['mean']['age'] == 32.5
```

## Performance Optimizations

### Template Caching

```python
from functools import lru_cache

class TemplateRegistry:
    @lru_cache(maxsize=128)
    def _load_template(self, template_path: str) -> jinja2.Template:
        """Cache compiled templates."""
        with open(template_path) as f:
            return self.env.from_string(f.read())
```

### Script Validation Caching

```python
@lru_cache(maxsize=256)
def validate_script_cached(script_hash: str, script: str) -> tuple[bool, list[str]]:
    """Cache validation results for identical scripts."""
    syntax_valid = validate_syntax(script)
    violations = check_forbidden_patterns(script)
    return syntax_valid, violations
```

### Resource Monitoring

```python
import psutil

async def monitor_process_resources(process: asyncio.subprocess.Process) -> dict:
    """Monitor process resource usage."""
    try:
        proc = psutil.Process(process.pid)
        return {
            'cpu_percent': proc.cpu_percent(),
            'memory_mb': proc.memory_info().rss / (1024 * 1024),
            'num_threads': proc.num_threads()
        }
    except psutil.NoSuchProcess:
        return {}
```

## Configuration

### Environment Variables

```bash
# Execution limits
SCRIPT_TIMEOUT=30
SCRIPT_MEMORY_LIMIT=2147483648  # 2GB
SCRIPT_CPU_LIMIT=30

# Template settings
TEMPLATE_CACHE_SIZE=128
VALIDATION_CACHE_SIZE=256

# Security
ENABLE_NETWORK_ACCESS=false
ALLOWED_IMPORTS=json,sys,pandas,numpy,scipy,sklearn
```

### Configuration Schema

```yaml
# config/execution.yaml
execution:
  sandbox:
    timeout: 30
    memory_limit: 2147483648
    cpu_limit: 30
    allow_network: false
    temp_dir: null

  security:
    forbidden_patterns:
      - '__import__'
      - 'exec\s*\('
      - 'eval\s*\('
      - 'open\s*\('
      - 'subprocess'
      - 'os\.system'

    allowed_imports:
      - json
      - sys
      - pandas
      - numpy
      - scipy
      - sklearn
      - matplotlib
      - seaborn

templates:
  cache_size: 128
  base_dir: "templates"
  categories:
    - stats
    - ml
    - utils
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `ScriptGenerator` base class
- [ ] Implement `TemplateRegistry` with caching
- [ ] Build `ScriptValidator` with AST analysis
- [ ] Create `ScriptExecutor` with subprocess isolation
- [ ] Implement `SandboxConfig` configuration system

### Phase 2: Template System
- [ ] Create base template structure
- [ ] Implement descriptive statistics templates
- [ ] Build correlation analysis templates
- [ ] Add machine learning training templates
- [ ] Create prediction and scoring templates

### Phase 3: Security Framework
- [ ] Implement forbidden pattern detection
- [ ] Add input sanitization for all parameters
- [ ] Create resource limit enforcement
- [ ] Build process monitoring and cleanup
- [ ] Add comprehensive error handling

### Phase 4: Integration
- [ ] Integrate with orchestrator workflow
- [ ] Add async support for Telegram bot
- [ ] Implement result processing pipeline
- [ ] Create configuration management system
- [ ] Add logging and monitoring hooks

### Phase 5: Testing
- [ ] Write unit tests for all components
- [ ] Create security validation test suite
- [ ] Build integration tests for full pipeline
- [ ] Add performance and stress tests
- [ ] Implement automated security scanning

### Phase 6: Documentation
- [ ] Complete API documentation
- [ ] Write security guidelines
- [ ] Create deployment instructions
- [ ] Document troubleshooting procedures
- [ ] Add monitoring and alerting guides

## Deployment Considerations

### Production Requirements
- Containerization for additional isolation
- Resource monitoring and alerting
- Log aggregation and security analysis
- Regular security audits and updates
- Backup and recovery procedures

### Monitoring Points
- Script generation success/failure rates
- Execution time distributions
- Memory usage patterns
- Security violation attempts
- Error rate analysis

This implementation plan provides a comprehensive foundation for building a secure, scalable script generation and execution system that meets the requirements of the statistical modeling agent while maintaining strict security boundaries.