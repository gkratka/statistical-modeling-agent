# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Telegram bot that performs statistical analysis and machine learning tasks through natural language conversations. Users can upload data, request analyses, train models, and get predictions.

## Core Principle
Safety first: All user-provided code is sandboxed, all inputs are validated, all outputs are sanitized.

## Commands

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your TELEGRAM_BOT_TOKEN and ANTHROPIC_API_KEY
```

### Running the Application
```bash
# Start the bot
python src/bot/telegram_bot.py

# Run with specific config
python src/bot/telegram_bot.py --config config/config.yaml
```

### Testing
```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_parser.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test function
pytest tests/unit/test_parser.py::TestParser::test_parse_basic_stats_request
```

### Code Quality
```bash
# Format code with Black
black src/ tests/ --line-length 88

# Check typing with mypy
mypy src/ --strict

# Lint with flake8
flake8 src/ tests/ --max-line-length 88

# Run all checks
black src/ tests/ --check && mypy src/ --strict && flake8 src/ tests/
```

## Project Architecture

### System Flow
The system follows a linear pipeline architecture with clear separation of concerns:

```
1. telegram_bot.py receives user message
   ↓
2. handlers.py routes to appropriate handler
   ↓
3. parser.py converts natural language to TaskDefinition object
   ↓
4. orchestrator.py determines task type and routes to engine
   ↓
5. stats_engine.py OR ml_engine.py processes the task
   ↓
6. script_generator.py creates executable Python script
   ↓
7. executor.py runs script in sandboxed environment
   ↓
8. result_processor.py formats output for user
   ↓
9. telegram_bot.py sends response back to user
```

### Module Interactions

#### Data Flow Objects
```python
# Task Definition (parser.py → orchestrator.py)
@dataclass
class TaskDefinition:
    task_type: Literal["stats", "ml_train", "ml_score"]
    operation: str  # "descriptive_stats", "correlation", "train_model", etc.
    parameters: dict[str, Any]
    data_source: Optional[DataSource]
    user_id: int
    conversation_id: str

# Script Result (executor.py → result_processor.py)
@dataclass
class ScriptResult:
    success: bool
    output: Optional[str]
    error: Optional[str]
    execution_time: float
    memory_usage: int
```

#### State Management
The `state_manager.py` maintains conversation context across multiple interactions:
```python
# Conversation state is stored per user_id + conversation_id
state = {
    "current_workflow": "ml_training",
    "step": "awaiting_target_selection",
    "data": uploaded_dataframe,
    "model_id": None,
    "history": [...]
}
```

### Critical Integration Points

1. **Bot → Parser Interface**
   - Bot passes raw message text and any attachments
   - Parser returns structured TaskDefinition or raises ParseError
   - Parser must handle multi-modal input (text + files)

2. **Orchestrator → Engine Interface**
   - Orchestrator validates task requirements before routing
   - Engines are stateless - receive TaskDefinition, return results
   - Orchestrator manages retry logic and fallbacks

3. **Engine → Script Generator Interface**
   - Engines prepare script parameters dict
   - Script generator uses Jinja2 templates from `templates/`
   - Generated scripts must be self-contained and import-safe

4. **Script Generator → Executor Interface**
   - Scripts are validated for dangerous operations before execution
   - Executor provides data via stdin as JSON
   - Scripts output results as JSON to stdout

## Coding Standards

### Type Annotations
Every function must have complete type annotations:
```python
from typing import Optional, Union, Literal
from dataclasses import dataclass
import pandas as pd

def calculate_statistics(
    data: pd.DataFrame,
    columns: list[str],
    operations: list[Literal["mean", "median", "std"]],
    group_by: Optional[str] = None
) -> dict[str, Union[float, dict[str, float]]]:
    """Calculate statistics with optional grouping."""
    pass
```

### Error Hierarchy
```python
# src/utils/exceptions.py
class AgentError(Exception):
    """Base exception for all agent errors"""
    pass

class ValidationError(AgentError):
    """Input validation failures"""
    pass

class ParseError(AgentError):
    """Natural language parsing failures"""
    pass

class ExecutionError(AgentError):
    """Script execution failures"""
    pass

class DataError(ValidationError):
    """Data-specific validation failures"""
    pass
```

### Import Structure
```python
# Standard library - alphabetical
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party - grouped by functionality
import numpy as np
import pandas as pd
from scipy import stats

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler

# Local - absolute imports only
from src.core.parser import Parser
from src.engines.stats_engine import StatsEngine
from src.utils.exceptions import AgentError
```

### Async Patterns
All Telegram handlers and I/O operations must be async:
```python
async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle incoming message asynchronously."""
    # Parse message asynchronously
    task = await parser.parse_async(update.message.text)

    # Execute with timeout
    result = await asyncio.wait_for(
        executor.run_async(script, data),
        timeout=30.0
    )

    # Send response
    await update.message.reply_text(result.formatted_output)
```

## Key Workflows

### Statistics Request Flow
```python
# 1. Handler receives stats request
@message_handler(filters=Filters.text)
async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 2. Parse request
    task = parser.parse_stats_request(update.message.text)
    # Example task: TaskDefinition(
    #     task_type="stats",
    #     operation="correlation",
    #     parameters={"columns": ["age", "income"]}
    # )

    # 3. Validate data availability
    data = await get_user_data(update.effective_user.id)
    if not data:
        await update.message.reply_text("Please upload data first")
        return

    # 4. Route through orchestrator
    result = await orchestrator.execute_task(task, data)

    # 5. Send formatted response
    await send_formatted_result(update, result)
```

### ML Training Multi-Step Conversation
```python
# State machine for ML workflow
class MLTrainingState:
    AWAITING_DATA = "awaiting_data"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    TRAINING = "training"
    COMPLETE = "complete"

# Handler manages state transitions
async def handle_ml_workflow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = context.user_data.get("ml_state", MLTrainingState.AWAITING_DATA)

    match state:
        case MLTrainingState.AWAITING_DATA:
            # Process uploaded file
            data = await process_upload(update.message.document)
            context.user_data["training_data"] = data
            context.user_data["ml_state"] = MLTrainingState.SELECTING_TARGET

            # Show column list for target selection
            columns = data.columns.tolist()
            await update.message.reply_text(
                f"Select target variable:\n" + "\n".join(f"{i+1}. {col}" for i, col in enumerate(columns))
            )

        case MLTrainingState.SELECTING_TARGET:
            # Process target selection
            target = parse_column_selection(update.message.text)
            context.user_data["target"] = target
            context.user_data["ml_state"] = MLTrainingState.SELECTING_FEATURES
            # Continue flow...
```

### Script Generation Templates
```python
# src/generators/templates/stats_template.py
CORRELATION_TEMPLATE = """
import pandas as pd
import numpy as np
import json
import sys

# Read data from stdin
data = json.loads(sys.stdin.read())
df = pd.DataFrame(data['dataframe'])

# Calculate correlations
columns = {columns}
correlation_matrix = df[columns].corr()

# Format results
results = {{
    'correlation_matrix': correlation_matrix.to_dict(),
    'summary': {{
        'strongest_correlation': find_strongest_correlation(correlation_matrix),
        'weakest_correlation': find_weakest_correlation(correlation_matrix)
    }}
}}

print(json.dumps(results, default=str))
"""

# Generator uses template
def generate_correlation_script(columns: list[str]) -> str:
    return CORRELATION_TEMPLATE.format(
        columns=repr(columns)
    )
```

### Sandboxed Execution
```python
# src/execution/executor.py
async def run_sandboxed(
    script: str,
    data: dict[str, Any],
    timeout: int = 30
) -> ScriptResult:
    """Execute script in isolated environment."""

    # 1. Validate script safety
    if not validate_script_safety(script):
        raise ExecutionError("Script contains unsafe operations")

    # 2. Prepare sandbox environment
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.py"
        script_path.write_text(script)

        # 3. Execute with resource limits
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
            env={"PYTHONPATH": ""},  # Isolated environment
        )

        # 4. Send data and get result
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=json.dumps(data).encode()),
            timeout=timeout
        )

        # 5. Parse and return result
        return ScriptResult(
            success=proc.returncode == 0,
            output=stdout.decode() if proc.returncode == 0 else None,
            error=stderr.decode() if proc.returncode != 0 else None,
            execution_time=time.time() - start_time
        )
```

## Testing Requirements

### Unit Test Structure
```python
# tests/unit/test_stats_engine.py
import pytest
import pandas as pd
from src.engines.stats_engine import StatsEngine, StatsTask

class TestStatsEngine:
    """Test statistical operations engine."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Provide sample data for tests."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [30000, 45000, 55000, 65000, 80000],
            'score': [75, 82, 88, 91, 95]
        })

    def test_calculate_descriptive_stats(self, sample_data):
        """Test basic descriptive statistics calculation."""
        engine = StatsEngine()
        task = StatsTask(
            operation="descriptive",
            columns=["age", "income"],
            statistics=["mean", "median", "std"]
        )

        result = engine.execute(task, sample_data)

        assert "age" in result
        assert "mean" in result["age"]
        assert result["age"]["mean"] == 35.0

    @pytest.mark.parametrize("column,expected_mean", [
        ("age", 35.0),
        ("income", 55000.0),
        ("score", 86.2),
    ])
    def test_mean_calculation(self, sample_data, column, expected_mean):
        """Test mean calculation for different columns."""
        engine = StatsEngine()
        result = engine.calculate_mean(sample_data[column])
        assert result == pytest.approx(expected_mean, rel=1e-2)
```

### Integration Test Pattern
```python
# tests/integration/test_full_workflow.py
@pytest.mark.asyncio
class TestFullWorkflow:
    """Test complete user workflows end-to-end."""

    async def test_stats_workflow(self, mock_bot, sample_csv):
        """Test complete statistics workflow."""
        # 1. User uploads file
        upload_response = await mock_bot.send_document(sample_csv)
        assert "Data uploaded successfully" in upload_response

        # 2. User requests statistics
        stats_response = await mock_bot.send_message(
            "Calculate mean and correlation for age and income"
        )

        # 3. Verify response contains expected elements
        assert "Mean Statistics" in stats_response
        assert "Correlation Matrix" in stats_response
        assert "age" in stats_response
        assert "income" in stats_response

    async def test_ml_training_workflow(self, mock_bot, training_data):
        """Test ML model training workflow."""
        # 1. Upload training data
        await mock_bot.send_document(training_data)

        # 2. Start ML workflow
        await mock_bot.send_message("/train")

        # 3. Select target variable
        response = await mock_bot.send_message("3")  # Select 3rd column
        assert "Select features" in response

        # 4. Select features
        response = await mock_bot.send_message("1,2")  # Columns 1 and 2
        assert "Model type" in response

        # 5. Confirm model type
        response = await mock_bot.send_message("neural network")
        assert "Training started" in response

        # 6. Wait for training completion
        final_response = await mock_bot.wait_for_response(timeout=60)
        assert "Model trained successfully" in final_response
        assert "Accuracy:" in final_response
```

### Test Coverage Requirements
- Each public function must have at least one test
- Error cases must be explicitly tested with `pytest.raises`
- Edge cases (empty data, single row, missing values) must be covered
- Mock external dependencies (Telegram API, file system)
- Integration tests must cover all user workflows defined in PRD

### Test Fixtures
```python
# tests/conftest.py
import pytest
import pandas as pd
from unittest.mock import AsyncMock

@pytest.fixture
def mock_telegram_update():
    """Create mock Telegram update object."""
    update = AsyncMock()
    update.effective_user.id = 12345
    update.message.text = ""
    update.message.reply_text = AsyncMock()
    return update

@pytest.fixture
def sample_dataframe():
    """Standard test dataframe."""
    return pd.DataFrame({
        'numeric_1': [1, 2, 3, 4, 5],
        'numeric_2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 1]
    })
```

## Configuration

### Environment Variables (.env)
```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional
LOG_LEVEL=INFO
MAX_FILE_SIZE=104857600  # 100MB
EXECUTION_TIMEOUT=30
```

### Configuration File (config/config.yaml)
Reference the template in PRD section 10 for complete configuration structure. Key settings:
- `execution.sandbox`: Use "docker" for production, "subprocess" for development
- `data.max_file_size`: Enforce file size limits
- `execution.timeout`: Maximum script execution time
- `models.save_dir`: Where trained models are persisted

## Security Patterns

### Script Validation
```python
def validate_script_safety(script: str) -> bool:
    """Check script for dangerous operations."""
    forbidden_patterns = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'open\s*\(',
        r'subprocess',
        r'os\.system',
        r'\.\./',  # Path traversal
    ]

    for pattern in forbidden_patterns:
        if re.search(pattern, script):
            return False

    return True
```

### Input Sanitization
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

## Performance Optimization

### Async I/O for File Operations
```python
async def load_large_csv(path: Path, chunk_size: int = 10000):
    """Stream large CSV files in chunks."""
    chunks = []
    async with aiofiles.open(path, mode='r') as file:
        reader = csv.DictReader(await file.readlines())
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                chunks.append(pd.DataFrame(chunk))
                chunk = []
        if chunk:
            chunks.append(pd.DataFrame(chunk))

    return pd.concat(chunks, ignore_index=True)
```

### Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_model_metadata(model_id: str) -> dict:
    """Cache model metadata to avoid repeated disk reads."""
    return load_model_metadata(model_id)
```

### Resource Limits
```python
# Set memory limits for script execution
resource.setrlimit(
    resource.RLIMIT_AS,
    (2 * 1024 * 1024 * 1024, -1)  # 2GB limit
)
```

## Common Pitfalls to Avoid
- NEVER concatenate user input directly into scripts
- NEVER store API keys in code (use .env)
- NEVER use globals for state (use state_manager)
- ALWAYS validate dataframe columns exist before accessing
- ALWAYS handle empty/null data cases

## Quick Decision Guide
- New Telegram command? → Add to handlers.py
- New statistical operation? → Add to stats_engine.py
- New ML model type? → Add to ml_engine.py
- New data format? → Add to data_loader.py
- Bug in script generation? → Check templates/ first