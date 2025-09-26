# CLAUDE.md - Statistical Modeling Agent

## Project Overview
You are helping build a Telegram-based statistical modeling agent that performs data analysis and machine learning tasks through conversational interactions. The agent should process user requests, generate Python scripts, execute them safely, and return results via Telegram.

## Project Architecture (What Claude needs to know)

### Core System Design
```
User → Telegram Bot → Request Parser → Task Orchestrator → Engine (Stats/ML) → Script Generator → Executor → Result Processor → User
```

### Module Responsibilities
- **telegram_bot.py**: Handle all Telegram interactions, NEVER put business logic here
- **parser.py**: Convert natural language to structured task definitions
- **orchestrator.py**: Route tasks to appropriate engines, maintain conversation state
- **stats_engine.py**: Handle all statistical computations
- **ml_engine.py**: Manage ML model training/scoring pipelines
- **script_generator.py**: Create executable Python scripts from task definitions
- **executor.py**: Run scripts in sandboxed environment with timeout protection
- **result_processor.py**: Format outputs for user consumption

### Key Design Principles
1. **Separation of Concerns**: Each module has ONE clear responsibility
2. **Stateless Operations**: Don't store state in modules, use state_manager.py
3. **Error Propagation**: Bubble errors up with context, handle at bot level
4. **Script Safety**: ALL generated scripts must be validated before execution

## Coding Standards (Rules Claude must follow)

### Python Style
- Use Python 3.10+ features (type hints, match statements, union types)
- Follow PEP 8 strictly (max line length: 88 for Black formatter)
- Always use type hints:
  ```python
  def process_data(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
  ```

### Import Organization
```python
# Standard library
import os
from typing import Optional, Dict, List

# Third-party
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import ContextTypes

# Local modules
from src.core import parser
from src.utils import validators
```

### Error Handling Pattern
```python
class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class DataValidationError(AgentError):
    """Raised when data validation fails"""
    pass

# Always use specific exceptions
try:
    result = process_data(df)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    return ErrorResponse(message=str(e), code="VALIDATION_ERROR")
```

### Documentation Standards
```python
def train_model(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    model_type: str = "neural_network"
) -> ModelResult:
    """
    Train a machine learning model on the provided dataset.

    Args:
        data: Input dataframe with training data
        target_column: Name of the target variable column
        feature_columns: List of feature column names
        model_type: Type of model to train (default: "neural_network")

    Returns:
        ModelResult: Object containing trained model and metrics

    Raises:
        DataValidationError: If data validation fails
        ModelTrainingError: If model training fails
    """
```

## Key Workflows (Implementation patterns)

### 1. Basic Statistics Workflow
```python
# When implementing stats workflow:
async def handle_stats_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Parse user message
    task = parser.parse_stats_request(update.message.text)

    # 2. Load and validate data
    data = await data_loader.load_from_telegram(update.message.document)
    validators.validate_data_schema(data, task.required_columns)

    # 3. Generate script
    script = script_generator.generate_stats_script(task, data.columns)

    # 4. Execute safely
    result = await executor.run_sandboxed(script, data, timeout=30)

    # 5. Process and return
    formatted = result_processor.format_stats_result(result)
    await update.message.reply_text(formatted)
```

### 2. ML Training Workflow
```python
# When implementing ML workflow:
async def handle_ml_training(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Multi-step conversation
    state = context.user_data.get('ml_state', MLState())

    # 2. Collect requirements step by step
    if not state.has_data:
        # Request data upload
        pass
    elif not state.has_target:
        # Ask for target variable
        pass
    elif not state.has_features:
        # Ask for features
        pass
    else:
        # 3. Train model
        model = await ml_engine.train_model(
            data=state.data,
            target=state.target,
            features=state.features
        )

        # 4. Save model
        model_id = await model_store.save(model, user_id=update.effective_user.id)

        # 5. Return results with option to score
        await update.message.reply_text(
            f"Model trained! Accuracy: {model.accuracy:.2%}\n"
            f"Model ID: {model_id}\n"
            "Send /score to make predictions on new data"
        )
```

### 3. Script Generation Pattern
```python
# Always use templates for script generation:
STATS_TEMPLATE = '''
import pandas as pd
import numpy as np

# Load data
df = pd.DataFrame(data)

# Perform analysis
{analysis_code}

# Return results
results = {result_extraction}
print(json.dumps(results))
'''

def generate_stats_script(task: StatsTask) -> str:
    analysis_code = build_analysis_code(task)
    result_extraction = build_result_extraction(task)
    return STATS_TEMPLATE.format(
        analysis_code=analysis_code,
        result_extraction=result_extraction
    )
```

## Testing Requirements (What tests Claude must write)

### Unit Test Pattern
```python
# tests/unit/test_parser.py
import pytest
from src.core.parser import parse_stats_request

class TestParser:
    def test_parse_basic_stats_request(self):
        """Test parsing of basic statistics request"""
        message = "Calculate mean and std for column age"
        result = parse_stats_request(message)

        assert result.operation == "descriptive_stats"
        assert result.statistics == ["mean", "std"]
        assert result.column == "age"

    def test_parse_invalid_request_raises_error(self):
        """Test that invalid requests raise appropriate errors"""
        with pytest.raises(ParseError):
            parse_stats_request("invalid gibberish")
```

### Integration Test Pattern
```python
# tests/integration/test_workflow.py
@pytest.mark.asyncio
async def test_end_to_end_stats_workflow(mock_telegram_bot, sample_data):
    """Test complete statistics workflow from request to response"""
    # Arrange
    update = create_mock_update("Calculate correlation matrix", sample_data)

    # Act
    response = await handle_stats_request(update, create_context())

    # Assert
    assert "Correlation Matrix" in response
    assert all(col in response for col in sample_data.columns)
```

### Test Coverage Requirements
- Minimum 80% code coverage
- All public functions must have tests
- All error paths must be tested
- Integration tests for each major workflow

## File Naming Conventions
- Snake_case for Python files: `script_generator.py`
- PascalCase for classes: `class DataValidator`
- UPPER_CASE for constants: `MAX_FILE_SIZE = 100_000_000`
- Descriptive names over abbreviations: `calculate_statistics` not `calc_stats`

## Security Considerations
1. **Never execute raw user input** - always validate and sanitize
2. **Use subprocess with timeout** for script execution
3. **Validate file types and sizes** before processing
4. **Sanitize all outputs** before sending to Telegram
5. **Never log sensitive data** (API keys, user data)

## Performance Guidelines
- Use async/await for all I/O operations
- Implement caching for repeated calculations
- Stream large datasets instead of loading entirely
- Set reasonable timeouts (30s for stats, 300s for ML)

## When Implementing New Features
1. Start with the test - write what you expect
2. Implement the simplest solution that passes
3. Refactor for clarity and performance
4. Document all public interfaces
5. Add integration test for the complete workflow

## Common Pitfalls to Avoid
- Don't put business logic in the Telegram bot handlers
- Don't generate scripts with string concatenation (use templates)
- Don't catch generic Exception (be specific)
- Don't store state in module-level variables
- Don't execute untrusted code without sandboxing

## Example Implementation Request
When asked to implement a feature, Claude should:
1. Clarify requirements if ambiguous
2. Write tests first (TDD approach)
3. Implement with proper error handling
4. Include comprehensive docstrings
5. Follow the established patterns above

Remember: This is a production system handling user data. Prioritize safety, clarity, and maintainability over cleverness.