# ML Training Workflow State Handlers - Comprehensive Implementation Plan

## Executive Summary

This document provides a complete implementation plan for ML training workflow state handlers in the Statistical Modeling Agent Telegram bot. The bot currently successfully initiates ML training workflows but lacks handlers to process subsequent user interactions (column selections, model type choices). This plan addresses that gap with a complete state management system.

## Current State Analysis

### What Works ✅
- Bot successfully initiates ML workflow when user sends "Train a model to predict house prices"
- Workflow detection correctly identifies invalid target columns
- Column selection prompt displays correctly
- StateManager infrastructure exists with workflow state tracking
- WorkflowState enum defined with all necessary states

### What's Missing ❌
- Message routing based on active workflow state
- Handlers for each workflow state (SELECTING_TARGET, SELECTING_FEATURES, CONFIRMING_MODEL, TRAINING)
- Column selection parsing (numbers and names)
- State transition logic
- Training execution integration
- Error handling for invalid inputs
- Workflow cancellation command

### Root Cause
After workflow initiation (handlers.py:203-223), the handler returns control to the bot. The next message from the user routes back to `message_handler()` which attempts to parse it as a new request instead of continuing the active workflow.

## Architecture Design

### 1. Message Flow (Current vs Proposed)

**Current Flow (Broken):**
```
User sends "5" → message_handler() → parser.parse() → Error: "Request not understood"
```

**Proposed Flow (Fixed):**
```
User sends "5" → message_handler() → Check workflow state → Route to WorkflowRouter
→ handle_target_selection("5") → Validate → Store → Transition to SELECTING_FEATURES
→ Send feature prompt
```

### 2. Integration Point

**Location:** `src/bot/handlers.py`, `message_handler()` function (around line 140)

**Modification:**
```python
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message_text = update.message.text

    # NEW: Check for active workflow BEFORE parsing
    from src.core.state_manager import StateManager
    from src.bot.workflow_handlers import WorkflowRouter

    state_manager = StateManager()
    session = await state_manager.get_or_create_session(
        user_id,
        f"chat_{update.effective_chat.id}"
    )

    # If user has active workflow, route to workflow handler
    if session.workflow_state is not None:
        workflow_router = WorkflowRouter(state_manager)
        return await workflow_router.handle(update, context, session)

    # Otherwise, continue with normal parsing flow
    # ... existing code ...
```

## Component Design

### 3. New Module: `src/bot/workflow_handlers.py`

#### Class: WorkflowRouter

**Purpose:** Route messages to appropriate state handlers based on active workflow state

**Methods:**
```python
class WorkflowRouter:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.logger = get_logger(__name__)

    async def handle(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session: Session
    ) -> None:
        """Route message to appropriate workflow state handler."""

        # Check for cancel command
        if update.message.text.lower() in ['/cancel', 'cancel']:
            return await self.cancel_workflow(update, session)

        # Route based on workflow state
        match session.workflow_state:
            case WorkflowState.SELECTING_TARGET:
                return await self.handle_target_selection(update, context, session)
            case WorkflowState.SELECTING_FEATURES:
                return await self.handle_feature_selection(update, context, session)
            case WorkflowState.CONFIRMING_MODEL:
                return await self.handle_model_confirmation(update, context, session)
            case WorkflowState.TRAINING:
                # Training state is typically non-interactive
                return await self.handle_training_status(update, context, session)
            case _:
                # Unknown state - clear and restart
                await self.state_manager.clear_workflow(session)
                await update.message.reply_text(
                    "Workflow state error. Please start again.",
                    parse_mode="Markdown"
                )
```

#### State Handler: Target Selection

```python
async def handle_target_selection(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session
) -> None:
    """Handle target column selection."""
    user_input = update.message.text.strip()

    # Get dataframe from session
    dataframe = await self.state_manager.get_data(session)
    columns = dataframe.columns.tolist()

    # Parse column selection (number or name)
    try:
        selected_column = parse_column_selection(user_input, columns)
    except ValueError as e:
        # Invalid selection - show error and re-prompt
        error_message = (
            f"❌ Invalid selection: {str(e)}\n\n"
            f"Please select a column by number (1-{len(columns)}) or name.\n"
            f"Example: '5' or 'price'"
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")
        return

    # Validate column exists (should always pass after parse, but double-check)
    if selected_column not in columns:
        await update.message.reply_text(
            f"❌ Column '{selected_column}' not found in dataset.",
            parse_mode="Markdown"
        )
        return

    # Store target column in session
    await self.state_manager.update_workflow_data(
        session,
        {"target_column": selected_column}
    )

    # Transition to feature selection
    await self.state_manager.update_workflow_state(
        session,
        WorkflowState.SELECTING_FEATURES
    )

    # Prepare feature selection prompt (exclude target column)
    available_features = [col for col in columns if col != selected_column]

    from src.bot.response_builder import ResponseBuilder
    response_builder = ResponseBuilder()

    feature_prompt = response_builder.format_feature_selection(
        available_features,
        selected_target=selected_column
    )

    await update.message.reply_text(feature_prompt, parse_mode="Markdown")

    self.logger.info(
        f"User {session.user_id} selected target: {selected_column}, "
        f"transitioned to SELECTING_FEATURES"
    )
```

#### State Handler: Feature Selection

```python
async def handle_feature_selection(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session
) -> None:
    """Handle feature column selection."""
    user_input = update.message.text.strip()

    # Get dataframe and target column
    dataframe = await self.state_manager.get_data(session)
    workflow_data = await self.state_manager.get_workflow_data(session)
    target_column = workflow_data.get("target_column")

    # Available features (exclude target)
    available_features = [
        col for col in dataframe.columns.tolist()
        if col != target_column
    ]

    # Parse feature selection (supports multiple formats)
    try:
        selected_features = parse_feature_selection(
            user_input,
            available_features,
            target_column
        )
    except ValueError as e:
        error_message = (
            f"❌ Invalid selection: {str(e)}\n\n"
            f"Examples:\n"
            f"• '1,2,3' - Select columns 1, 2, and 3\n"
            f"• 'age,income,sqft' - Select by name\n"
            f"• '1-5' - Select range of columns\n"
            f"• 'all' - Select all available features"
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")
        return

    # Validate at least one feature selected
    if not selected_features:
        await update.message.reply_text(
            "❌ Please select at least one feature column.",
            parse_mode="Markdown"
        )
        return

    # Store features in session
    await self.state_manager.update_workflow_data(
        session,
        {"feature_columns": selected_features}
    )

    # Transition to model confirmation
    await self.state_manager.update_workflow_state(
        session,
        WorkflowState.CONFIRMING_MODEL
    )

    # Show model type selection prompt
    from src.bot.response_builder import ResponseBuilder
    response_builder = ResponseBuilder()

    model_prompt = response_builder.format_model_type_selection(
        target_column=target_column,
        feature_count=len(selected_features)
    )

    await update.message.reply_text(model_prompt, parse_mode="Markdown")

    self.logger.info(
        f"User {session.user_id} selected features: {selected_features}, "
        f"transitioned to CONFIRMING_MODEL"
    )
```

#### State Handler: Model Confirmation

```python
async def handle_model_confirmation(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session
) -> None:
    """Handle model type confirmation."""
    user_input = update.message.text.strip().lower()

    # Map user input to model types
    model_type_map = {
        'linear': 'linear_regression',
        'linear regression': 'linear_regression',
        '1': 'linear_regression',
        'random': 'random_forest',
        'random forest': 'random_forest',
        'forest': 'random_forest',
        '2': 'random_forest',
        'neural': 'neural_network',
        'neural network': 'neural_network',
        'nn': 'neural_network',
        '3': 'neural_network',
        'auto': 'auto',
        'automatic': 'auto',
        '4': 'auto'
    }

    model_type = model_type_map.get(user_input)

    if not model_type:
        error_message = (
            f"❌ Invalid model type: '{user_input}'\n\n"
            f"Please select:\n"
            f"1. Linear Regression\n"
            f"2. Random Forest\n"
            f"3. Neural Network\n"
            f"4. Auto (best model automatically selected)"
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")
        return

    # Store model type in session
    await self.state_manager.update_workflow_data(
        session,
        {"model_type": model_type}
    )

    # Transition to training state
    await self.state_manager.update_workflow_state(
        session,
        WorkflowState.TRAINING
    )

    # Show training started message
    await update.message.reply_text(
        "🚀 **Training Started**\n\nPlease wait while the model is being trained...",
        parse_mode="Markdown"
    )

    # Execute training
    await self.execute_training(update, context, session)
```

#### Training Execution

```python
async def execute_training(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: Session
) -> None:
    """Execute ML model training."""

    # Get all workflow data
    dataframe = await self.state_manager.get_data(session)
    workflow_data = await self.state_manager.get_workflow_data(session)

    target_column = workflow_data.get("target_column")
    feature_columns = workflow_data.get("feature_columns")
    model_type = workflow_data.get("model_type")

    # Create TaskDefinition
    from src.core.parser import TaskDefinition

    task = TaskDefinition(
        task_type="ml_train",
        operation="train_model",
        parameters={
            "target_column": target_column,
            "feature_columns": feature_columns,
            "model_type": model_type
        },
        data_source=None,  # Data already in session
        user_id=session.user_id,
        conversation_id=session.conversation_id
    )

    # Execute through orchestrator
    from src.core.orchestrator import TaskOrchestrator
    orchestrator = TaskOrchestrator()

    try:
        result = await orchestrator.execute_task(task, dataframe)

        # Format and send results
        from src.bot.result_formatter import format_ml_training_result
        formatted_result = format_ml_training_result(result)

        await update.message.reply_text(
            formatted_result,
            parse_mode="Markdown"
        )

        # Transition to COMPLETE
        await self.state_manager.update_workflow_state(
            session,
            WorkflowState.COMPLETE
        )

        # Clear workflow after successful completion
        await self.state_manager.clear_workflow(session)

        self.logger.info(
            f"User {session.user_id} completed ML training workflow successfully"
        )

    except Exception as e:
        self.logger.error(f"Training execution failed: {str(e)}", exc_info=True)

        error_message = (
            f"❌ **Training Failed**\n\n"
            f"Error: {str(e)}\n\n"
            f"The workflow has been cancelled. Please try again."
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")

        # Clear workflow on error
        await self.state_manager.clear_workflow(session)
```

#### Workflow Cancellation

```python
async def cancel_workflow(
    self,
    update: Update,
    session: Session
) -> None:
    """Cancel active workflow."""

    workflow_type = session.workflow_type
    current_state = session.workflow_state

    # Clear workflow
    await self.state_manager.clear_workflow(session)

    # Send confirmation
    await update.message.reply_text(
        f"❌ **Workflow Cancelled**\n\n"
        f"Your {workflow_type.value if workflow_type else 'workflow'} has been cancelled.\n"
        f"Send a new request to start over.",
        parse_mode="Markdown"
    )

    self.logger.info(
        f"User {session.user_id} cancelled workflow at state {current_state}"
    )
```

### 4. Utility Functions

#### Column Selection Parser

```python
def parse_column_selection(user_input: str, columns: list[str]) -> str:
    """
    Parse column selection from user input.

    Supports:
    - Number: "5" → columns[4]
    - Name: "price" → "price"

    Args:
        user_input: User's input text
        columns: List of available column names

    Returns:
        Selected column name

    Raises:
        ValueError: If selection is invalid
    """
    user_input = user_input.strip()

    # Try parsing as number first
    if user_input.isdigit():
        index = int(user_input) - 1  # 1-based to 0-based

        if index < 0 or index >= len(columns):
            raise ValueError(
                f"Number out of range. Please select 1-{len(columns)}."
            )

        return columns[index]

    # Try matching by name (case-insensitive)
    normalized_input = user_input.lower()

    for col in columns:
        if col.lower() == normalized_input:
            return col

    # No match found
    raise ValueError(
        f"Column '{user_input}' not found. "
        f"Available columns: {', '.join(columns)}"
    )
```

#### Feature Selection Parser

```python
def parse_feature_selection(
    user_input: str,
    available_features: list[str],
    target_column: str
) -> list[str]:
    """
    Parse feature column selection from user input.

    Supports:
    - Numbers: "1,2,3"
    - Names: "age,income,sqft"
    - Range: "1-5"
    - All: "all"
    - All except: "all except price" (for reference)

    Args:
        user_input: User's input text
        available_features: List of available feature column names (excluding target)
        target_column: Target column name (for reference in error messages)

    Returns:
        List of selected feature column names

    Raises:
        ValueError: If selection is invalid
    """
    user_input = user_input.strip().lower()

    # Handle "all" keyword
    if user_input == 'all':
        return available_features

    selected_features = []

    # Split by comma
    parts = [p.strip() for p in user_input.split(',')]

    for part in parts:
        # Check for range (e.g., "1-5")
        if '-' in part and part.replace('-', '').isdigit():
            range_parts = part.split('-')
            if len(range_parts) == 2:
                start = int(range_parts[0]) - 1  # 1-based to 0-based
                end = int(range_parts[1])  # end is inclusive in 1-based

                if start < 0 or end > len(available_features):
                    raise ValueError(
                        f"Range {part} out of bounds. "
                        f"Valid range: 1-{len(available_features)}"
                    )

                selected_features.extend(available_features[start:end])
                continue

        # Try parsing as number
        if part.isdigit():
            index = int(part) - 1

            if index < 0 or index >= len(available_features):
                raise ValueError(
                    f"Number {part} out of range. "
                    f"Please select 1-{len(available_features)}."
                )

            selected_features.append(available_features[index])
            continue

        # Try matching by name
        matched = False
        for feature in available_features:
            if feature.lower() == part:
                selected_features.append(feature)
                matched = True
                break

        if not matched:
            raise ValueError(
                f"Feature '{part}' not found or is the target column. "
                f"Available features: {', '.join(available_features)}"
            )

    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for feature in selected_features:
        if feature not in seen:
            seen.add(feature)
            unique_features.append(feature)

    return unique_features
```

### 5. Response Builder Extensions

Add to `src/bot/response_builder.py`:

```python
def format_feature_selection(
    self,
    available_features: list[str],
    selected_target: str
) -> str:
    """Format feature selection prompt."""

    numbered_features = '\n'.join(
        f"{i+1}. {feature}"
        for i, feature in enumerate(available_features)
    )

    return (
        f"**Step 2/4: Select Feature Columns**\n\n"
        f"Target: `{selected_target}`\n\n"
        f"Available features:\n{numbered_features}\n\n"
        f"**How to select:**\n"
        f"• Single: `1` or `age`\n"
        f"• Multiple: `1,2,3` or `age,income,sqft`\n"
        f"• Range: `1-5`\n"
        f"• All: `all`\n\n"
        f"Type `/cancel` to cancel workflow."
    )

def format_model_type_selection(
    self,
    target_column: str,
    feature_count: int
) -> str:
    """Format model type selection prompt."""

    return (
        f"**Step 3/4: Select Model Type**\n\n"
        f"Target: `{target_column}`\n"
        f"Features: {feature_count} columns\n\n"
        f"**Available models:**\n"
        f"1. **Linear Regression** - Fast, interpretable\n"
        f"2. **Random Forest** - Robust, handles non-linearity\n"
        f"3. **Neural Network** - Complex patterns, requires more data\n"
        f"4. **Auto** - Best model selected automatically\n\n"
        f"Select by number (1-4) or name.\n\n"
        f"Type `/cancel` to cancel workflow."
    )
```

### 6. Cancel Command Registration

Add to `src/bot/telegram_bot.py`:

```python
# In _setup_handlers method
self.application.add_handler(CommandHandler("cancel", cancel_handler))
```

Create cancel handler in `src/bot/handlers.py`:

```python
@telegram_handler
@log_user_action("Cancel workflow")
async def cancel_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /cancel command."""
    user_id = update.effective_user.id

    from src.core.state_manager import StateManager

    state_manager = StateManager()
    session = await state_manager.get_or_create_session(
        user_id,
        f"chat_{update.effective_chat.id}"
    )

    if session.workflow_state is None:
        await update.message.reply_text(
            "No active workflow to cancel.",
            parse_mode="Markdown"
        )
        return

    from src.bot.workflow_handlers import WorkflowRouter
    workflow_router = WorkflowRouter(state_manager)
    await workflow_router.cancel_workflow(update, session)
```

## Testing Strategy

### 7. Integration Tests

Create `tests/integration/test_workflow_state_handlers.py`:

```python
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock

from src.bot.workflow_handlers import WorkflowRouter, parse_column_selection, parse_feature_selection
from src.core.state_manager import StateManager, WorkflowState, WorkflowType

@pytest.mark.asyncio
class TestWorkflowStateHandlers:
    """Test workflow state handlers."""

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'age': [25, 30, 35],
            'income': [30000, 45000, 55000],
            'sqft': [1000, 1500, 2000],
            'price': [200000, 300000, 400000]
        })

    @pytest.fixture
    async def mock_session(self, sample_dataframe):
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        return session

    async def test_target_selection_by_number(self, mock_session):
        """Test target selection using number input."""
        # Setup
        update = MagicMock()
        update.message.text = "4"  # Select "price" (4th column)
        update.message.reply_text = AsyncMock()

        state_manager = StateManager()
        router = WorkflowRouter(state_manager)

        # Execute
        await router.handle_target_selection(update, None, mock_session)

        # Verify
        workflow_data = await state_manager.get_workflow_data(mock_session)
        assert workflow_data["target_column"] == "price"
        assert mock_session.workflow_state == WorkflowState.SELECTING_FEATURES

    async def test_target_selection_by_name(self, mock_session):
        """Test target selection using column name."""
        update = MagicMock()
        update.message.text = "income"
        update.message.reply_text = AsyncMock()

        state_manager = StateManager()
        router = WorkflowRouter(state_manager)

        await router.handle_target_selection(update, None, mock_session)

        workflow_data = await state_manager.get_workflow_data(mock_session)
        assert workflow_data["target_column"] == "income"

    async def test_invalid_target_selection(self, mock_session):
        """Test invalid target selection shows error."""
        update = MagicMock()
        update.message.text = "99"  # Invalid number
        update.message.reply_text = AsyncMock()

        state_manager = StateManager()
        router = WorkflowRouter(state_manager)

        await router.handle_target_selection(update, None, mock_session)

        # Should show error message
        update.message.reply_text.assert_called_once()
        error_msg = update.message.reply_text.call_args[0][0]
        assert "Invalid selection" in error_msg

        # State should remain unchanged
        assert mock_session.workflow_state == WorkflowState.SELECTING_TARGET

    async def test_feature_selection_multiple_columns(self, mock_session):
        """Test feature selection with multiple columns."""
        # Set target first
        state_manager = StateManager()
        await state_manager.update_workflow_data(mock_session, {"target_column": "price"})
        await state_manager.update_workflow_state(mock_session, WorkflowState.SELECTING_FEATURES)

        update = MagicMock()
        update.message.text = "1,2,3"  # age, income, sqft
        update.message.reply_text = AsyncMock()

        router = WorkflowRouter(state_manager)
        await router.handle_feature_selection(update, None, mock_session)

        workflow_data = await state_manager.get_workflow_data(mock_session)
        assert set(workflow_data["feature_columns"]) == {"age", "income", "sqft"}
        assert mock_session.workflow_state == WorkflowState.CONFIRMING_MODEL

    async def test_feature_selection_range(self, mock_session):
        """Test feature selection with range notation."""
        state_manager = StateManager()
        await state_manager.update_workflow_data(mock_session, {"target_column": "price"})
        await state_manager.update_workflow_state(mock_session, WorkflowState.SELECTING_FEATURES)

        update = MagicMock()
        update.message.text = "1-3"  # age, income, sqft
        update.message.reply_text = AsyncMock()

        router = WorkflowRouter(state_manager)
        await router.handle_feature_selection(update, None, mock_session)

        workflow_data = await state_manager.get_workflow_data(mock_session)
        assert len(workflow_data["feature_columns"]) == 3

    async def test_model_type_selection(self, mock_session):
        """Test model type selection."""
        state_manager = StateManager()
        await state_manager.update_workflow_state(mock_session, WorkflowState.CONFIRMING_MODEL)

        update = MagicMock()
        update.message.text = "2"  # Random Forest
        update.message.reply_text = AsyncMock()

        router = WorkflowRouter(state_manager)
        await router.handle_model_confirmation(update, None, mock_session)

        workflow_data = await state_manager.get_workflow_data(mock_session)
        assert workflow_data["model_type"] == "random_forest"
        assert mock_session.workflow_state == WorkflowState.TRAINING

    async def test_workflow_cancellation(self, mock_session):
        """Test workflow cancellation."""
        update = MagicMock()
        update.message.text = "/cancel"
        update.message.reply_text = AsyncMock()

        state_manager = StateManager()
        router = WorkflowRouter(state_manager)

        await router.cancel_workflow(update, mock_session)

        # Workflow should be cleared
        assert mock_session.workflow_state is None
        assert mock_session.workflow_type is None

class TestColumnParsing:
    """Test column parsing utility functions."""

    def test_parse_column_by_number(self):
        columns = ['age', 'income', 'price']
        assert parse_column_selection("2", columns) == "income"

    def test_parse_column_by_name(self):
        columns = ['age', 'income', 'price']
        assert parse_column_selection("price", columns) == "price"

    def test_parse_column_case_insensitive(self):
        columns = ['Age', 'Income', 'Price']
        assert parse_column_selection("age", columns) == "Age"

    def test_parse_column_invalid_number(self):
        columns = ['age', 'income', 'price']
        with pytest.raises(ValueError, match="out of range"):
            parse_column_selection("99", columns)

    def test_parse_features_multiple(self):
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("1,3", features, "price")
        assert result == ['age', 'sqft']

    def test_parse_features_range(self):
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("1-2", features, "price")
        assert result == ['age', 'income']

    def test_parse_features_all(self):
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("all", features, "price")
        assert result == features
```

### 8. End-to-End Test

```python
@pytest.mark.asyncio
async def test_complete_ml_workflow():
    """Test complete ML training workflow from start to finish."""

    # Setup mock bot and data
    update = MagicMock()
    context = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat.id = 67890
    update.message.reply_text = AsyncMock()

    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'income': [30000, 45000, 55000, 65000, 80000],
        'price': [200000, 300000, 350000, 400000, 500000]
    })

    state_manager = StateManager()
    session = await state_manager.get_or_create_session(12345, "chat_67890")
    await state_manager.store_data(session, df)

    # Step 1: Initiate workflow (this already works)
    await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
    assert session.workflow_state == WorkflowState.SELECTING_TARGET

    # Step 2: Select target
    router = WorkflowRouter(state_manager)
    update.message.text = "price"
    await router.handle_target_selection(update, context, session)

    session = await state_manager.get_or_create_session(12345, "chat_67890")
    assert session.workflow_state == WorkflowState.SELECTING_FEATURES

    # Step 3: Select features
    update.message.text = "1,2"  # age, income
    await router.handle_feature_selection(update, context, session)

    session = await state_manager.get_or_create_session(12345, "chat_67890")
    assert session.workflow_state == WorkflowState.CONFIRMING_MODEL

    # Step 4: Select model
    update.message.text = "auto"
    await router.handle_model_confirmation(update, context, session)

    # Should transition to training and execute
    # (Mock orchestrator to avoid actual training in tests)
```

## Error Handling & Edge Cases

### 9. Error Scenarios

| Scenario | Handler Behavior |
|----------|------------------|
| Invalid target number (99) | Show error, re-prompt same state |
| Invalid target name ("xyz") | Show error with available columns |
| Empty feature selection | Show error, require at least one feature |
| Features include target | Automatically exclude target, show warning |
| Invalid model type | Show error with valid options |
| Training execution fails | Show error message, clear workflow |
| User sends /cancel | Clear workflow, show confirmation |
| Session timeout (24h old) | Auto-clear workflow on next message |
| Bot restart during workflow | Workflow persists if state_manager uses persistent storage |

### 10. State Persistence

**Requirement:** Workflows must survive bot restarts

**Solution:** Ensure StateManager uses persistent storage (database or file-based)

```python
# In state_manager.py
class StateManager:
    def __init__(self):
        # Use persistent storage
        self.storage = PersistentSessionStorage()  # Not in-memory

    async def cleanup_old_workflows(self, max_age_hours: int = 24):
        """Clean up workflows older than max_age_hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        await self.storage.delete_sessions_older_than(cutoff_time)
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/bot/workflow_handlers.py`
- [ ] Implement `WorkflowRouter` class
- [ ] Implement `parse_column_selection()` utility
- [ ] Implement `parse_feature_selection()` utility
- [ ] Add workflow routing to `handlers.py` message_handler

### Phase 2: State Handlers
- [ ] Implement `handle_target_selection()`
- [ ] Implement `handle_feature_selection()`
- [ ] Implement `handle_model_confirmation()`
- [ ] Implement `execute_training()`
- [ ] Implement `cancel_workflow()`

### Phase 3: UI/UX
- [ ] Add `format_feature_selection()` to ResponseBuilder
- [ ] Add `format_model_type_selection()` to ResponseBuilder
- [ ] Create `/cancel` command handler
- [ ] Register `/cancel` in telegram_bot.py

### Phase 4: Testing
- [ ] Create `tests/integration/test_workflow_state_handlers.py`
- [ ] Write tests for target selection (valid/invalid)
- [ ] Write tests for feature selection (multiple formats)
- [ ] Write tests for model confirmation
- [ ] Write tests for workflow cancellation
- [ ] Write end-to-end workflow test
- [ ] Test column parsing utilities

### Phase 5: Error Handling
- [ ] Add input validation with helpful errors
- [ ] Implement session timeout cleanup
- [ ] Add logging for all state transitions
- [ ] Test error recovery scenarios

### Phase 6: Documentation
- [ ] Update CLAUDE.md with workflow state handler patterns
- [ ] Add workflow diagram to documentation
- [ ] Document supported input formats
- [ ] Create user guide for ML training workflow

## Success Criteria

### Functional Requirements ✅
- [ ] User can select target column by number or name
- [ ] User can select multiple features using various formats (comma, range, "all")
- [ ] User can select model type by number or name
- [ ] Training executes successfully with selected parameters
- [ ] User can cancel workflow at any state with `/cancel`
- [ ] Invalid inputs show helpful error messages without breaking workflow

### Non-Functional Requirements ✅
- [ ] All state transitions logged for debugging
- [ ] Workflows persist across bot restarts
- [ ] Old workflows auto-cleaned up after 24 hours
- [ ] Integration tests achieve >90% coverage
- [ ] Response time <2 seconds per state transition
- [ ] Error messages guide users toward correct input format

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| StateManager not persistent | Workflows lost on restart | Verify storage backend, add tests |
| Column name parsing ambiguity | Wrong columns selected | Case-insensitive exact match, show confirmation |
| Training execution timeout | User sees "training" forever | Add timeout, show progress updates |
| Concurrent workflows same user | State conflicts | Use conversation_id in session key |
| Memory leak from uncleaned sessions | Performance degradation | Implement periodic cleanup task |

## Future Enhancements

1. **Multi-step feature engineering**: Allow users to create derived features before training
2. **Model comparison**: Automatically train multiple models and show comparison
3. **Hyperparameter tuning**: Let advanced users customize model parameters
4. **Progress updates**: Stream training progress updates to user
5. **Model versioning**: Save and list previously trained models
6. **Prediction workflow**: Add similar state-based workflow for making predictions
7. **Data preprocessing**: Guide users through missing value handling, normalization

## References

- Original bug fix: `claudedocs/BUG_FIX_ML_TRAINING_ERROR.md`
- Test checklist: `claudedocs/ML_WORKFLOW_TEST_CHECKLIST.md`
- Critical steps: `claudedocs/CRITICAL_NEXT_STEPS.md`
- StateManager: `src/core/state_manager.py`
- Current handlers: `src/bot/handlers.py`
- Response formatting: `src/bot/response_builder.py`
- Parser logic: `src/core/parser.py`

---

**Document Version**: 1.0
**Created**: 2025-10-02
**Status**: Implementation Ready
**Next Action**: Begin Phase 1 - Core Infrastructure
