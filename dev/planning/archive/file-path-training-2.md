# Local File Path Training - Phase 2: Deferred Data Loading

**Status**: Planning
**Created**: 2025-10-08
**Purpose**: Enable training on massive datasets (10M+ rows) by deferring data loading until training execution

---

## Problem Statement

Current local file path workflow loads entire dataset into memory during bot interaction to:
1. Display data preview
2. Auto-detect schema
3. Validate columns

**Issue**: Telegram has 4096 character message limit. Large datasets (10M rows) cause:
- Message truncation/failure
- Memory pressure on bot
- Slow response times

---

## Solution Overview

Add **deferred loading mode** where:
1. Bot validates file path only (not data)
2. User provides schema manually (no auto-detection)
3. Data loads from disk only during ML training execution
4. Schema validation happens at training time (delayed error feedback)

---

## Workflow Comparison

### Current: Immediate Loading
```
/train
  ↓
Choose Data Source (Telegram Upload | Local Path)
  ↓
Provide File Path → VALIDATE PATH
  ↓
LOAD ENTIRE DATASET (⚠️ fails for 10M rows)
  ↓
Auto-Detect Schema → Display Preview
  ↓
Confirm Schema
  ↓
Select Model Type
  ↓
Train Model
```

### New: Deferred Loading Option
```
/train
  ↓
Choose Data Source (Telegram Upload | Local Path)
  ↓
Provide File Path → VALIDATE PATH
  ↓
CHOOSE LOAD OPTION ← NEW DECISION POINT
  ├─ Load Now (current workflow)
  │    ↓
  │  Load Data → Auto-Detect → Confirm
  │
  └─ Defer Loading (new workflow)
       ↓
     PROVIDE SCHEMA MANUALLY (text or file upload)
       ↓
     Parse Schema → Validate Format
       ↓
     Skip to Model Selection (no target/feature selection needed)
  ↓
Train Model (data loaded here for deferred path)
```

---

## Technical Design

### 1. New State Machine States

**File**: `src/core/state_manager.py`

```python
class MLTrainingState(Enum):
    # Existing states...
    CHOOSING_DATA_SOURCE = "choosing_data_source"
    AWAITING_FILE_PATH = "awaiting_file_path"
    CONFIRMING_SCHEMA = "confirming_schema"

    # NEW STATES
    CHOOSING_LOAD_OPTION = "choosing_load_option"
    AWAITING_SCHEMA_INPUT = "awaiting_schema_input"

    SELECTING_MODEL_TYPE = "selecting_model_type"
    # ... rest of states
```

**New Transitions**:
```python
ML_TRAINING_TRANSITIONS = {
    MLTrainingState.AWAITING_FILE_PATH.value: [
        MLTrainingState.CHOOSING_LOAD_OPTION.value  # NEW
    ],

    MLTrainingState.CHOOSING_LOAD_OPTION.value: [  # NEW
        MLTrainingState.CONFIRMING_SCHEMA.value,  # Load now path
        MLTrainingState.AWAITING_SCHEMA_INPUT.value  # Defer path
    ],

    MLTrainingState.AWAITING_SCHEMA_INPUT.value: [  # NEW
        MLTrainingState.SELECTING_MODEL_TYPE.value  # Skip target/feature selection
    ],

    # Existing transitions...
}
```

### 2. Session Data Structure

**File**: `src/core/state_manager.py`

```python
@dataclass
class UserSession:
    # Existing fields...
    user_id: int
    conversation_id: str
    workflow_type: Optional[WorkflowType] = None
    current_state: Optional[str] = None
    data_source: Optional[str] = None
    file_path: Optional[str] = None
    uploaded_data: Optional[pd.DataFrame] = None
    detected_schema: Optional[dict] = None

    # NEW FIELDS
    load_deferred: bool = False  # True if user chose deferred loading
    manual_schema: Optional[dict] = None  # User-provided schema
    # Format: {
    #     'target': str,
    #     'features': List[str],
    #     'task': Optional[str]  # 'regression' | 'classification'
    # }
```

### 3. Schema Parser Utility

**New File**: `src/utils/schema_parser.py`

```python
"""
Parse user-provided schema from multiple input formats.
"""

from typing import Dict, List, Optional
import json
import re

class SchemaParseError(Exception):
    """Schema parsing failed."""
    pass


def parse_schema_input(text: str) -> Dict[str, any]:
    """
    Parse schema from multiple formats.

    Supported Formats:

    1. Key-Value (recommended for Telegram):
       target: price
       features: sqft, bedrooms, bathrooms
       task: regression

    2. JSON:
       {"target": "price", "features": ["sqft", "bedrooms"], "task": "regression"}

    3. Simple List (assumes first column = target):
       price
       sqft
       bedrooms

    Returns:
        {
            'target': str,
            'features': List[str],
            'task': Optional[str]
        }

    Raises:
        SchemaParseError: If parsing fails or required fields missing
    """
    text = text.strip()

    # Try JSON first
    if text.startswith('{'):
        try:
            schema = json.loads(text)
            return _validate_schema_dict(schema)
        except json.JSONDecodeError:
            pass

    # Try key-value format
    if ':' in text:
        return _parse_key_value_format(text)

    # Try simple list format
    if '\n' in text or ',' in text:
        return _parse_simple_list_format(text)

    raise SchemaParseError(
        "Could not parse schema. Please use one of the supported formats."
    )


def _parse_key_value_format(text: str) -> Dict:
    """Parse 'key: value' format."""
    schema = {}

    for line in text.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()

        if key == 'target':
            schema['target'] = value
        elif key in ('features', 'feature'):
            # Handle comma-separated or newline-separated
            schema['features'] = [f.strip() for f in value.split(',')]
        elif key == 'task':
            schema['task'] = value.lower()

    return _validate_schema_dict(schema)


def _parse_simple_list_format(text: str) -> Dict:
    """Parse simple column list (first = target)."""
    columns = [c.strip() for c in re.split(r'[,\n]', text) if c.strip()]

    if len(columns) < 2:
        raise SchemaParseError("Need at least 2 columns (target + 1 feature)")

    return {
        'target': columns[0],
        'features': columns[1:],
        'task': None  # User must specify separately
    }


def _validate_schema_dict(schema: Dict) -> Dict:
    """Validate schema has required fields."""
    if 'target' not in schema:
        raise SchemaParseError("Missing 'target' column")

    if 'features' not in schema or not schema['features']:
        raise SchemaParseError("Missing 'features' columns")

    # Ensure features is a list
    if isinstance(schema['features'], str):
        schema['features'] = [f.strip() for f in schema['features'].split(',')]

    # Validate task type if provided
    if 'task' in schema and schema['task']:
        valid_tasks = ['regression', 'classification']
        if schema['task'].lower() not in valid_tasks:
            raise SchemaParseError(
                f"Invalid task type: {schema['task']}. Must be one of: {valid_tasks}"
            )
        schema['task'] = schema['task'].lower()

    return schema
```

### 4. Handler Functions

**File**: `src/bot/ml_handlers/ml_training_local_path.py`

```python
# NEW HANDLER 1: Load Option Selection
async def handle_load_option_selection(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle load option button click (load now vs defer)."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    choice = query.data.split(":")[-1]  # "load_now" or "defer"

    session = await self.state_manager.get_session(user_id, str(chat_id))

    if choice == "load_now":
        # Existing workflow: load data immediately
        session.load_deferred = False
        # ... (existing code for loading and schema detection)

    elif choice == "defer":
        # New workflow: defer loading, ask for manual schema
        session.load_deferred = True
        await self.state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_SCHEMA_INPUT.value
        )

        await query.edit_message_text(
            LocalPathMessages.schema_input_prompt(),
            parse_mode="Markdown"
        )


# NEW HANDLER 2: Schema Input
async def handle_schema_input(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle manual schema input (text or file upload)."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    session = await self.state_manager.get_session(user_id, str(chat_id))

    # Validate state
    if session.current_state != MLTrainingState.AWAITING_SCHEMA_INPUT.value:
        return

    # Get schema text (from message or uploaded file)
    if update.message.document:
        # User uploaded schema file
        file = await update.message.document.get_file()
        schema_text = (await file.download_as_bytearray()).decode('utf-8')
    else:
        # User typed schema
        schema_text = update.message.text.strip()

    try:
        # Parse schema
        from src.utils.schema_parser import parse_schema_input
        parsed_schema = parse_schema_input(schema_text)

        # Store in session
        session.manual_schema = parsed_schema

        # Transition to model selection (skip target/feature selection)
        await self.state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_MODEL_TYPE.value
        )

        # Show confirmation
        await update.message.reply_text(
            LocalPathMessages.schema_accepted_message(parsed_schema),
            parse_mode="Markdown"
        )

    except Exception as e:
        # Schema parsing failed
        await update.message.reply_text(
            LocalPathMessages.schema_parse_error(str(e)),
            parse_mode="Markdown"
        )


# MODIFIED: File Path Handler
async def handle_file_path_input(self, update, context):
    """Handle file path input - now transitions to load option."""
    # ... existing validation code ...

    # NEW: Instead of loading immediately, ask for load option
    await self.state_manager.transition_state(
        session,
        MLTrainingState.CHOOSING_LOAD_OPTION.value
    )

    keyboard = [
        [InlineKeyboardButton("📊 Load Now", callback_data="load_option:load_now")],
        [InlineKeyboardButton("⏭️ Defer Loading", callback_data="load_option:defer")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        LocalPathMessages.load_option_prompt(session.file_path),
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
```

### 5. Messages

**File**: `src/bot/messages/local_path_messages.py`

```python
@staticmethod
def load_option_prompt(file_path: str) -> str:
    """Prompt for load option selection."""
    return (
        "📊 **Data Loading Options**\n\n"
        f"File: `{file_path}`\n\n"
        "**Option 1: Load Now** 📥\n"
        "✓ View data preview\n"
        "✓ Auto-detect schema\n"
        "✓ Validate columns immediately\n"
        "⚠️ May be slow for large files (>1M rows)\n\n"
        "**Option 2: Defer Loading** ⏭️\n"
        "✓ Provide schema manually\n"
        "✓ Faster workflow (no loading delay)\n"
        "✓ Works with massive datasets (10M+ rows)\n"
        "⚠️ Column errors caught at training time\n\n"
        "Choose your preferred method:"
    )


@staticmethod
def schema_input_prompt() -> str:
    """Prompt for manual schema input."""
    return (
        "📝 **Provide Data Schema**\n\n"
        "Enter your dataset schema in any of these formats:\n\n"
        "**Format 1 (Recommended):**\n"
        "```\n"
        "target: price\n"
        "features: sqft, bedrooms, bathrooms\n"
        "task: regression\n"
        "```\n\n"
        "**Format 2 (JSON):**\n"
        "```json\n"
        '{"target": "price", "features": ["sqft", "bedrooms"], "task": "regression"}\n'
        "```\n\n"
        "**Format 3 (Simple List):**\n"
        "```\n"
        "price\n"
        "sqft\n"
        "bedrooms\n"
        "```\n"
        "(First column = target)\n\n"
        "Or upload a `schema.txt` file with the same format."
    )


@staticmethod
def schema_accepted_message(schema: dict) -> str:
    """Confirmation that schema was parsed successfully."""
    features_str = ", ".join(schema['features'])
    task_str = f" | Task: {schema.get('task', 'auto-detect')}"

    return (
        "✅ **Schema Accepted**\n\n"
        f"Target: `{schema['target']}`\n"
        f"Features: `{features_str}`{task_str}\n\n"
        "Data will be loaded during training.\n"
        "Proceeding to model selection..."
    )


@staticmethod
def schema_parse_error(error: str) -> str:
    """Error message for schema parsing failure."""
    return (
        "❌ **Schema Parse Error**\n\n"
        f"Error: {error}\n\n"
        "Please check your schema format and try again.\n"
        "Type `/help schema` for format examples."
    )
```

### 6. ML Engine Integration

**File**: `src/engines/ml_engine.py`

The ML engine likely already supports lazy loading. Key check:

```python
def train_model(self, data, target_column, feature_columns, ...):
    """
    Train ML model.

    Args:
        data: Can be pd.DataFrame OR file path (str)
    """
    # Add lazy loading support
    if isinstance(data, str):
        # It's a file path - load now
        data = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        # Already loaded
        pass
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    # Continue with training...
```

If not already supported, add this lazy loading check at the beginning of `train_model()`.

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Add new states to `MLTrainingState` enum
- [ ] Add state transitions
- [ ] Add session fields (`load_deferred`, `manual_schema`)
- [ ] Create `schema_parser.py` with all format support
- [ ] Add unit tests for schema parser

### Phase 2: Bot Handlers
- [ ] Add `handle_load_option_selection()` handler
- [ ] Add `handle_schema_input()` handler
- [ ] Modify `handle_file_path_input()` to show load options
- [ ] Register callback handlers for load option buttons
- [ ] Register message handler for schema input

### Phase 3: Messages & UX
- [ ] Add `load_option_prompt()` message
- [ ] Add `schema_input_prompt()` message
- [ ] Add `schema_accepted_message()` message
- [ ] Add `schema_parse_error()` message
- [ ] Design inline keyboard layouts

### Phase 4: ML Engine
- [ ] Verify ML engine supports lazy loading
- [ ] Add lazy loading if not present
- [ ] Handle deferred schema in training workflow
- [ ] Add validation for manual schema at training time

### Phase 5: Testing
- [ ] Unit tests for schema parser (all 3 formats)
- [ ] Integration test: deferred path end-to-end
- [ ] Integration test: immediate path (backward compatibility)
- [ ] Error handling tests (invalid schema, wrong columns)
- [ ] Large dataset test (10M rows)

---

## Testing Strategy

### Unit Tests

**Schema Parser Tests** (`tests/unit/test_schema_parser.py`):
```python
def test_parse_key_value_format():
    schema = parse_schema_input("target: price\nfeatures: sqft, bedrooms")
    assert schema['target'] == 'price'
    assert schema['features'] == ['sqft', 'bedrooms']

def test_parse_json_format():
    schema = parse_schema_input('{"target": "price", "features": ["sqft"]}')
    assert schema['target'] == 'price'

def test_parse_simple_list():
    schema = parse_schema_input("price\nsqft\nbedrooms")
    assert schema['target'] == 'price'
    assert schema['features'] == ['sqft', 'bedrooms']

def test_invalid_schema_raises_error():
    with pytest.raises(SchemaParseError):
        parse_schema_input("invalid format")
```

### Integration Tests

**Deferred Loading Workflow** (`tests/integration/test_deferred_loading.py`):
```python
@pytest.mark.asyncio
async def test_complete_deferred_workflow(mock_bot, large_csv):
    # 1. /train
    await mock_bot.send_message("/train")

    # 2. Choose local path
    await mock_bot.click_button("📂 Use Local Path")

    # 3. Provide file path
    await mock_bot.send_message(str(large_csv))

    # 4. Choose defer loading
    await mock_bot.click_button("⏭️ Defer Loading")

    # 5. Provide schema
    await mock_bot.send_message("target: price\nfeatures: sqft, bedrooms")

    # 6. Verify schema accepted
    response = await mock_bot.get_last_message()
    assert "Schema Accepted" in response

    # 7. Select model
    await mock_bot.send_message("random_forest")

    # 8. Train
    response = await mock_bot.wait_for_training()
    assert "Model trained successfully" in response
```

---

## Benefits

### User Benefits
✅ **Massive Dataset Support**: Train on 10M+ row datasets without Telegram message limits
✅ **Faster Workflow**: Skip data loading delay if schema is known
✅ **Memory Efficient**: Bot doesn't hold large DataFrames in memory
✅ **Flexibility**: Choose preview (load now) vs speed (defer)

### Technical Benefits
✅ **Backward Compatible**: Existing immediate loading workflow unchanged
✅ **Clean Architecture**: Schema parser is reusable utility
✅ **State Machine Clarity**: New states clearly represent decision points
✅ **Lazy Loading**: Data loaded only when needed (training time)

---

## Trade-offs & Limitations

### User Experience
⚠️ **Delayed Error Feedback**: Column name typos only caught at training time (not during bot interaction)
⚠️ **No Data Preview**: Users don't see data sample before training
⚠️ **Schema Knowledge Required**: User must know column names and types upfront

### Technical
⚠️ **Error Handling Complexity**: Need clear error messages when training fails due to bad schema
⚠️ **Schema Validation Limited**: Can't validate column names against actual file without loading
⚠️ **Testing Complexity**: Need large test datasets to properly test deferred path

---

## Estimated Effort

### Development Time
- **Schema Parser**: 2 hours (including tests)
- **State Machine Changes**: 1 hour
- **Handler Implementation**: 3 hours
- **Messages & UX**: 1 hour
- **ML Engine Integration**: 1 hour
- **Testing**: 2 hours

**Total**: ~10 hours (1.5 days)

### Lines of Code
- **New Code**: ~400 lines
  - Schema parser: ~150 lines
  - Handlers: ~150 lines
  - Messages: ~100 lines
- **Modified Code**: ~150 lines
  - State machine: ~50 lines
  - ML engine: ~50 lines
  - Existing handlers: ~50 lines
- **Test Code**: ~300 lines

**Total**: ~850 lines

---

## Future Enhancements

### Phase 3 (Future)
- **Schema Templates**: Pre-built schema templates for common datasets (housing, sales, etc.)
- **Column Name Suggestions**: Fuzzy matching to suggest corrections for typos
- **Partial Preview**: Show first 100 rows even for large datasets (streaming preview)
- **Schema Inference from Metadata**: Use file metadata (Parquet schema, CSV headers) without full load
- **Async Data Loading**: Load data in background while user selects model

---

## Related Documents
- `file-path-training-1.md` - Phase 1: Path validation and security
- `CLAUDE.md` - Overall project architecture
- `/tests/unit/handlers/test_ml_training_local_path.py` - Existing handler tests

---

## Status Updates

**2025-10-08**: Initial planning document created based on ultrathink analysis
