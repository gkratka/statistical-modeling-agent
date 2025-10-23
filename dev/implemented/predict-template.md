# Prediction Template Features Implementation Plan

## Overview

Add template functionality to the `/predict` workflow, mirroring the template system already implemented in `/train` workflow. This enables users to save prediction configurations and reuse them for future predictions.

**Two Features**:
1. **Save Template** - After prediction file is saved, offer option to save configuration as template
2. **Load Template** - At workflow start, offer option to load saved template instead of manual configuration

## Architecture Changes

### 1. New Data Structures

**File**: `src/core/prediction_template.py`

```python
@dataclass
class PredictionTemplate:
    """Prediction workflow template configuration."""
    template_id: str              # user_{user_id}_{template_name}_{timestamp}
    template_name: str            # User-provided name
    user_id: int                  # Template owner

    # Prediction configuration
    file_path: str                # Data file path for predictions
    model_id: str                 # Trained model to use
    feature_columns: List[str]    # Features (must match model)
    output_column_name: str       # Prediction column name

    # Optional fields
    save_path: Optional[str]      # Default output location
    description: Optional[str]    # User notes

    # Metadata
    created_at: str               # ISO timestamp
    last_used: Optional[str]      # ISO timestamp
```

**File**: `src/core/prediction_template_manager.py`
- CRUD operations: save_template(), load_template(), list_templates(), delete_template()
- Validation: validate_template_name(), template_exists()
- Storage: `templates/predictions/user_{user_id}/{template_name}.json`

### 2. State Machine Changes

**File**: `src/core/state_manager.py`

**New MLPredictionState States**:
```python
LOADING_PRED_TEMPLATE = "loading_pred_template"      # Browsing templates
CONFIRMING_PRED_TEMPLATE = "confirming_pred_template" # Reviewing selected template
SAVING_PRED_TEMPLATE = "saving_pred_template"        # Entering template name
```

**New State Transitions**:
```python
ML_PREDICTION_TRANSITIONS = {
    # Existing transitions...

    # Load template path (Feature 2)
    MLPredictionState.STARTED.value: {
        MLPredictionState.CHOOSING_DATA_SOURCE.value,  # existing
        MLPredictionState.LOADING_PRED_TEMPLATE.value  # NEW
    },
    MLPredictionState.LOADING_PRED_TEMPLATE.value: {
        MLPredictionState.CONFIRMING_PRED_TEMPLATE.value,
        MLPredictionState.STARTED.value  # back
    },
    MLPredictionState.CONFIRMING_PRED_TEMPLATE.value: {
        MLPredictionState.READY_TO_RUN.value,  # after loading data
        MLPredictionState.LOADING_PRED_TEMPLATE.value  # back
    },

    # Save template path (Feature 1)
    MLPredictionState.COMPLETE.value: {
        MLPredictionState.AWAITING_SAVE_PATH.value,  # existing
        MLPredictionState.SAVING_PRED_TEMPLATE.value  # NEW
    },
    MLPredictionState.SAVING_PRED_TEMPLATE.value: {
        MLPredictionState.COMPLETE.value  # after save or cancel
    }
}
```

## Feature 1: Save Template at End of Prediction

### User Flow

**Step 1**: After prediction file saved successfully
- Current: Shows "✅ Prediction file saved!" with "✅ Done" button
- **NEW**: Add "💾 Save as Template" button

**Step 2**: User clicks "💾 Save as Template"
- Transition to SAVING_PRED_TEMPLATE state
- Show prompt: "Enter template name"
- Display validation rules (alphanumeric + underscore, max 32 chars)
- Show "❌ Cancel" button

**Step 3**: User enters template name
- Validate name format
- Check if template already exists
- Extract configuration from session:
  * file_path: session.file_path
  * model_id: session.selections["model_id"]
  * feature_columns: Get from model's training features
  * output_column_name: session.selections.get("prediction_column_name", "prediction")
- Save template via PredictionTemplateManager
- Show success message: "✅ Template '{name}' saved!"
- Return to COMPLETE state

### Implementation Details

**File**: `src/bot/ml_handlers/prediction_template_handlers.py`

**Handler Methods**:
1. `handle_template_save_request(update, context)` - Button click handler
   - Get session, validate state
   - Transition to SAVING_PRED_TEMPLATE
   - Display prompt with cancel button

2. `handle_template_name_input(update, context)` - Text input handler
   - Validate template name
   - Build template config from session
   - Call template_manager.save_template()
   - Show success/error message
   - Transition back to COMPLETE

**Integration Point**: `src/bot/ml_handlers/prediction_handlers.py`

Modify `_execute_file_save()` method (around line 1585) to add template save button:

```python
# After file save success
keyboard = [
    [InlineKeyboardButton("💾 Save as Template", callback_data="save_pred_template")],
    [InlineKeyboardButton("✅ Done", callback_data="complete")]
]
```

**Callback Pattern**: `save_pred_template`

## Feature 2: Load Template at Beginning of Prediction

### User Flow

**Step 1**: User runs `/predict` command
- Current: Shows "Choose data source" with 2 buttons (Upload, Local Path)
- **NEW**: Add 3rd button "📋 Use Template"

**Step 2**: User clicks "📋 Use Template"
- Transition to LOADING_PRED_TEMPLATE state
- List user's saved prediction templates as buttons
- Each button: "📄 {template_name}"
- Show "🔙 Back" button to return

**Step 3**: User selects a template
- Load template data via PredictionTemplateManager
- Validate model_id exists
- Transition to CONFIRMING_PRED_TEMPLATE state
- Display template summary:
  * Template name
  * File path for predictions
  * Model ID and type
  * Feature columns
  * Output column name
  * Created date
- Show buttons: "✅ Use This Template" and "🔙 Back to Templates"

**Step 4**: User confirms template
- Populate session with template data:
  * session.file_path = template.file_path
  * session.selections["model_id"] = template.model_id
  * session.selections["feature_columns"] = template.feature_columns
  * session.selections["prediction_column_name"] = template.output_column_name
- Validate file path via path_validator
- Load data from file_path
- Update template.last_used timestamp
- Transition to READY_TO_RUN state (skip feature selection, model selection)
- Show "Ready to predict" message with "🚀 Start Prediction" button

### Implementation Details

**File**: `src/bot/ml_handlers/prediction_template_handlers.py`

**Handler Methods**:
1. `handle_template_source_selection(update, context)` - "Use Template" button click
   - Get session, validate state
   - Transition to LOADING_PRED_TEMPLATE
   - List templates via template_manager.list_templates()
   - Display as buttons with callback pattern

2. `handle_template_selection(update, context)` - User picks template
   - Extract template_name from callback_data
   - Load template via template_manager.load_template()
   - Validate model exists
   - Transition to CONFIRMING_PRED_TEMPLATE
   - Display template summary

3. `handle_template_confirmation(update, context)` - User confirms
   - Populate session with template data
   - Validate file_path via path_validator
   - Load data via data_loader.load_from_local_path()
   - Update template.last_used
   - Transition to READY_TO_RUN
   - Show ready message

4. `handle_cancel_template(update, context)` - Cancel operations
   - Restore previous state via session.restore_previous_state()
   - Show cancellation message

**Integration Point**: `src/bot/ml_handlers/prediction_handlers.py`

Modify `handle_start_prediction()` method (line 85-135) to add template button:

```python
# Data source selection
keyboard = [
    [InlineKeyboardButton("📥 Upload File via Telegram", callback_data="data_source_telegram")],
    [InlineKeyboardButton("📁 Use Local File Path", callback_data="data_source_local")],
    [InlineKeyboardButton("📋 Use Template", callback_data="use_pred_template")],  # NEW
    [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
]
```

**Callback Patterns**:
- `use_pred_template` - Start template loading
- `load_pred_template:{name}` - Select specific template
- `confirm_pred_template` - Confirm template selection
- `cancel_pred_template` - Cancel operation

## UI Messages

**File**: `src/bot/messages/prediction_template_messages.py`

**Required Messages**:
```python
# Save workflow
PRED_TEMPLATE_SAVE_PROMPT = "📝 *Enter template name:*\n\n*Rules:* ..."
PRED_TEMPLATE_SAVED_SUCCESS = "✅ *Template '{name}' saved!*"
PRED_TEMPLATE_INVALID_NAME = "❌ *Invalid name:* {error}"
PRED_TEMPLATE_EXISTS = "⚠️ *Template '{name}' already exists*"
PRED_TEMPLATE_SAVE_FAILED = "❌ *Failed to save:* {error}"

# Load workflow
PRED_TEMPLATE_LOAD_PROMPT = "📋 *Select a template:*\n\nYou have {count} template(s)."
PRED_TEMPLATE_NO_TEMPLATES = "📋 *No templates found*"
PRED_TEMPLATE_NOT_FOUND = "❌ *Template '{name}' not found*"
PRED_TEMPLATE_LOAD_FAILED = "❌ *Failed to load:* {error}"
PRED_TEMPLATE_MODEL_INVALID = "❌ *Model '{model_id}' not found*"
PRED_TEMPLATE_FILE_PATH_INVALID = "❌ *File path invalid:* {error}"
PRED_TEMPLATE_DATA_LOADED = "✅ *Data loaded from template!*"

# Helper function
def format_pred_template_summary(template_name, file_path, model_id,
                                 features, output_col, created_at) -> str:
    """Format template summary for display."""
```

## Handler Registration

**File**: `src/bot/telegram_bot.py`

Add to `_setup_handlers()` method (after line 211):

```python
# Register prediction template handlers
from src.bot.ml_handlers.prediction_template_handlers import PredictionTemplateHandlers
from src.core.prediction_template_manager import PredictionTemplateManager
from src.core.prediction_template import PredictionTemplateConfig

# Initialize managers
pred_template_config = PredictionTemplateConfig.get_default()
pred_template_manager = PredictionTemplateManager(pred_template_config)

# Create handlers
pred_template_handlers = PredictionTemplateHandlers(
    state_manager=state_manager,
    template_manager=pred_template_manager,
    data_loader=data_loader,
    path_validator=path_validator
)

# Register callback handlers
self.application.add_handler(
    CallbackQueryHandler(
        pred_template_handlers.handle_template_source_selection,
        pattern="^use_pred_template$"
    )
)
self.application.add_handler(
    CallbackQueryHandler(
        pred_template_handlers.handle_template_selection,
        pattern="^load_pred_template:.+$"
    )
)
self.application.add_handler(
    CallbackQueryHandler(
        pred_template_handlers.handle_template_confirmation,
        pattern="^confirm_pred_template$"
    )
)
self.application.add_handler(
    CallbackQueryHandler(
        pred_template_handlers.handle_template_save_request,
        pattern="^save_pred_template$"
    )
)
self.application.add_handler(
    CallbackQueryHandler(
        pred_template_handlers.handle_cancel_template,
        pattern="^cancel_pred_template$"
    )
)

# Store in bot_data
self.application.bot_data['pred_template_handlers'] = pred_template_handlers
self.application.bot_data['pred_template_manager'] = pred_template_manager
```

## Error Handling & Validation

### Template Load Validation

1. **Model Validation**:
   - Check if model_id exists in user's models
   - If not found, show error: "Model '{model_id}' not found. Please choose a different template."

2. **File Path Validation**:
   - Use path_validator.validate_path() before loading
   - If invalid, show error with reason
   - Allow user to select different template

3. **Feature Column Validation**:
   - Verify template.feature_columns match model's expected features
   - If mismatch, show warning but allow continuation

### Template Save Validation

1. **Name Validation**:
   - Alphanumeric and underscore only
   - Max 32 characters
   - No reserved names
   - Must be unique

2. **Configuration Validation**:
   - Ensure model_id exists in session
   - Ensure file_path is present
   - Ensure feature_columns are populated

### State Management

- Use `session.save_state_snapshot()` before each state transition
- Allow `session.restore_previous_state()` for back navigation and cancellations
- Handle workflow cleanup on errors

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_prediction_template.py`
- Test PredictionTemplate dataclass creation
- Test to_dict/from_dict conversion
- Test validation logic

**File**: `tests/unit/test_prediction_template_manager.py`
- Test save_template() - create and update
- Test load_template() - success and not found
- Test list_templates() - sorting by last_used
- Test delete_template()
- Test validate_template_name() - all validation rules
- Test template_exists()

### Integration Tests

**File**: `tests/integration/test_prediction_template_workflow.py`
- Test complete save template workflow
- Test complete load template workflow
- Test template with invalid model_id
- Test template with invalid file_path
- Test cancel operations at each step
- Test template name validation errors
- Test back navigation

## File Summary

### New Files (4)
1. `src/core/prediction_template.py` - Data structures
2. `src/core/prediction_template_manager.py` - CRUD operations
3. `src/bot/ml_handlers/prediction_template_handlers.py` - Telegram handlers
4. `src/bot/messages/prediction_template_messages.py` - UI messages

### Modified Files (3)
1. `src/core/state_manager.py` - Add 3 new states, update transitions
2. `src/bot/ml_handlers/prediction_handlers.py` - Add template buttons
3. `src/bot/telegram_bot.py` - Register handlers

### Test Files (2)
1. `tests/unit/test_prediction_template_manager.py`
2. `tests/integration/test_prediction_template_workflow.py`

## Implementation Phases

**Phase 1: Core Infrastructure** (~2 hours)
- Create prediction_template.py with dataclass
- Create prediction_template_manager.py with CRUD ops
- Add states to state_manager.py
- Write unit tests

**Phase 2: Feature 1 - Save Template** (~2 hours)
- Create prediction_template_handlers.py with save methods
- Create prediction_template_messages.py with save messages
- Modify prediction_handlers.py to add save button
- Write integration tests for save workflow

**Phase 3: Feature 2 - Load Template** (~3 hours)
- Add load methods to prediction_template_handlers.py
- Add load messages to prediction_template_messages.py
- Modify prediction_handlers.py to add load button
- Write integration tests for load workflow

**Phase 4: Integration & Testing** (~2 hours)
- Register handlers in telegram_bot.py
- End-to-end testing of both workflows
- Error handling and edge cases
- Documentation updates

**Total Estimated Time**: ~9 hours

## Risk Mitigation

1. **Model Deletion Risk**: Template references deleted model
   - Mitigation: Validate model existence when loading template
   - Show clear error message and alternative options

2. **File Path Changes**: Saved path no longer valid
   - Mitigation: Validate path before loading data
   - Provide option to update template or choose different one

3. **Feature Mismatch**: Template features don't match model
   - Mitigation: Warn user but allow continuation
   - Consider adding feature validation in future

4. **State Consistency**: Template workflow interrupted
   - Mitigation: Use state snapshots for rollback
   - Handle cleanup on errors

## Success Criteria

✅ Users can save prediction configuration as template after file save
✅ Users can load saved template at start of prediction workflow
✅ Template list shows all user's prediction templates
✅ Template summary displays all configuration details
✅ Model validation prevents using deleted models
✅ Path validation prevents using invalid file paths
✅ All unit tests pass (target: 20+ tests)
✅ All integration tests pass (target: 8+ scenarios)
✅ State machine transitions work correctly
✅ Error messages are clear and actionable
