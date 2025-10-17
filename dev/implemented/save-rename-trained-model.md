# Model Custom Naming & Save Feature - Implementation Plan

**Feature**: Custom model naming with optional save confirmation after ML training
**Status**: Planning Complete - Ready for Implementation
**Created**: 2025-10-13
**Estimated Effort**: 8 hours
**Related**: ml-engine.md, file-path-training.md, predict-workflow.md

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Solution Design](#solution-design)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Testing Strategy](#testing-strategy)
6. [Migration & Rollout](#migration--rollout)
7. [Future Enhancements](#future-enhancements)

---

## 1. Problem Analysis

### Current Behavior

**Issue Identified from User Feedback**:
```
✅ Training Complete!

📊 Metrics:
• Accuracy: 72.03%
• Loss: 0.6240

🆔 Model ID: model_7715560927_keras_binary_classification_20251014_044444

You can now use this model for predictions.
```

**Problems**:
1. ❌ **No explicit save confirmation** - Models auto-save silently
2. ❌ **Cryptic model IDs** - `model_7715560927_keras_binary_classification_20251014_044444`
3. ❌ **Hard to identify later** - "Which model was my housing price predictor?"
4. ❌ **No user control** - Can't provide meaningful names
5. ❌ **Poor UX in model lists** - Difficult to distinguish models

### User Requirements

**Primary Goals**:
1. Add save/name button after training completes
2. Allow custom, memorable model naming
3. Retrieve models by custom names in `/models` and `/predict`
4. Maintain backward compatibility with existing models

**User Stories**:
- *"As a data scientist, I want to name my models so I can easily identify them later"*
- *"As a user, I want meaningful model names like 'Housing Predictor v2' instead of cryptic IDs"*
- *"As a user, I want optional naming so I can skip if I'm in a hurry"*

---

## 2. Solution Design

### Recommended Approach: Auto-Save with Optional Custom Naming

**Design Philosophy**:
- ✅ **Keep auto-save** - Don't break existing behavior, reduce data loss risk
- ✅ **Make naming optional** - Users can skip if desired
- ✅ **Provide defaults** - Auto-generate readable names if user skips
- ✅ **Simple workflow** - 2-click process maximum

### User Workflow

#### Step 1: Training Completes (Enhanced)

```
✅ Training Complete!

📊 Metrics:
• Accuracy: 72.03%
• Loss: 0.6240

🆔 Model ID: model_7715560927_keras_binary_classification...

┌─────────────────────────────────────────┐
│ [💾 Name Model] [Skip - Use Default]   │
└─────────────────────────────────────────┘
```

#### Step 2: User Clicks "Name Model"

```
💾 Give Your Model a Name

Enter a memorable name for your model:

Examples:
• "Housing Price Predictor"
• "Customer Churn Model v2"
• "Spam Classifier - Production"

Type your name below:
```

#### Step 3: User Provides Name

```
User types: "Customer Churn Predictor"
```

#### Step 4: Confirmation

```
✅ Model Named Successfully!

📝 Name: "Customer Churn Predictor"
🆔 Model ID: model_7715560927...
📅 Created: Jan 14, 2025

You can now:
• Use /predict to make predictions
• View models with /models
• Train another with /train
```

#### Alternative: User Clicks "Skip"

Auto-generates readable default:
```
✅ Model Saved!

📝 Name: "Binary Classification - Jan 14, 2025"
�ID: model_7715560927...

You can now use this model for predictions.
```

---

## 3. Architecture

### System Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Telegram Bot Interface                      │
│                   (telegram_bot.py)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Training complete callback
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Training Handler                             │
│          (ml_training_local_path.py)                         │
│                                                              │
│  • handle_training_complete() → Show naming options         │
│  • handle_name_model_callback() → Prompt for name           │
│  • handle_model_name_input() → Process custom name          │
│  • handle_skip_naming_callback() → Use default name         │
└────────┬─────────────┬─────────────┬───────────────────────┘
         │             │             │
         │             │             │
         ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐
│StateManager  │ │MLEngine      │ │Model Storage             │
│              │ │              │ │                          │
│• New states: │ │• set_model_  │ │metadata.json:            │
│  TRAINING_   │ │  name()      │ │{                         │
│  COMPLETE    │ │• get_model_  │ │  "model_id": "...",      │
│  NAMING_     │ │  by_name()   │ │  "custom_name": "...",   │
│  MODEL       │ │• _validate_  │ │  "display_name": "...",  │
│  MODEL_      │ │  model_name()│ │  ...                     │
│  NAMED       │ │              │ │}                         │
└──────────────┘ └──────────────┘ └──────────────────────────┘
```

### Data Flow

```
Training Complete
       │
       ▼
Store model_id in session
       │
       ▼
Update state → TRAINING_COMPLETE
       │
       ▼
Display naming options
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
   Name Model      Skip      (timeout → auto-skip)
       │             │             │
       ▼             ▼             │
  NAMING_MODEL   Generate      ────┘
       │          default name
       ▼             │
  Prompt user       │
       │             │
       ▼             │
  Receive name      │
       │             │
       ▼             ▼
  Validate name  Save to metadata
       │             │
       ▼             ▼
  Save to      MODEL_NAMED state
  metadata           │
       │             │
       └─────┬───────┘
             ▼
       Confirmation message
```

---

## 4. Implementation Details

### Phase 1: Data Model Updates (30 min)

#### 1.1 Update Model Metadata Schema

**File**: `models/user_{user_id}/model_{model_id}/metadata.json`

**New Fields**:
```json
{
  "model_id": "model_7715560927_keras_binary_classification_20251014_044444",
  "custom_name": "Customer Churn Predictor",  // NEW: User-provided name
  "display_name": "Customer Churn Predictor", // NEW: For UI display
  "user_id": 7715560927,
  "model_type": "keras_binary_classification",
  "task_type": "classification",
  "created_at": "2025-01-14T21:44:00Z",
  "target_column": "churn",
  "feature_columns": ["age", "tenure", "monthly_spend"],
  "metrics": {
    "accuracy": 0.7203,
    "loss": 0.6240
  },
  // ... existing fields ...
}
```

**Field Specifications**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `custom_name` | string | No | User-provided custom name (3-100 chars) |
| `display_name` | string | Yes | Name to display in UI (custom or generated) |

**Validation Rules**:
- `custom_name`: 3-100 characters, alphanumeric + spaces/hyphens/underscores
- `display_name`: Always present, either custom or auto-generated
- Backward compatibility: If fields missing, generate on-the-fly

#### 1.2 Default Name Generation

**Function**: `_generate_default_name()`

```python
def _generate_default_name(
    model_type: str,
    task_type: str,
    created_at: str
) -> str:
    """
    Generate default model name when user skips custom naming.

    Format: "{Model Type Display} - {Date}"

    Args:
        model_type: Technical model type (e.g., 'keras_binary_classification')
        task_type: Task type ('classification' or 'regression')
        created_at: ISO format timestamp

    Returns:
        Human-readable default name

    Examples:
        >>> _generate_default_name('keras_binary_classification', 'classification', '2025-01-14T21:44:00Z')
        'Binary Classification - Jan 14, 2025'

        >>> _generate_default_name('random_forest', 'regression', '2025-01-10T15:30:00Z')
        'Random Forest - Jan 10, 2025'
    """
    # Convert timestamp to readable date
    from datetime import datetime
    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
    date_str = dt.strftime("%b %d, %Y")

    # Convert model type to display name
    model_display = model_type.replace('_', ' ').title()

    # Simplify common names
    simplifications = {
        'Keras Binary Classification': 'Binary Classification',
        'Keras Multiclass Classification': 'Multiclass Classification',
        'Keras Regression': 'Neural Network Regression',
        'Random Forest': 'Random Forest',
        'Logistic': 'Logistic Regression',
        'Linear': 'Linear Regression'
    }

    model_display = simplifications.get(model_display, model_display)

    return f"{model_display} - {date_str}"
```

---

### Phase 2: MLEngine Enhancements (1.5 hours)

#### 2.1 Add Name Management Methods

**File**: `src/engines/ml_engine.py`

**New Method 1: Set Model Name**

```python
def set_model_name(
    self,
    user_id: int,
    model_id: str,
    custom_name: str
) -> bool:
    """
    Set custom name for a trained model.

    Updates the model's metadata.json with custom_name and display_name.

    Args:
        user_id: User identifier
        model_id: Model identifier (technical ID)
        custom_name: User-provided custom name

    Returns:
        True if successful, False otherwise

    Raises:
        ModelNotFoundError: If model doesn't exist
        ValidationError: If custom_name is invalid format

    Example:
        >>> ml_engine.set_model_name(
        ...     user_id=12345,
        ...     model_id="model_12345_linear_20251014",
        ...     custom_name="Housing Price Predictor"
        ... )
        True
    """
    # 1. Validate name format
    is_valid, error_msg = self._validate_model_name(custom_name)
    if not is_valid:
        raise ValidationError(error_msg)

    # 2. Check for duplicate names (warn only, don't block)
    existing = self.get_model_by_name(user_id, custom_name)
    if existing:
        self.logger.warning(
            f"User {user_id} already has a model named '{custom_name}'"
        )

    # 3. Get model directory
    model_dir = self._get_model_dir(user_id, model_id)
    if not model_dir.exists():
        raise ModelNotFoundError(f"Model {model_id} not found")

    # 4. Load existing metadata
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # 5. Update custom_name and display_name
    metadata['custom_name'] = custom_name
    metadata['display_name'] = custom_name

    # 6. Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    self.logger.info(f"Model {model_id} named: {custom_name}")
    return True
```

**New Method 2: Get Model by Name**

```python
def get_model_by_name(
    self,
    user_id: int,
    custom_name: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve model by custom name.

    Searches user's models for one with matching custom_name.
    If multiple models have the same name, returns the most recent.

    Args:
        user_id: User identifier
        custom_name: Custom model name to search for

    Returns:
        Model info dict if found, None otherwise

    Example:
        >>> model = ml_engine.get_model_by_name(12345, "Housing Predictor")
        >>> print(model['model_id'])
        'model_12345_linear_20251014_123456'
    """
    models = self.list_models(user_id)

    # Find all matching models
    matches = [
        model for model in models
        if model.get('custom_name') == custom_name
    ]

    if not matches:
        return None

    # If multiple matches, return most recent
    if len(matches) > 1:
        self.logger.warning(
            f"Multiple models named '{custom_name}' for user {user_id}, "
            f"returning most recent"
        )
        matches.sort(key=lambda m: m.get('created_at', ''), reverse=True)

    return matches[0]
```

**New Method 3: Validate Model Name**

```python
def _validate_model_name(self, name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate model name format.

    Rules:
    - Length: 3-100 characters
    - Allowed: letters, numbers, spaces, hyphens, underscores
    - Not allowed: special characters, path separators, quotes

    Args:
        name: Proposed model name

    Returns:
        (is_valid, error_message)

    Examples:
        >>> _validate_model_name("My Model")
        (True, None)

        >>> _validate_model_name("ab")
        (False, "Name must be at least 3 characters")

        >>> _validate_model_name("model/test")
        (False, "Name can only contain letters, numbers, spaces, hyphens, and underscores")
    """
    # Check empty/whitespace
    if not name or not name.strip():
        return False, "Name cannot be empty"

    name = name.strip()

    # Length validation
    if len(name) < 3:
        return False, "Name must be at least 3 characters"

    if len(name) > 100:
        return False, "Name must be less than 100 characters"

    # Character validation
    import re
    pattern = r'^[a-zA-Z0-9\s\-_]+$'
    if not re.match(pattern, name):
        return False, (
            "Name can only contain letters, numbers, spaces, "
            "hyphens, and underscores"
        )

    return True, None
```

#### 2.2 Update list_models() Method

**Enhancement**: Add display names to model list

```python
def list_models(
    self,
    user_id: int,
    task_type: Optional[str] = None,
    model_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List user's models with optional filtering.

    Now includes display_name field for UI presentation.

    Args:
        user_id: User identifier
        task_type: Optional filter by task type
        model_type: Optional filter by model type

    Returns:
        List of model info dicts with display_name

    Example:
        >>> models = ml_engine.list_models(12345)
        >>> for model in models:
        ...     print(f"{model['display_name']} ({model['model_id']})")
        Housing Price Predictor (model_12345_linear_20251014)
        Binary Classification - Jan 10, 2025 (model_12345_keras_20251010)
    """
    # ... existing code to load models ...

    # Enhancement: Add display_name to each model
    for model in models:
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if custom_name exists
        custom_name = metadata.get('custom_name')

        if custom_name:
            model['display_name'] = custom_name
            model['custom_name'] = custom_name
        else:
            # Generate default display name
            default_name = self._generate_default_name(
                model_type=metadata['model_type'],
                task_type=metadata['task_type'],
                created_at=metadata['created_at']
            )
            model['display_name'] = default_name
            model['custom_name'] = None

    return models
```

---

### Phase 3: State Machine Updates (30 min)

#### 3.1 Add New States

**File**: `src/core/state_manager.py`

**New State Enum Values**:

```python
class MLTrainingState(Enum):
    """States for ML training workflow."""

    # ... existing states ...
    AWAITING_TARGET_SELECTION = "awaiting_target_selection"
    AWAITING_FEATURE_SELECTION = "awaiting_feature_selection"
    CONFIRMING_PARAMETERS = "confirming_parameters"
    TRAINING_IN_PROGRESS = "training_in_progress"

    # NEW STATES for model naming
    TRAINING_COMPLETE = "training_complete"      # Training finished, showing options
    NAMING_MODEL = "naming_model"                # User is entering custom name
    MODEL_NAMED = "model_named"                  # Name set, workflow complete
```

#### 3.2 State Transitions

**Update**: `ML_TRAINING_TRANSITIONS` dictionary

```python
ML_TRAINING_TRANSITIONS = {
    # ... existing transitions ...

    MLTrainingState.TRAINING_IN_PROGRESS: [
        MLTrainingState.TRAINING_COMPLETE,     # Training finishes successfully
        MLTrainingState.STARTED                # Error → restart
    ],

    # NEW TRANSITIONS
    MLTrainingState.TRAINING_COMPLETE: [
        MLTrainingState.NAMING_MODEL,          # User clicks "Name Model"
        MLTrainingState.MODEL_NAMED            # User clicks "Skip"
    ],

    MLTrainingState.NAMING_MODEL: [
        MLTrainingState.MODEL_NAMED,           # After name provided
        MLTrainingState.TRAINING_COMPLETE      # Back button (optional)
    ],

    MLTrainingState.MODEL_NAMED: [
        # Terminal state - workflow complete
    ]
}
```

#### 3.3 State Prerequisites

```python
ML_TRAINING_PREREQUISITES = {
    # ... existing prerequisites ...

    # NEW PREREQUISITES
    MLTrainingState.NAMING_MODEL: lambda s: (
        s.temp_data.get('pending_model_id') is not None
    ),

    MLTrainingState.MODEL_NAMED: lambda s: (
        s.temp_data.get('pending_model_id') is not None
    )
}
```

---

### Phase 4: Telegram Handler Updates (2 hours)

#### 4.1 Update Training Complete Handler

**File**: `src/bot/ml_handlers/ml_training_local_path.py` (around line 1800)

**Current Code**:
```python
if result.get('success'):
    metrics_text = self._format_keras_metrics(result.get('metrics', {}))
    await update.effective_message.reply_text(
        f"✅ **Training Complete!**\n\n"
        f"📊 **Metrics:**\n{metrics_text}\n\n"
        f"🆔 **Model ID:** `{result.get('model_id', 'N/A')}`\n\n"
        f"You can now use this model for predictions.",
        parse_mode="Markdown"
    )
```

**New Code**:
```python
if result.get('success'):
    metrics_text = self._format_keras_metrics(result.get('metrics', {}))
    model_id = result.get('model_id', 'N/A')

    # Store model_id in session for naming workflow
    session = await self.state_manager.get_session(
        user_id=user_id,
        conversation_id=str(update.effective_chat.id)
    )
    session.temp_data['pending_model_id'] = model_id
    session.current_state = MLTrainingState.TRAINING_COMPLETE.value
    await self.state_manager.update_session(session)

    # Create inline keyboard with naming options
    keyboard = [
        [
            InlineKeyboardButton(
                "💾 Name Model",
                callback_data=f"name_model_{model_id}"
            ),
            InlineKeyboardButton(
                "Skip - Use Default",
                callback_data=f"skip_naming_{model_id}"
            )
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.effective_message.reply_text(
        f"✅ **Training Complete!**\n\n"
        f"📊 **Metrics:**\n{metrics_text}\n\n"
        f"🆔 **Model ID:** `{model_id}`\n\n"
        f"💾 **Would you like to name your model?**",
        parse_mode="Markdown",
        reply_markup=reply_markup
    )
```

#### 4.2 Add Name Model Callback Handler

**New Method**: `handle_name_model_callback()`

```python
async def handle_name_model_callback(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle 'Name Model' button click.

    Transitions state to NAMING_MODEL and prompts user for custom name.

    Callback data format: "name_model_{model_id}"
    """
    query = update.callback_query
    await query.answer()

    # Extract model_id from callback_data
    model_id = query.data.replace("name_model_", "")
    user_id = update.effective_user.id

    # Update state to NAMING_MODEL
    session = await self.state_manager.get_session(
        user_id=user_id,
        conversation_id=str(update.effective_chat.id)
    )
    session.current_state = MLTrainingState.NAMING_MODEL.value
    session.temp_data['pending_model_id'] = model_id
    await self.state_manager.update_session(session)

    # Prompt for custom name with examples
    await query.edit_message_text(
        "💾 **Give Your Model a Name**\n\n"
        "Enter a memorable name for your model:\n\n"
        "*Examples:*\n"
        "• Housing Price Predictor\n"
        "• Customer Churn Model v2\n"
        "• Spam Classifier - Production\n"
        "• Sentiment Analysis - Twitter\n\n"
        "*Rules:*\n"
        "• 3-100 characters\n"
        "• Letters, numbers, spaces, hyphens, underscores\n\n"
        "Type your name below:",
        parse_mode="Markdown"
    )
```

#### 4.3 Add Model Name Input Handler

**New Method**: `handle_model_name_input()`

```python
async def handle_model_name_input(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle user's model name text input.

    Validates name, saves to metadata, and sends confirmation.
    Only processes messages when in NAMING_MODEL state.
    """
    user_id = update.effective_user.id
    custom_name = update.message.text.strip()

    # Get session to verify state
    session = await self.state_manager.get_session(
        user_id=user_id,
        conversation_id=str(update.effective_chat.id)
    )

    # Only process if in NAMING_MODEL state
    if session.current_state != MLTrainingState.NAMING_MODEL.value:
        return  # Ignore message in wrong state

    # Get pending model_id
    model_id = session.temp_data.get('pending_model_id')
    if not model_id:
        await update.message.reply_text(
            "❌ Error: No pending model found. Please start training again."
        )
        return

    # Validate and set name
    try:
        success = self.ml_engine.set_model_name(
            user_id=user_id,
            model_id=model_id,
            custom_name=custom_name
        )

        if success:
            # Update state to MODEL_NAMED
            session.current_state = MLTrainingState.MODEL_NAMED.value
            await self.state_manager.update_session(session)

            # Get model info for confirmation
            model_info = self.ml_engine.get_model_info(user_id, model_id)
            created_date = datetime.fromisoformat(
                model_info['created_at'].replace('Z', '+00:00')
            ).strftime("%b %d, %Y")

            # Send success confirmation
            await update.message.reply_text(
                f"✅ **Model Named Successfully!**\n\n"
                f"📝 **Name:** {custom_name}\n"
                f"🆔 **Model ID:** `{model_id}`\n"
                f"📅 **Created:** {created_date}\n\n"
                f"You can now:\n"
                f"• Use `/predict` to make predictions\n"
                f"• View models with `/models`\n"
                f"• Train another with `/train`",
                parse_mode="Markdown"
            )

    except ValidationError as e:
        # Name validation failed
        await update.message.reply_text(
            f"❌ **Invalid Model Name**\n\n"
            f"{str(e)}\n\n"
            f"Please try again with a different name.",
            parse_mode="Markdown"
        )

    except Exception as e:
        # Unexpected error
        self.logger.error(f"Error setting model name: {e}")
        await update.message.reply_text(
            f"❌ **Error Saving Name**\n\n"
            f"An unexpected error occurred. Please try again or use `/train` to start over."
        )
```

#### 4.4 Add Skip Naming Callback Handler

**New Method**: `handle_skip_naming_callback()`

```python
async def handle_skip_naming_callback(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle 'Skip - Use Default' button click.

    Generates and sets default model name, then confirms to user.

    Callback data format: "skip_naming_{model_id}"
    """
    query = update.callback_query
    await query.answer()

    # Extract model_id
    model_id = query.data.replace("skip_naming_", "")
    user_id = update.effective_user.id

    # Get model info to generate default name
    model_info = self.ml_engine.get_model_info(user_id, model_id)

    # Generate default name
    default_name = self.ml_engine._generate_default_name(
        model_type=model_info['model_type'],
        task_type=model_info['task_type'],
        created_at=model_info['created_at']
    )

    # Set default name
    self.ml_engine.set_model_name(
        user_id=user_id,
        model_id=model_id,
        custom_name=default_name
    )

    # Update state to MODEL_NAMED
    session = await self.state_manager.get_session(
        user_id=user_id,
        conversation_id=str(update.effective_chat.id)
    )
    session.current_state = MLTrainingState.MODEL_NAMED.value
    await self.state_manager.update_session(session)

    # Format created date
    created_date = datetime.fromisoformat(
        model_info['created_at'].replace('Z', '+00:00')
    ).strftime("%b %d, %Y")

    # Send confirmation
    await query.edit_message_text(
        f"✅ **Model Saved!**\n\n"
        f"📝 **Name:** {default_name}\n"
        f"🆔 **Model ID:** `{model_id}`\n"
        f"📅 **Created:** {created_date}\n\n"
        f"You can now use this model for predictions with `/predict`.",
        parse_mode="Markdown"
    )
```

#### 4.5 Register Handlers in telegram_bot.py

**File**: `src/bot/telegram_bot.py`

**Add to Application Setup**:

```python
def main():
    """Start the bot."""
    # ... existing setup ...

    # ML Training Handler instance
    ml_training_handler = MLTrainingHandler(...)

    # ... existing handlers ...

    # NEW: Model naming callback handlers
    app.add_handler(CallbackQueryHandler(
        ml_training_handler.handle_name_model_callback,
        pattern="^name_model_"
    ))

    app.add_handler(CallbackQueryHandler(
        ml_training_handler.handle_skip_naming_callback,
        pattern="^skip_naming_"
    ))

    # NEW: Model name input handler (only in NAMING_MODEL state)
    async def model_name_filter(update: Update) -> bool:
        """Filter to only process messages in NAMING_MODEL state."""
        if not update.message or not update.message.text:
            return False

        user_id = update.effective_user.id
        state_manager = ... # Get state manager instance
        session = await state_manager.get_session(user_id, ...)

        return session.current_state == MLTrainingState.NAMING_MODEL.value

    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        ml_training_handler.handle_model_name_input
        # Filter applied: only processes in NAMING_MODEL state
    ))

    # ... rest of setup ...
```

---

### Phase 5: UI Display Updates (1 hour)

#### 5.1 Update /models Command

**File**: Handler for `/models` command

**Current Display**:
```
🤖 Your Trained Models (3)

1. Model: linear
   • Type: regression
   • R²: 0.85
   • Created: 2025-01-14
   • ID: model_12345_linear_20251014_123456
```

**New Display** (with custom names):
```
🤖 **Your Trained Models** (3)

1. 📊 **"Housing Price Predictor"**
   • Type: Random Forest (Regression)
   • R²: 0.85
   • Created: Jan 14, 2025
   • Model ID: `model_12345_rf_20251014...`

2. 🎯 **"Customer Churn Model v2"**
   • Type: Logistic Regression (Classification)
   • Accuracy: 89.2%
   • Created: Jan 12, 2025
   • Model ID: `model_12345_logistic_20251012...`

3. 📧 **"Binary Classification - Jan 10, 2025"**
   • Type: Neural Network (Classification)
   • Accuracy: 95.1%
   • Created: Jan 10, 2025
   • Model ID: `model_12345_keras_20251010...`
   • *(Default name - not customized)*
```

**Implementation**:
```python
async def handle_models_command(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Display user's trained models with custom names."""
    user_id = update.effective_user.id

    # Get models (now includes display_name)
    models = self.ml_engine.list_models(user_id)

    if not models:
        await update.message.reply_text(
            "You don't have any trained models yet.\n\n"
            "Use /train to create your first model!"
        )
        return

    # Build message with custom names
    text = f"🤖 **Your Trained Models** ({len(models)})\n\n"

    for i, model in enumerate(models, 1):
        # Get display name (custom or default)
        display_name = model['display_name']
        is_custom = model.get('custom_name') is not None

        # Select emoji based on task type
        emoji = "📊" if model['task_type'] == 'regression' else "🎯"

        # Format model entry
        text += f"{i}. {emoji} **\"{display_name}\"**\n"
        text += f"   • Type: {model['model_type'].replace('_', ' ').title()}"
        text += f" ({model['task_type'].title()})\n"

        # Add primary metric
        metrics = model.get('metrics', {})
        if 'r2' in metrics:
            text += f"   • R²: {metrics['r2']:.3f}\n"
        elif 'accuracy' in metrics:
            text += f"   • Accuracy: {metrics['accuracy']:.1%}\n"

        # Format date
        created_date = datetime.fromisoformat(
            model['created_at'].replace('Z', '+00:00')
        ).strftime("%b %d, %Y")
        text += f"   • Created: {created_date}\n"

        # Show truncated model ID
        model_id = model['model_id']
        text += f"   • Model ID: `{model_id[:40]}...`\n"

        # Indicate if default name
        if not is_custom:
            text += f"   • *(Default name - not customized)*\n"

        text += "\n"

    text += "Use `/predict` to make predictions with these models."

    await update.message.reply_text(text, parse_mode="Markdown")
```

#### 5.2 Update /predict Model Selection

**File**: Prediction workflow handler

**Current Model List**:
```
🤖 Select a Trained Model

Available models (3):

1. linear (regression)
   • R²: 0.85

2. logistic (classification)
   • Accuracy: 89.2%
```

**New Model List** (with custom names):
```
🤖 **Select a Trained Model**

Available models (3):

1. **"Housing Price Predictor"**
   • Random Forest (Regression)
   • Features: sqft, bedrooms, bathrooms
   • R²: 0.85

2. **"Customer Churn Model v2"**
   • Logistic Regression (Classification)
   • Features: age, tenure, monthly_spend
   • Accuracy: 89.2%

3. **"Binary Classification - Jan 10, 2025"**
   • Neural Network (Classification)
   • Features: 20 features
   • Accuracy: 95.1%

[Select Model]
```

**Implementation Enhancement**:
```python
def _format_model_selection(self, models: List[Dict[str, Any]]) -> str:
    """Format model list for selection UI."""
    text = "🤖 **Select a Trained Model**\n\n"
    text += f"Available models ({len(models)}):\n\n"

    for i, model in enumerate(models, 1):
        # Use display_name (custom or default)
        display_name = model['display_name']

        text += f"{i}. **\"{display_name}\"**\n"

        # Model type
        model_type_display = model['model_type'].replace('_', ' ').title()
        task_type_display = model['task_type'].title()
        text += f"   • {model_type_display} ({task_type_display})\n"

        # Features
        n_features = model.get('n_features', 0)
        if n_features <= 5:
            features_str = ", ".join(model.get('feature_names', []))
        else:
            features_str = f"{n_features} features"
        text += f"   • Features: {features_str}\n"

        # Primary metric
        metrics = model.get('metrics', {})
        if 'r2' in metrics:
            text += f"   • R²: {metrics['r2']:.3f}\n"
        elif 'accuracy' in metrics:
            text += f"   • Accuracy: {metrics['accuracy']:.1%}\n"

        text += "\n"

    return text
```

---

## 5. Testing Strategy

### Unit Tests (1 hour)

**File**: `tests/unit/test_ml_engine_naming.py`

```python
"""Unit tests for ML Engine model naming functionality."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
from src.utils.exceptions import ValidationError, ModelNotFoundError


class TestModelNaming:
    """Test model naming functionality."""

    @pytest.fixture
    def ml_engine(self):
        """Create ML engine instance."""
        config = MLEngineConfig.get_default()
        return MLEngine(config)

    @pytest.fixture
    def sample_model(self, ml_engine, tmp_path):
        """Create a sample model for testing."""
        # Create model directory
        user_dir = tmp_path / "models" / "user_12345"
        model_dir = user_dir / "model_12345_test_20251014"
        model_dir.mkdir(parents=True)

        # Create metadata.json
        metadata = {
            "model_id": "model_12345_test_20251014",
            "user_id": 12345,
            "model_type": "linear",
            "task_type": "regression",
            "created_at": "2025-01-14T12:00:00Z",
            "metrics": {"r2": 0.85}
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        return "model_12345_test_20251014", model_dir

    def test_set_model_name_success(self, ml_engine, sample_model):
        """Test successfully setting a custom model name."""
        model_id, model_dir = sample_model

        # Set custom name
        success = ml_engine.set_model_name(
            user_id=12345,
            model_id=model_id,
            custom_name="My Test Model"
        )

        assert success is True

        # Verify metadata was updated
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        assert metadata['custom_name'] == "My Test Model"
        assert metadata['display_name'] == "My Test Model"

    def test_set_model_name_invalid_too_short(self, ml_engine, sample_model):
        """Test validation rejects names that are too short."""
        model_id, _ = sample_model

        with pytest.raises(ValidationError) as exc:
            ml_engine.set_model_name(12345, model_id, "ab")

        assert "at least 3 characters" in str(exc.value)

    def test_set_model_name_invalid_too_long(self, ml_engine, sample_model):
        """Test validation rejects names that are too long."""
        model_id, _ = sample_model
        long_name = "a" * 101

        with pytest.raises(ValidationError) as exc:
            ml_engine.set_model_name(12345, model_id, long_name)

        assert "less than 100 characters" in str(exc.value)

    def test_set_model_name_invalid_characters(self, ml_engine, sample_model):
        """Test validation rejects names with invalid characters."""
        model_id, _ = sample_model

        invalid_names = [
            "model/test",      # Forward slash
            "model\\test",     # Backslash
            "model:test",      # Colon
            "model;test",      # Semicolon
            "model<test>",     # Angle brackets
        ]

        for invalid_name in invalid_names:
            with pytest.raises(ValidationError) as exc:
                ml_engine.set_model_name(12345, model_id, invalid_name)

            assert "can only contain" in str(exc.value)

    def test_set_model_name_valid_characters(self, ml_engine, sample_model):
        """Test validation accepts names with valid characters."""
        model_id, model_dir = sample_model

        valid_names = [
            "My Model",
            "Model-v2",
            "Model_test",
            "Model 123",
            "My-Model_v2 Test"
        ]

        for valid_name in valid_names:
            success = ml_engine.set_model_name(12345, model_id, valid_name)
            assert success is True

    def test_get_model_by_name_found(self, ml_engine, sample_model):
        """Test retrieving model by custom name."""
        model_id, _ = sample_model

        # Set name first
        ml_engine.set_model_name(12345, model_id, "Test Model")

        # Retrieve by name
        model = ml_engine.get_model_by_name(12345, "Test Model")

        assert model is not None
        assert model['model_id'] == model_id
        assert model['custom_name'] == "Test Model"

    def test_get_model_by_name_not_found(self, ml_engine):
        """Test retrieving non-existent model name."""
        model = ml_engine.get_model_by_name(12345, "Nonexistent Model")
        assert model is None

    def test_get_model_by_name_duplicate_returns_most_recent(
        self, ml_engine, tmp_path
    ):
        """Test that duplicate names return the most recent model."""
        # Create two models with same name but different dates
        user_dir = tmp_path / "models" / "user_12345"

        # Model 1 (older)
        model1_dir = user_dir / "model_12345_test_20251010"
        model1_dir.mkdir(parents=True)
        metadata1 = {
            "model_id": "model_12345_test_20251010",
            "custom_name": "Duplicate Name",
            "created_at": "2025-01-10T12:00:00Z"
        }
        with open(model1_dir / "metadata.json", 'w') as f:
            json.dump(metadata1, f)

        # Model 2 (newer)
        model2_dir = user_dir / "model_12345_test_20251014"
        model2_dir.mkdir(parents=True)
        metadata2 = {
            "model_id": "model_12345_test_20251014",
            "custom_name": "Duplicate Name",
            "created_at": "2025-01-14T12:00:00Z"
        }
        with open(model2_dir / "metadata.json", 'w') as f:
            json.dump(metadata2, f)

        # Should return newer model
        model = ml_engine.get_model_by_name(12345, "Duplicate Name")
        assert model['model_id'] == "model_12345_test_20251014"

    def test_generate_default_name_format(self, ml_engine):
        """Test default name generation format."""
        name = ml_engine._generate_default_name(
            model_type="keras_binary_classification",
            task_type="classification",
            created_at="2025-01-14T21:44:00Z"
        )

        assert "Binary Classification" in name
        assert "Jan 14, 2025" in name

    def test_list_models_includes_display_names(self, ml_engine, sample_model):
        """Test that list_models includes display_name field."""
        model_id, _ = sample_model

        # Set custom name
        ml_engine.set_model_name(12345, model_id, "Named Model")

        # List models
        models = ml_engine.list_models(12345)

        assert len(models) > 0
        model = models[0]
        assert 'display_name' in model
        assert model['display_name'] == "Named Model"

    def test_list_models_generates_default_for_unnamed(
        self, ml_engine, sample_model
    ):
        """Test that list_models generates default names for unnamed models."""
        models = ml_engine.list_models(12345)

        # Model has no custom name, should have generated display name
        model = models[0]
        assert 'display_name' in model
        assert model.get('custom_name') is None
        # Should have default format
        assert " - " in model['display_name']
        assert "2025" in model['display_name']
```

### Integration Tests (1 hour)

**File**: `tests/integration/test_model_naming_workflow.py`

```python
"""Integration tests for model naming workflow."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from telegram import Update, CallbackQuery, InlineKeyboardMarkup
from src.bot.ml_handlers.ml_training_local_path import MLTrainingHandler
from src.core.state_manager import StateManager, MLTrainingState


@pytest.mark.asyncio
class TestModelNamingWorkflow:
    """Test complete model naming workflow through Telegram."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = AsyncMock(spec=Update)
        update.effective_user.id = 12345
        update.effective_chat.id = 67890
        update.message = AsyncMock()
        update.message.text = ""
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_callback_query(self, mock_update):
        """Create mock callback query."""
        query = AsyncMock(spec=CallbackQuery)
        query.data = ""
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        mock_update.callback_query = query
        return mock_update, query

    @pytest.fixture
    def ml_training_handler(self):
        """Create ML training handler with mocked dependencies."""
        state_manager = MagicMock(spec=StateManager)
        ml_engine = MagicMock()

        handler = MLTrainingHandler(
            state_manager=state_manager,
            ml_engine=ml_engine,
            data_loader=MagicMock()
        )

        return handler, state_manager, ml_engine

    async def test_training_complete_shows_naming_options(
        self, mock_update, ml_training_handler
    ):
        """Test that training completion shows naming buttons."""
        handler, state_manager, ml_engine = ml_training_handler

        # Mock training result
        result = {
            'success': True,
            'model_id': 'model_12345_test_20251014',
            'metrics': {'accuracy': 0.85}
        }

        # Call handler (simplified - would be called from training completion)
        # ... handler code would create keyboard ...

        # Verify state updated
        assert state_manager.update_session.called

        # Verify message contains naming options
        # This would be tested by checking the reply_markup

    async def test_name_model_button_prompts_for_name(
        self, mock_callback_query, ml_training_handler
    ):
        """Test clicking 'Name Model' button prompts for custom name."""
        mock_update, query = mock_callback_query
        handler, state_manager, ml_engine = ml_training_handler

        # Set callback data
        query.data = "name_model_model_12345_test_20251014"

        # Mock session
        session = MagicMock()
        session.temp_data = {}
        state_manager.get_session.return_value = session

        # Handle callback
        await handler.handle_name_model_callback(mock_update, MagicMock())

        # Verify state transition
        assert session.current_state == MLTrainingState.NAMING_MODEL.value

        # Verify prompt was sent
        query.edit_message_text.assert_called_once()
        call_args = query.edit_message_text.call_args[0][0]
        assert "Give Your Model a Name" in call_args
        assert "Examples:" in call_args

    async def test_custom_name_input_success(
        self, mock_update, ml_training_handler
    ):
        """Test successful custom name input."""
        handler, state_manager, ml_engine = ml_training_handler

        # Mock session in NAMING_MODEL state
        session = MagicMock()
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.temp_data = {'pending_model_id': 'model_12345_test'}
        state_manager.get_session.return_value = session

        # Mock ML engine success
        ml_engine.set_model_name.return_value = True
        ml_engine.get_model_info.return_value = {
            'model_id': 'model_12345_test',
            'created_at': '2025-01-14T12:00:00Z'
        }

        # User inputs custom name
        mock_update.message.text = "My Custom Model Name"

        # Handle input
        await handler.handle_model_name_input(mock_update, MagicMock())

        # Verify name was set
        ml_engine.set_model_name.assert_called_once_with(
            user_id=12345,
            model_id='model_12345_test',
            custom_name="My Custom Model Name"
        )

        # Verify state transition
        assert session.current_state == MLTrainingState.MODEL_NAMED.value

        # Verify success message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Model Named Successfully" in call_args
        assert "My Custom Model Name" in call_args

    async def test_custom_name_input_validation_error(
        self, mock_update, ml_training_handler
    ):
        """Test custom name validation error handling."""
        handler, state_manager, ml_engine = ml_training_handler

        # Mock session
        session = MagicMock()
        session.current_state = MLTrainingState.NAMING_MODEL.value
        session.temp_data = {'pending_model_id': 'model_12345_test'}
        state_manager.get_session.return_value = session

        # Mock validation error
        from src.utils.exceptions import ValidationError
        ml_engine.set_model_name.side_effect = ValidationError(
            "Name must be at least 3 characters"
        )

        # User inputs invalid name
        mock_update.message.text = "ab"

        # Handle input
        await handler.handle_model_name_input(mock_update, MagicMock())

        # Verify error message
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Invalid Model Name" in call_args
        assert "at least 3 characters" in call_args

    async def test_skip_naming_uses_default(
        self, mock_callback_query, ml_training_handler
    ):
        """Test clicking 'Skip' button uses default name."""
        mock_update, query = mock_callback_query
        handler, state_manager, ml_engine = ml_training_handler

        # Set callback data
        query.data = "skip_naming_model_12345_test"

        # Mock session
        session = MagicMock()
        state_manager.get_session.return_value = session

        # Mock model info
        ml_engine.get_model_info.return_value = {
            'model_id': 'model_12345_test',
            'model_type': 'linear',
            'task_type': 'regression',
            'created_at': '2025-01-14T12:00:00Z'
        }

        # Mock default name generation
        ml_engine._generate_default_name.return_value = (
            "Linear Regression - Jan 14, 2025"
        )

        # Handle callback
        await handler.handle_skip_naming_callback(mock_update, MagicMock())

        # Verify default name was set
        ml_engine.set_model_name.assert_called_once()
        call_args = ml_engine.set_model_name.call_args
        assert call_args[1]['custom_name'] == "Linear Regression - Jan 14, 2025"

        # Verify state transition
        assert session.current_state == MLTrainingState.MODEL_NAMED.value

        # Verify confirmation
        query.edit_message_text.assert_called_once()
        call_args = query.edit_message_text.call_args[0][0]
        assert "Model Saved" in call_args
        assert "Linear Regression - Jan 14, 2025" in call_args
```

### Manual Testing Checklist

**Preparation**:
- [ ] Clear existing models or create test user
- [ ] Prepare test dataset for training
- [ ] Have bot running and responsive

**Happy Path**:
- [ ] Train a model successfully
- [ ] See naming options after training completes
- [ ] Click "Name Model" button
- [ ] Enter custom name "Test Model 1"
- [ ] Receive success confirmation
- [ ] Check /models - verify custom name appears
- [ ] Use /predict - verify custom name in model selection

**Skip Naming**:
- [ ] Train another model
- [ ] Click "Skip - Use Default"
- [ ] Verify default name format (e.g., "Linear Regression - Jan 14, 2025")
- [ ] Check /models - verify default name appears

**Validation**:
- [ ] Try name with 2 characters → should reject
- [ ] Try name with special characters → should reject
- [ ] Try name with valid characters → should accept
- [ ] Try duplicate name → should accept with warning

**UI Display**:
- [ ] /models shows custom names prominently
- [ ] /models distinguishes default vs custom names
- [ ] /predict model selection uses custom names
- [ ] Model IDs still accessible for technical reference

---

## 6. Migration & Rollout

### Existing Models Compatibility

**Automatic Migration**:
- No manual migration needed
- Existing models without `custom_name` field work normally
- `list_models()` generates display names on-the-fly for legacy models
- Default name format applied automatically

**Migration Behavior**:
```python
# Old metadata.json (no custom_name)
{
    "model_id": "model_12345_linear_20251010",
    "model_type": "linear",
    "created_at": "2025-01-10T12:00:00Z"
}

# Displayed as:
# "Linear Regression - Jan 10, 2025"
# (Generated at runtime, not stored)
```

### Backward Compatibility

**API Compatibility**:
- ✅ `list_models()` - Returns additional `display_name` field (non-breaking)
- ✅ `get_model_info()` - Returns additional `custom_name` field (non-breaking)
- ✅ `train_model()` - No changes to signature
- ✅ `predict()` - No changes to signature

**Data Compatibility**:
- ✅ Old metadata.json files work without modification
- ✅ New fields optional, not required
- ✅ No database migrations needed

### Rollout Plan

**Phase 1: Infrastructure (Days 1-2)**
1. Implement MLEngine methods (`set_model_name`, `get_model_by_name`, validation)
2. Update model metadata schema (add fields)
3. Add state machine states and transitions
4. Write unit tests for core functionality

**Phase 2: Telegram Integration (Days 3-4)**
1. Implement training completion handler updates
2. Add name model callback handler
3. Add model name input handler
4. Add skip naming callback handler
5. Register handlers in telegram_bot.py
6. Write integration tests

**Phase 3: UI Updates (Day 5)**
1. Update /models command display
2. Update /predict model selection display
3. Test UI in staging environment

**Phase 4: Testing & QA (Days 6-7)**
1. Run full test suite
2. Manual testing on staging
3. Fix any bugs discovered
4. Performance testing

**Phase 5: Production Deploy (Day 8)**
1. Deploy to production
2. Monitor logs for errors
3. Gather user feedback
4. Hot-fix if critical issues found

### Risk Mitigation

**Risk 1: State Machine Conflicts**
- *Mitigation*: Extensive state transition testing
- *Rollback*: Can disable naming feature without breaking training

**Risk 2: Name Validation Issues**
- *Mitigation*: Comprehensive validation test cases
- *Fallback*: Allow all names if validation fails (log warning)

**Risk 3: Performance Impact**
- *Mitigation*: Name operations are file I/O only, no heavy compute
- *Monitoring*: Track response times for /models command

---

## 7. Future Enhancements

### Priority 1: Near-Term (Next 2-4 weeks)

**1. Model Renaming**
- New command: `/rename`
- Workflow: Select model → Enter new name → Confirm
- Preserve model history/versioning

**2. Search by Name**
- Filter models in /models by name substring
- Example: `/models search housing` → shows all models with "housing" in name

**3. Batch Operations**
- Rename multiple models at once
- Delete models by name pattern

### Priority 2: Medium-Term (1-3 months)

**4. Name Suggestions**
- AI-generated name suggestions based on:
  - Task type
  - Features used
  - Model type
  - Performance metrics
- Example: "High-Accuracy Classification (95.1%) on Customer Data"

**5. Name History**
- Track all name changes for a model
- View rename history
- Restore previous name

**6. Emoji Support**
- Allow emojis in model names for visual categorization
- Example: "🏠 Housing Price Predictor", "📧 Email Spam Filter"

### Priority 3: Long-Term (3-6 months)

**7. Model Collections/Tags**
- Group models into collections
- Example: "Production Models", "Experimental", "Version 2.0"

**8. Name Templates**
- Pre-defined naming templates for consistency
- Example: "{ModelType} - {Dataset} - {Date}"

**9. Advanced Search**
- Search by name, date range, metrics, tags
- Sort by relevance, date, performance

**10. Collaboration Features**
- Share named models between users
- Team model libraries with shared naming conventions

---

## Appendix A: Code Snippets

### Example Usage in Client Code

```python
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig

# Initialize
ml_engine = MLEngine(MLEngineConfig.get_default())
user_id = 12345

# Train model (existing workflow)
result = ml_engine.train_model(
    data=training_data,
    task_type='classification',
    model_type='logistic',
    target_column='churn',
    feature_columns=['age', 'tenure', 'spend'],
    user_id=user_id
)

model_id = result['model_id']

# NEW: Set custom name
ml_engine.set_model_name(
    user_id=user_id,
    model_id=model_id,
    custom_name="Customer Churn Predictor v1"
)

# NEW: Retrieve by name
model = ml_engine.get_model_by_name(user_id, "Customer Churn Predictor v1")
print(f"Found model: {model['model_id']}")

# List models (now includes display_name)
models = ml_engine.list_models(user_id)
for model in models:
    print(f"{model['display_name']} - {model['metrics'].get('accuracy', 'N/A')}")
```

---

## Appendix B: Configuration

### No Configuration Changes Required

The model naming feature requires no configuration file changes. All behavior is built into the code.

**Optional Configuration** (for future):
```yaml
# config/config.yaml

model_naming:
  enabled: true
  allow_duplicates: true
  max_name_length: 100
  min_name_length: 3
  auto_generate_default: true
  default_name_format: "{model_type} - {date}"
```

---

## Appendix C: Error Messages

### User-Facing Error Messages

**Validation Errors**:
```
❌ Invalid Model Name

Name must be at least 3 characters

Please try again with a different name.
```

```
❌ Invalid Model Name

Name can only contain letters, numbers, spaces, hyphens, and underscores

Please try again with a different name.
```

**System Errors**:
```
❌ Error Saving Name

An unexpected error occurred. Please try again or use /train to start over.
```

```
❌ Error: No pending model found.

Please start training again with /train.
```

---

## Appendix D: Related Documentation

- `ml-engine.md` - ML Engine architecture and API reference
- `file-path-training.md` - Local path training workflow
- `predict-workflow.md` - ML prediction workflow specification
- `state_manager.py` - State machine implementation
- `telegram_bot.py` - Bot handler registration

---

**Document Version**: 1.0
**Last Updated**: 2025-10-13
**Author**: Claude (AI Assistant)
**Status**: Ready for Implementation Review