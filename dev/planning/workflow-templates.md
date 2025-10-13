# ML Training Templates System - Implementation Plan

## Overview
Allow users to save and reuse complete ML training configurations (file path, schema, model selection, hyperparameters) as named templates.

## User Workflows

### Workflow 1: Create Template
1. User starts `/train` and completes full configuration:
   - Select data source (Local Path)
   - Provide file path
   - Confirm schema (target + features)
   - Select model category and type
   - Configure hyperparameters
2. **Before training starts**, bot shows new button: "💾 Save as Template"
3. User clicks button → Bot prompts: "Enter a name for this template:"
4. User provides name (e.g., "housing_rf_model")
5. Bot validates name → saves template → confirms: "✅ Template 'housing_rf_model' saved!"
6. Bot continues to training (or user can cancel)

### Workflow 2: Retrieve Template
1. User starts `/train`
2. Bot shows data source selection with **new option**: "📋 Use Template"
3. User selects "Use Template"
4. Bot displays list of user's templates as buttons
5. User clicks template (e.g., "housing_rf_model")
6. Bot loads template → displays configuration summary
7. Bot asks: "Load data now or defer?" (respect deferred loading)
8. If load now: validates path → loads data → continues to training
9. If defer: saves config → waits for `/continue` to load data

## Architecture

### Data Structure

```python
@dataclass
class TrainingTemplate:
    """Complete ML training configuration template."""
    template_id: str  # "tmpl_{user_id}_{sanitized_name}_{timestamp}"
    template_name: str  # User-provided name
    user_id: int

    # Data configuration
    file_path: str  # Absolute path to data file

    # Schema configuration
    target_column: str
    feature_columns: List[str]

    # Model configuration
    model_category: str  # "regression", "classification", "neural_network"
    model_type: str  # "random_forest", "linear", "keras_binary_classification", etc.

    # Training configuration
    hyperparameters: Dict[str, Any]  # Model-specific hyperparameters

    # Metadata
    created_at: str  # ISO format timestamp
    last_used: Optional[str]  # ISO format timestamp
    description: Optional[str]  # Optional user description
```

### Storage Structure

```
templates/
├── user_12345/
│   ├── housing_rf_model.json
│   ├── credit_logistic.json
│   └── sales_neural_net.json
└── user_67890/
    └── customer_clustering.json
```

Each template JSON file contains the TrainingTemplate dataclass serialized.

### Core Module: `src/core/template_manager.py`

```python
class TemplateManager:
    """Manage ML training templates with CRUD operations."""

    def __init__(self, config: TemplateConfig):
        self.templates_dir = Path(config.templates_dir)
        self.max_templates_per_user = config.max_templates_per_user
        self.name_pattern = config.allowed_name_pattern
        self.name_max_length = config.name_max_length

    def save_template(
        self,
        user_id: int,
        template_name: str,
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Save a new template.

        Returns:
            (success: bool, message: str)
        """
        pass

    def load_template(
        self,
        user_id: int,
        template_name: str
    ) -> Optional[TrainingTemplate]:
        """Load template by name."""
        pass

    def list_templates(
        self,
        user_id: int
    ) -> List[TrainingTemplate]:
        """List all templates for user, sorted by last_used desc."""
        pass

    def delete_template(
        self,
        user_id: int,
        template_name: str
    ) -> bool:
        """Delete template by name."""
        pass

    def rename_template(
        self,
        user_id: int,
        old_name: str,
        new_name: str
    ) -> Tuple[bool, str]:
        """Rename existing template."""
        pass

    def validate_template_name(self, name: str) -> Tuple[bool, str]:
        """
        Validate template name.

        Rules:
        - Only alphanumeric and underscore
        - Max length from config
        - Not empty

        Returns:
            (is_valid: bool, error_message: str)
        """
        pass

    def template_exists(self, user_id: int, name: str) -> bool:
        """Check if template exists."""
        pass
```

## State Machine Updates

### New State

Add to `MLTrainingState` enum in `src/core/state_manager.py`:

```python
class MLTrainingState(Enum):
    # ... existing states ...
    SAVING_TEMPLATE = "saving_template"  # NEW
    LOADING_TEMPLATE = "loading_template"  # NEW
    CONFIRMING_TEMPLATE = "confirming_template"  # NEW
```

### New Transitions

Update `ML_TRAINING_TRANSITIONS` dictionary:

```python
ML_TRAINING_TRANSITIONS = {
    # ... existing transitions ...

    # Template save workflow (from hyperparameters complete)
    MLTrainingState.COLLECTING_HYPERPARAMETERS: [
        MLTrainingState.SAVING_TEMPLATE,  # User clicks "Save as Template"
        MLTrainingState.TRAINING  # User proceeds to training
    ],

    MLTrainingState.SAVING_TEMPLATE: [
        MLTrainingState.TRAINING,  # After saving, continue to training
        MLTrainingState.COMPLETE  # User cancels after saving
    ],

    # Template load workflow (from data source selection)
    MLTrainingState.CHOOSING_DATA_SOURCE: [
        # ... existing transitions ...
        MLTrainingState.LOADING_TEMPLATE  # User selects "Use Template"
    ],

    MLTrainingState.LOADING_TEMPLATE: [
        MLTrainingState.CONFIRMING_TEMPLATE  # After selecting template
    ],

    MLTrainingState.CONFIRMING_TEMPLATE: [
        MLTrainingState.CHOOSING_LOAD_OPTION,  # Proceed with template config
        MLTrainingState.LOADING_TEMPLATE  # Go back to template list
    ]
}
```

## Integration Points

### 1. Save Template Entry Point

**File**: `src/bot/ml_handlers/ml_training_local_path.py`

**Location**: After hyperparameter collection completes

**Modification**: In `handle_model_selection()` or after Keras architecture configuration, add button:

```python
keyboard = [
    [InlineKeyboardButton("💾 Save as Template", callback_data="template_save")],
    [InlineKeyboardButton("🚀 Start Training", callback_data="start_training")],
    [InlineKeyboardButton("🔙 Back", callback_data="back")]
]
```

**New Handler**: `handle_template_save_request()`

```python
async def handle_template_save_request(
    self,
    update: Update,
    context: CallbackContext
) -> None:
    """Handle 'Save as Template' button click."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    session = self.state_manager.get_session(user_id, query.message.chat_id)

    # Transition to SAVING_TEMPLATE state
    self.state_manager.transition_state(
        user_id,
        query.message.chat_id,
        MLTrainingState.SAVING_TEMPLATE
    )

    await query.edit_message_text(
        "📝 Enter a name for this template:\n\n"
        "Rules:\n"
        "• Only letters, numbers, and underscores\n"
        "• Maximum 32 characters\n"
        "• Must be unique\n\n"
        "Example: housing_rf_model",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("❌ Cancel", callback_data="cancel_template")
        ]])
    )
```

**New Handler**: `handle_template_name_input()`

```python
async def handle_template_name_input(
    self,
    update: Update,
    context: CallbackContext
) -> None:
    """Handle template name text input."""
    user_id = update.effective_user.id
    conversation_id = update.message.chat_id
    session = self.state_manager.get_session(user_id, conversation_id)

    if session.current_state != MLTrainingState.SAVING_TEMPLATE.value:
        return

    template_name = update.message.text.strip()

    # Validate name
    is_valid, error_msg = self.template_manager.validate_template_name(template_name)
    if not is_valid:
        await update.message.reply_text(
            f"❌ Invalid template name: {error_msg}\n\n"
            "Please try again:"
        )
        return

    # Check if exists
    if self.template_manager.template_exists(user_id, template_name):
        await update.message.reply_text(
            f"⚠️ Template '{template_name}' already exists.\n\n"
            "Choose a different name or delete the existing template first."
        )
        return

    # Build template config from session
    template_config = {
        "file_path": session.file_path,
        "target_column": session.selections.get("target_column"),
        "feature_columns": session.selections.get("feature_columns"),
        "model_category": session.selections.get("model_category"),
        "model_type": session.selections.get("model_type"),
        "hyperparameters": session.selections.get("hyperparameters", {})
    }

    # Save template
    success, message = self.template_manager.save_template(
        user_id=user_id,
        template_name=template_name,
        config=template_config
    )

    if success:
        await update.message.reply_text(
            f"✅ Template '{template_name}' saved successfully!\n\n"
            "You can now use this template for future training sessions."
        )

        # Offer to continue training or finish
        keyboard = [
            [InlineKeyboardButton("🚀 Start Training Now", callback_data="start_training")],
            [InlineKeyboardButton("✅ Done (Exit)", callback_data="complete")]
        ]
        await update.message.reply_text(
            "What would you like to do next?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text(
            f"❌ Failed to save template: {message}"
        )
```

### 2. Retrieve Template Entry Point

**File**: `src/bot/ml_handlers/ml_training_local_path.py`

**Location**: `handle_start_training()` - data source selection

**Modification**: Add "Use Template" button to data source selection:

```python
async def handle_start_training(self, update: Update, context: CallbackContext):
    # ... existing code ...

    keyboard = [
        [InlineKeyboardButton("📤 Upload via Telegram", callback_data="source_telegram")],
        [InlineKeyboardButton("📁 Local File Path", callback_data="source_local")],
        [InlineKeyboardButton("📋 Use Template", callback_data="source_template")],  # NEW
    ]

    # ... rest of existing code ...
```

**New Handler**: `handle_template_source_selection()`

```python
async def handle_template_source_selection(
    self,
    update: Update,
    context: CallbackContext
) -> None:
    """Handle 'Use Template' data source selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    conversation_id = query.message.chat_id

    # Transition to LOADING_TEMPLATE state
    self.state_manager.transition_state(
        user_id,
        conversation_id,
        MLTrainingState.LOADING_TEMPLATE
    )

    # Get user's templates
    templates = self.template_manager.list_templates(user_id)

    if not templates:
        await query.edit_message_text(
            "📋 No templates found.\n\n"
            "Create a template by completing a training workflow "
            "and clicking 'Save as Template' before training starts."
        )
        return

    # Display templates as buttons
    keyboard = []
    for template in templates:
        # Show name and basic info
        button_text = f"📄 {template.template_name}"
        callback_data = f"load_template:{template.template_name}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back")])

    await query.edit_message_text(
        "📋 Select a template:\n\n"
        f"You have {len(templates)} saved template(s).",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
```

**New Handler**: `handle_template_selection()`

```python
async def handle_template_selection(
    self,
    update: Update,
    context: CallbackContext
) -> None:
    """Handle specific template selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    conversation_id = query.message.chat_id

    # Extract template name from callback_data
    template_name = query.data.split(":", 1)[1]

    # Load template
    template = self.template_manager.load_template(user_id, template_name)
    if not template:
        await query.edit_message_text(
            f"❌ Template '{template_name}' not found.\n\n"
            "It may have been deleted."
        )
        return

    # Update last_used timestamp
    template.last_used = datetime.now(timezone.utc).isoformat()
    self.template_manager.save_template(user_id, template_name, template.__dict__)

    # Populate session with template data
    session = self.state_manager.get_session(user_id, conversation_id)
    session.file_path = template.file_path
    session.selections["target_column"] = template.target_column
    session.selections["feature_columns"] = template.feature_columns
    session.selections["model_category"] = template.model_category
    session.selections["model_type"] = template.model_type
    session.selections["hyperparameters"] = template.hyperparameters

    # Display configuration summary
    features_str = ", ".join(template.feature_columns[:3])
    if len(template.feature_columns) > 3:
        features_str += f" ... (+{len(template.feature_columns) - 3} more)"

    summary = (
        f"📋 Template: *{template.template_name}*\n\n"
        f"📁 Data: `{template.file_path}`\n"
        f"🎯 Target: `{template.target_column}`\n"
        f"📊 Features: {features_str}\n"
        f"🤖 Model: {template.model_category} / {template.model_type}\n\n"
        f"Created: {template.created_at[:10]}"
    )

    # Transition to CONFIRMING_TEMPLATE
    self.state_manager.transition_state(
        user_id,
        conversation_id,
        MLTrainingState.CONFIRMING_TEMPLATE
    )

    # Ask about loading data
    keyboard = [
        [InlineKeyboardButton("📥 Load Data Now", callback_data="template_load_now")],
        [InlineKeyboardButton("⏳ Defer Loading", callback_data="template_defer")],
        [InlineKeyboardButton("🔙 Back to Templates", callback_data="back")]
    ]

    await query.edit_message_text(
        summary,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
```

**New Handler**: `handle_template_load_option()`

```python
async def handle_template_load_option(
    self,
    update: Update,
    context: CallbackContext
) -> None:
    """Handle template data loading option (now vs defer)."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    conversation_id = query.message.chat_id
    session = self.state_manager.get_session(user_id, conversation_id)

    if query.data == "template_load_now":
        session.load_deferred = False

        # Validate and load file
        file_path = session.file_path
        validation_result = self.path_validator.validate_path(file_path)

        if not validation_result["is_valid"]:
            await query.edit_message_text(
                f"❌ Invalid file path: {validation_result['error']}\n\n"
                "The template file path is no longer valid."
            )
            return

        try:
            # Load data
            df = self.data_loader.load_from_local_path(file_path)

            # Store in bot_data
            data_key = f"user_{user_id}_conv_{conversation_id}_data"
            context.bot_data[data_key] = df

            await query.edit_message_text(
                f"✅ Data loaded successfully!\n\n"
                f"Rows: {len(df)}\n"
                f"Columns: {len(df.columns)}\n\n"
                "Ready to train model."
            )

            # Transition to TRAINING
            self.state_manager.transition_state(
                user_id,
                conversation_id,
                MLTrainingState.TRAINING
            )

            # Start training
            await self.start_training(update, context)

        except Exception as e:
            await query.edit_message_text(
                f"❌ Error loading data: {str(e)}"
            )

    elif query.data == "template_defer":
        session.load_deferred = True

        await query.edit_message_text(
            "⏳ Data loading deferred.\n\n"
            "Configuration saved. Use /continue when ready to load data and train."
        )

        # Transition to COMPLETE
        self.state_manager.transition_state(
            user_id,
            conversation_id,
            MLTrainingState.COMPLETE
        )
```

## Configuration Updates

### File: `config/config.yaml`

Add new section:

```yaml
templates:
  enabled: true
  templates_dir: ./templates
  max_templates_per_user: 50
  allowed_name_pattern: "^[a-zA-Z0-9_]{1,32}$"
  name_max_length: 32
```

### Config Dataclass

Update `src/core/state_manager.py` or create `src/core/template_config.py`:

```python
@dataclass
class TemplateConfig:
    """Configuration for template system."""
    enabled: bool = True
    templates_dir: str = "./templates"
    max_templates_per_user: int = 50
    allowed_name_pattern: str = r"^[a-zA-Z0-9_]{1,32}$"
    name_max_length: int = 32

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TemplateConfig":
        """Create from config dictionary."""
        return cls(**config.get("templates", {}))
```

## UI Messages

### File: `src/bot/messages/template_messages.py`

```python
"""Template-related UI messages."""

TEMPLATE_SAVE_PROMPT = (
    "📝 Enter a name for this template:\n\n"
    "Rules:\n"
    "• Only letters, numbers, and underscores\n"
    "• Maximum 32 characters\n"
    "• Must be unique\n\n"
    "Example: housing_rf_model"
)

TEMPLATE_SAVED_SUCCESS = (
    "✅ Template '{name}' saved successfully!\n\n"
    "You can now use this template for future training sessions."
)

TEMPLATE_LOAD_SUMMARY = (
    "📋 Template: *{name}*\n\n"
    "📁 Data: `{file_path}`\n"
    "🎯 Target: `{target}`\n"
    "📊 Features: {features}\n"
    "🤖 Model: {category} / {type}\n\n"
    "Created: {created}"
)

TEMPLATE_NO_TEMPLATES = (
    "📋 No templates found.\n\n"
    "Create a template by completing a training workflow "
    "and clicking 'Save as Template' before training starts."
)

TEMPLATE_INVALID_NAME = (
    "❌ Invalid template name: {error}\n\n"
    "Please try again:"
)

TEMPLATE_EXISTS = (
    "⚠️ Template '{name}' already exists.\n\n"
    "Choose a different name or delete the existing template first."
)
```

## Testing Strategy

### Unit Tests: `tests/unit/test_template_manager.py`

```python
class TestTemplateManager:
    """Test template CRUD operations."""

    def test_save_template_success(self):
        """Test successful template save."""
        pass

    def test_save_template_invalid_name(self):
        """Test template save with invalid name."""
        pass

    def test_save_template_duplicate(self):
        """Test template save with duplicate name."""
        pass

    def test_load_template_success(self):
        """Test successful template load."""
        pass

    def test_load_template_not_found(self):
        """Test loading non-existent template."""
        pass

    def test_list_templates_empty(self):
        """Test listing with no templates."""
        pass

    def test_list_templates_multiple(self):
        """Test listing multiple templates."""
        pass

    def test_delete_template_success(self):
        """Test successful template deletion."""
        pass

    def test_validate_name_valid(self):
        """Test name validation with valid names."""
        pass

    def test_validate_name_invalid(self):
        """Test name validation with invalid names."""
        pass
```

### Integration Tests: `tests/integration/test_template_workflow.py`

```python
@pytest.mark.asyncio
class TestTemplateWorkflow:
    """Test complete template workflows."""

    async def test_save_and_load_template(self):
        """Test full save and load workflow."""
        pass

    async def test_template_with_deferred_loading(self):
        """Test template with deferred data loading."""
        pass

    async def test_template_list_display(self):
        """Test template list UI."""
        pass

    async def test_template_invalid_path(self):
        """Test template with invalid file path."""
        pass
```

## Implementation Phases

### Phase 1: Core Infrastructure (3-4 hours)
1. Create `src/core/training_template.py` with TrainingTemplate dataclass
2. Create `src/core/template_manager.py` with TemplateManager class
3. Add templates section to `config/config.yaml`
4. Create unit tests for TemplateManager

### Phase 2: UI Integration - Save Workflow (2-3 hours)
5. Add SAVING_TEMPLATE state to MLTrainingState enum
6. Add state transitions for SAVING_TEMPLATE
7. Create `src/bot/messages/template_messages.py`
8. Add "Save as Template" button after hyperparameter collection
9. Implement `handle_template_save_request()` handler
10. Implement `handle_template_name_input()` text handler

### Phase 3: UI Integration - Load Workflow (2-3 hours)
11. Add LOADING_TEMPLATE and CONFIRMING_TEMPLATE states
12. Add state transitions for template loading
13. Modify `handle_start_training()` to add "Use Template" button
14. Implement `handle_template_source_selection()` handler
15. Implement `handle_template_selection()` handler
16. Implement `handle_template_load_option()` handler

### Phase 4: Management Features (1-2 hours)
17. Add template deletion capability
18. Optional: Add `/templates` command for listing templates
19. Optional: Add template rename functionality

### Phase 5: Testing & Polish (2 hours)
20. Write integration tests
21. Test error cases (missing file, invalid name, etc.)
22. Add logging throughout
23. Update `CLAUDE.md` documentation

**Total Estimated Time**: 10-14 hours

## Security Considerations

1. **Name Sanitization**: Validate template names to prevent path traversal
2. **Path Validation**: Re-validate file paths when loading templates
3. **User Isolation**: Templates stored per-user, no cross-user access
4. **File System Limits**: Enforce max templates per user
5. **Input Validation**: Sanitize all user inputs (name, description)

## Future Enhancements

1. **Template Sharing**: Allow users to export/import templates
2. **Template Descriptions**: Add optional description field
3. **Template Tags**: Categorize templates with tags
4. **Template Search**: Search templates by name/description
5. **Template Versioning**: Track template modifications over time
6. **Template Analytics**: Track usage frequency and success rates
