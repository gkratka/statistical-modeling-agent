<objective>
Implement `/template` command system for the Telegram bot that allows users to:
1. Record ML workflows (/train or /predict) as reusable templates
2. Run recorded templates automatically with a single command
3. Manage templates (list, view, delete)

This enables users to automate repetitive ML workflows by saving configurations and replaying them.
</objective>

<context>
This is a Telegram bot for statistical modeling and ML. Users currently use:
- `/train` - Multi-step workflow to train ML models (data source → target → features → model type → hyperparameters → training)
- `/predict` - Multi-step workflow to run predictions (model selection → data source → output path)

Relevant files to examine:
- `src/bot/telegram_bot.py` - Command registration patterns
- `src/bot/handlers/ml_training_local_path.py` - /train workflow handlers
- `src/bot/ml_handlers/prediction_template_handlers.py` - /predict workflow handlers
- `src/core/state_manager.py` - State machine patterns
- `src/bot/messages/` - Message template patterns
</context>

<requirements>

## 1. Template Storage
- Store templates in `templates/<user_id>.json` (one JSON file per user)
- Template schema:
```json
{
  "TRAIN_CATBST_CLASS2": {
    "type": "train",
    "created_at": "2024-12-29T14:00:00Z",
    "config": {
      "file_path": "/path/to/data.csv",
      "defer_loading": true,
      "target_column": "class2",
      "feature_columns": ["col1", "col2"],
      "model_type": "catboost_binary_classification",
      "model_parameters": {"iterations": 100},
      "model_name": "my_catboost_model"
    }
  },
  "PREDICT_CATBST_CLASS2": {
    "type": "predict",
    "created_at": "2024-12-29T15:00:00Z",
    "config": {
      "model_id": "model_12345_catboost_...",
      "file_path": "/path/to/predict_data.csv",
      "output_path": "/path/to/predictions.csv"
    }
  }
}
```

## 2. Template Naming Convention
- **Train templates**: `TRAIN_<MODEL-NAME>_<TARGET-COLUMN>`
  - MODEL-NAME: max 8 characters, uppercase, alphanumeric
  - TARGET-COLUMN: max 8 characters, uppercase, alphanumeric
  - Example: `TRAIN_CATBST_CLASS2`

- **Predict templates**: `PREDICT_<MODEL-NAME>_<PREDICT-COLUMN>`
  - Same constraints as train
  - Example: `PREDICT_CATBST_CLASS2`

- Validation regex: `^(TRAIN|PREDICT)_[A-Z0-9]{1,8}_[A-Z0-9]{1,8}$`

## 3. Recording Templates

### For /train workflow:
- Add "Save as Template" button in training completion screen (below "Skip - Use Default" button for model naming)
- When clicked, prompt user to enter template name
- Validate name format, save all workflow parameters
- Show confirmation with saved template name

### For /predict workflow:
- Use existing "Save as Template" button shown after prediction completion
- When clicked, prompt user to enter template name
- Validate name format, save all workflow parameters
- Show confirmation with saved template name

## 4. /template Command Usage

### `/template` (no arguments)
Show help message with:
- Instructions on how to use templates
- Naming format explanation
- List of user's saved templates (grouped by type: TRAIN / PREDICT)
- Example: "You have 3 templates: TRAIN_CATBST_CLASS2, TRAIN_XGBST_PRICE, PREDICT_CATBST_CLASS2"

### `/template <name>`
- Validate template exists for user
- Load template config
- Execute the appropriate workflow automatically:
  - For TRAIN templates: Trigger training with saved config
  - For PREDICT templates: Trigger prediction with saved config
- Show progress and completion messages

### `/template delete <name>`
- Delete the specified template
- Show confirmation

### `/template list`
- Show detailed list of all templates with creation date and config summary

## 5. State Machine
Add states to state_manager.py:
- `AWAITING_TEMPLATE_NAME` - Waiting for user to input template name during recording
- `TEMPLATE_EXECUTING` - Template workflow is running

</requirements>

<implementation>

## File Structure
Create/modify these files:

1. **NEW: `src/bot/handlers/template_handlers.py`**
   - TemplateHandlers class with:
     - `handle_template_command()` - Main /template handler
     - `handle_template_execution()` - Execute a template
     - `handle_template_delete()` - Delete a template
     - `handle_template_list()` - List templates
     - `handle_save_template_button()` - Handle "Save as Template" button click
     - `handle_template_name_input()` - Receive and validate template name
     - `_validate_template_name()` - Regex validation
     - `_save_template()` - Persist to JSON file
     - `_load_templates()` - Load user's templates
     - `_delete_template()` - Remove from JSON file

2. **NEW: `src/bot/messages/template_messages.py`**
   - TemplateMessages class with all user-facing messages:
     - `template_help()` - Instructions when `/template` called with no args
     - `template_list()` - Format template list display
     - `template_name_prompt()` - Ask for template name
     - `template_saved()` - Confirmation after saving
     - `template_executing()` - Progress message during execution
     - `template_complete()` - Success message
     - `template_not_found()` - Error when template doesn't exist
     - `template_invalid_name()` - Name format validation error
     - `template_deleted()` - Deletion confirmation

3. **MODIFY: `src/bot/handlers/ml_training_local_path.py`**
   - In `_handle_model_name_response()` or training completion handler:
     - Add "Save as Template" button below "Skip - Use Default"
     - Handle button click to start template recording flow

4. **MODIFY: `src/bot/ml_handlers/prediction_template_handlers.py`**
   - Connect existing "Save as Template" button to template recording flow
   - Capture all prediction parameters for template

5. **MODIFY: `src/core/state_manager.py`**
   - Add template-related states to appropriate workflow enums

6. **MODIFY: `src/bot/telegram_bot.py`**
   - Register `/template` command handler
   - Register callback patterns for template buttons

7. **NEW: `templates/` directory**
   - Will contain `<user_id>.json` files (created automatically)

## Key Implementation Details

### Template Recording Flow (Train):
1. Training completes → Show "Save as Template" button
2. User clicks button → Bot sends "Enter template name (format: TRAIN_XXXXXXXX_XXXXXXXX):"
3. User types name → Validate format
4. If valid → Save template to `templates/<user_id>.json`
5. Confirm: "Template TRAIN_CATBST_CLASS2 saved! Use `/template TRAIN_CATBST_CLASS2` to run."

### Template Execution Flow:
1. User sends `/template TRAIN_CATBST_CLASS2`
2. Load template from JSON
3. If train template:
   - Call existing training job submission with saved parameters
   - Skip all interactive steps
4. If predict template:
   - Call existing prediction job submission with saved parameters
   - Skip all interactive steps
5. Show progress → Show completion

### Error Handling:
- Template not found → Suggest `/template` to see available templates
- Invalid name format → Show correct format with examples
- File path no longer exists → Error message with suggestion to re-record
- Model ID no longer exists (predict) → Error message

</implementation>

<output>
Create/modify files with relative paths:
- `./src/bot/handlers/template_handlers.py` - New handler class
- `./src/bot/messages/template_messages.py` - New messages class
- `./src/bot/handlers/ml_training_local_path.py` - Add template save button
- `./src/bot/ml_handlers/prediction_template_handlers.py` - Connect template save
- `./src/core/state_manager.py` - Add template states
- `./src/bot/telegram_bot.py` - Register template command and callbacks
- `./src/bot/messages/__init__.py` - Export TemplateMessages
</output>

<verification>
Before declaring complete:
1. `/template` with no args shows help + template list
2. Training workflow shows "Save as Template" button at completion
3. Clicking button prompts for name, validates format
4. Valid name saves template to `templates/<user_id>.json`
5. `/template <name>` executes the saved workflow
6. `/template delete <name>` removes template
7. Invalid names show clear error with format guidance
</verification>

<success_criteria>
- Template recording works for both /train and /predict workflows
- Template execution replays full workflow automatically
- Name validation enforces TRAIN/PREDICT_XXXXXXXX_XXXXXXXX format
- Templates persist in JSON files per user
- Full CRUD operations available (create, read, list, delete)
- Error handling covers missing templates, invalid names, missing files
</success_criteria>
