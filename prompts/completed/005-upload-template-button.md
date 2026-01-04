<objective>
Add an "Upload Template" button to both /train and /predict workflows that allows users to upload JSON template files from their local machine and execute them.

This enables users to share templates across devices or restore templates from local backups without needing them saved in the bot's database.
</objective>

<context>
Project: Telegram bot for ML training and prediction workflows
Tech stack: Python, python-telegram-bot, async handlers

Current workflow for /train:
1. User runs /train
2. Bot shows data source options (Upload File, Use Local Path, Use Template)
3. If "Use Template" → shows list of saved templates from bot database
4. User selects template → bot shows template details with "Load Data & Train" button
5. User clicks button → training executes

The same pattern exists for /predict workflow.

Relevant files to examine:
- src/bot/ml_handlers/template_handlers.py (train template selection)
- src/bot/ml_handlers/prediction_template_handlers.py (predict template handling)
- src/bot/ml_handlers/ml_training_local_path.py (training workflow)
- src/bot/ml_handlers/prediction_handlers.py (prediction workflow)
</context>

<requirements>
1. Add "Upload Template" button to template selection screen:
   - Position: Just above the "Back" button in the template list
   - Icon suggestion: "📤 Upload Template" or "📁 Upload Template"
   - Applies to BOTH /train and /predict workflows

2. Upload flow:
   - User clicks "Upload Template" button
   - Bot responds: "Please send me a JSON template file"
   - Bot enters state waiting for file upload
   - User uploads .json file via Telegram
   - Bot validates the JSON structure matches expected template schema

3. Validation requirements:
   - For TRAIN templates: must have file_path, target_column, feature_columns, model_type
   - For PREDICT templates: must have file_path, model_id, feature_columns, output_column_name
   - Check "type" field matches current workflow (train vs predict)
   - Validate JSON is parseable

4. After successful upload:
   - Display template details (same format as selecting from list - see Image #3)
   - Show "Load Data & Train" button (for train) or "Load Data & Predict" button (for predict)
   - User clicks to execute

5. Error handling:
   - Invalid JSON → "Invalid JSON file. Please upload a valid template."
   - Wrong template type → "This is a [predict] template. Please upload a [train] template."
   - Missing required fields → "Template missing required fields: [list fields]"
   - Non-.json file → "Please upload a .json file"
</requirements>

<implementation>
1. Add new callback data handlers:
   - `upload_train_template` - for /train workflow
   - `upload_pred_template` - for /predict workflow

2. Add new states to track upload mode:
   - `AWAITING_TRAIN_TEMPLATE_UPLOAD`
   - `AWAITING_PRED_TEMPLATE_UPLOAD`

3. Create document handler to receive uploaded JSON files when in upload state

4. Reuse existing template display and execution logic:
   - Train: reuse `_show_template_details()` and `_execute_training_from_template()`
   - Predict: reuse similar methods in prediction_template_handlers.py

5. Button placement in template list keyboard:
   ```python
   keyboard = [
       # ... existing template buttons ...
       [InlineKeyboardButton("📤 Upload Template", callback_data="upload_train_template")],
       [InlineKeyboardButton("Back", callback_data="back_to_data_source")]
   ]
   ```
</implementation>

<output>
Modify these files:
- `./src/bot/ml_handlers/template_handlers.py` - Add upload button and handlers for /train
- `./src/bot/ml_handlers/prediction_template_handlers.py` - Add upload button and handlers for /predict
- `./src/core/state_manager.py` - Add new states if needed
- `./src/bot/messages/` - Add any new message templates

Follow existing code patterns and style conventions from @CLAUDE.md
</output>

<verification>
Test the complete flow:

For /train:
1. Run /train → select "Use Template" → verify "Upload Template" button appears
2. Click "Upload Template" → verify bot asks for JSON file
3. Upload valid train template JSON → verify template details display
4. Click "Load Data & Train" → verify training executes
5. Test error cases: invalid JSON, wrong template type, missing fields

For /predict:
1. Run /predict → navigate to template selection → verify "Upload Template" button
2. Same upload and execution flow as above
3. Test error cases

Check state cleanup - ensure upload state clears properly after success or cancel.
</verification>

<success_criteria>
- "Upload Template" button visible in both /train and /predict template screens
- Users can upload JSON files and have templates load correctly
- Validation catches all invalid inputs with helpful error messages
- Uploaded templates execute identically to database-saved templates
- No regressions to existing template selection flow
</success_criteria>
