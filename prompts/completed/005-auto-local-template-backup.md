<objective>
Implement automatic local backup of templates when users save them in the Telegram bot.

When a user saves a template (for training or prediction), the system should ALSO save a JSON backup file to the local filesystem in the same directory as the data file.

This provides users with a local backup in case templates are lost due to bot updates or bugs.
</objective>

<context>
This is a Telegram bot for ML training and predictions. Users can save "templates" containing their configuration (target column, features, model type, hyperparameters) for reuse.

Current template save locations:
- Telegram: `templates/{user_id}.json` (unified storage)
- ML Training: `src/bot/ml_handlers/template_handlers.py`
- Prediction: `src/bot/handlers/template_handlers.py`

Key files to examine:
@src/bot/ml_handlers/template_handlers.py - ML training template save logic
@src/bot/handlers/template_handlers.py - Prediction template save logic
@src/core/state_manager.py - Session state with file_path info
</context>

<requirements>
1. When a template is saved to Telegram, ALSO save a JSON backup file locally
2. Backup location: Same directory as the user's data file
   - If local path: Use the directory from `session.file_path`
   - If file upload: Use the directory from wherever the DataFrame was loaded
3. Backup filename format: `template_{TEMPLATE_NAME}.json`
4. Backup must include ALL fields:
   - Template name
   - Template type (train/predict)
   - File path
   - Target column
   - Feature columns
   - Model type/category
   - ALL hyperparameters (including defaults used during training)
   - Created timestamp
5. NO user prompt needed - backup happens automatically alongside Telegram save
6. Handle errors gracefully (log warning if backup fails, don't block main save)
</requirements>

<implementation>
1. In `src/bot/ml_handlers/template_handlers.py`:
   - After successful Telegram save, add local backup logic
   - Get directory from `session.file_path` (or uploaded file path from session)
   - Include `hyperparameters_used` from training result in template config
   - Use `Path(file_path).parent` to get directory
   - Save as `template_{template_name}.json`

2. In `src/bot/handlers/template_handlers.py`:
   - Same approach for prediction templates
   - Include prediction-specific config

3. Template JSON structure should include:
   ```json
   {
     "name": "TRAIN_XGBST_CLASS2",
     "type": "train",
     "created_at": "2026-01-03T18:23:36",
     "file_path": "/path/to/data.csv",
     "target_column": "class2",
     "feature_columns": ["col1", "col2", ...],
     "model_category": "classification",
     "model_type": "xgboost_binary_classification",
     "hyperparameters": {
       "n_estimators": 100,
       "max_depth": 6,
       "learning_rate": 0.1,
       ...all parameters including defaults...
     }
   }
   ```

4. For hyperparameters:
   - Training templates: Use `hyperparameters_used` from training result (stored in session)
   - This captures ALL actual hyperparameters including defaults
</implementation>

<constraints>
- Do NOT prompt user for backup confirmation
- Do NOT block main Telegram save if backup fails
- Use try/except with logging for backup errors
- Avoid Markdown special characters in any messages (use plain text)
</constraints>

<output>
Modify files:
- `src/bot/ml_handlers/template_handlers.py` - Add local backup after ML template save
- `src/bot/handlers/template_handlers.py` - Add local backup after prediction template save (if applicable)
</output>

<verification>
1. Train a model using local path, save template with name "TEST_BACKUP"
2. Check that `template_TEST_BACKUP.json` exists in data file directory
3. Verify JSON contains all hyperparameters including defaults
4. Test with file upload path as well
5. Test that Telegram save still works if backup directory is read-only (graceful failure)
</verification>

<success_criteria>
- Template saved to Telegram AND local JSON file created
- JSON includes ALL hyperparameters (not just user-configured ones)
- Backup location matches data file directory
- No user prompts added for backup
- Main workflow unaffected if backup fails
</success_criteria>
