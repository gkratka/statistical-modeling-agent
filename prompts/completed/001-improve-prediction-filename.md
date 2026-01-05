<objective>
Improve the prediction output filename in /predict workflow to include model type and prediction column info.

Current format: `predictions_job_c13cf90cc57b.csv`
Target format: `predictions_job_XGBOOST_CLASS2_c13cf90cc57b.csv`

The middle segment should convey:
- Model type (e.g., XGBOOST, CATBOOST, LIGHTGBM, RANDOM_FOREST)
- Prediction column name (e.g., CLASS2, PRICE, TARGET)
</objective>

<context>
Read the CLAUDE.md for project conventions.

Key file: `src/bot/ml_handlers/prediction_handlers.py`
- Line 2252: `_generate_output_filename()` method already exists
- Line 1901: Current temp file naming pattern
- Line 1911: Telegram download filename

The model info and prediction column should be available in session or from model metadata.
</context>

<requirements>
1. Modify `_generate_output_filename()` to include model type and prediction column
2. Extract model type from model_id or model metadata
3. Extract prediction column from session.selections or output_column_name
4. Format: `predictions_job_{MODEL_TYPE}_{PRED_COLUMN}_{job_id}.csv`
5. Ensure backwards compatibility - handle cases where info is missing
</requirements>

<implementation>
1. Find all places where prediction output filenames are generated
2. Update naming pattern to include model type (uppercase) and prediction column (uppercase)
3. If prediction column or model type unavailable, use "MODEL" or "PRED" as fallback
4. Keep the job_id/timestamp suffix for uniqueness
</implementation>

<output>
Modify: `./src/bot/ml_handlers/prediction_handlers.py`
</output>

<verification>
- Run /predict workflow and verify filename includes model type and prediction column
- Check console logs show correct filename generation
</verification>

<success_criteria>
- Prediction CSV files saved with format: `predictions_job_XGBOOST_CLASS2_abc123.csv`
- All existing functionality continues to work
- Missing info gracefully falls back to defaults
</success_criteria>
