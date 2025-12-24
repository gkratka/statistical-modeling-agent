<objective>
Modify all binary classification models to output probability scores instead of rounded binary values (0/1).

Currently, binary models round predictions to discrete values (0 or 1), but we need the raw probability scores to distinguish between high-confidence predictions (e.g., 0.99) and low-confidence ones (e.g., 0.51).

Output probabilities should have 8 decimal places precision.
</objective>

<context>
This is a Telegram bot that performs ML training and predictions. The prediction workflow:
1. User runs /predict with a trained model
2. Worker executes prediction via `execute_predict_job()` in `worker/statsbot_worker.py`
3. Results are sent back to bot and displayed to user

Binary classification models include:
- logistic regression
- decision tree classifier
- random forest classifier
- gradient boosting classifier
- lightgbm binary classification
- xgboost binary classification
- catboost binary classification
- naive bayes
- SVM classifier
- MLP classifier (sklearn neural network)
- Keras neural network (binary classification)

@worker/statsbot_worker.py - Contains prediction execution logic
@src/engines/ml_engine.py - Contains ML engine with predict method
</context>

<research>
Thoroughly analyze the codebase to find:

1. Where predictions are made for binary classification models
2. How `model.predict()` vs `model.predict_proba()` is used
3. Any rounding or type conversion applied to predictions
4. How predictions are formatted before sending to bot

Key files to examine:
- worker/statsbot_worker.py (execute_predict_job function)
- src/engines/ml_engine.py (predict method)
- Any preprocessing/postprocessing of prediction results
</research>

<requirements>
1. For ALL binary classification models, use `predict_proba()` instead of `predict()` when available
2. Extract the probability of the positive class (typically index 1 of predict_proba output)
3. Round probabilities to 8 decimal places using `round(prob, 8)` or `np.round(prob, 8)`
4. Ensure backwards compatibility - regression models should continue using `predict()`
5. Handle edge cases:
   - Models that don't have predict_proba (use decision_function or fall back to predict)
   - Multi-class models should NOT be affected (only binary classification)
   - Neural networks (Keras/MLP) may have different probability output format
   - Keras already outputs probabilities via sigmoid, just needs 8 decimal rounding

<implementation_details>
The change should be in the prediction logic, NOT training. Detection approach:
- Check if model has `predict_proba` method
- Check if task_type == "classification" AND number of unique classes == 2
- OR check model metadata for binary classification indicator
</implementation_details>
</requirements>

<constraints>
- Do NOT modify training logic, only prediction output
- Do NOT change regression model behavior
- Do NOT change multi-class classification behavior
- Maintain existing result format structure (predictions array, count, etc.)
- Keep 8 decimal precision (not more, not less) for consistency
</constraints>

<output>
Modify files as needed. Expected changes in:
- `worker/statsbot_worker.py` - prediction execution
- Possibly `src/engines/ml_engine.py` - if prediction logic exists there

After implementation, the prediction output should look like:
- Before: [0, 1, 1, 0, 1]
- After: [0.12345678, 0.98765432, 0.87654321, 0.23456789, 0.51234567]
</output>

<verification>
Before declaring complete:
1. Verify predict_proba is called for binary classification models
2. Verify 8 decimal places are preserved in output
3. Verify regression models still use predict() and output continuous values
4. Run existing tests: `python3 -m pytest tests/unit/test_worker_dataframe.py -v`
5. Check that no syntax errors exist in modified files
</verification>

<success_criteria>
- Binary classification predictions show probability (0.0 to 1.0) with 8 decimals
- Regression predictions unchanged
- All existing tests pass
- No breaking changes to result format structure
</success_criteria>
</content>
</invoke>