<objective>
Add an optional "filter" step to the /join workflow after confirming key columns.

After users confirm their join key columns, the bot should ask if they want to add filters to apply during the join operation. Users can either skip (no filters) or type filter expressions like `month = 1` or `status = 'active'`.
</objective>

<context>
This is a Telegram bot that performs dataframe operations. The /join workflow:
1. User runs /join
2. Selects operation type (left_join, inner_join, concat, etc.)
3. Selects number of dataframes (2, 3, or 4)
4. Provides each dataframe via upload or local path
5. Selects key columns for join operations (or skips for union/concat)
6. **NEW STEP: Optional filter input** ← Add this
7. Selects output path (default or custom)
8. Worker executes join and returns result

Join operations that use key columns: left_join, right_join, inner_join, outer_join, merge
Stack operations that skip key columns: union, concat, cross_join

@src/bot/handlers/join_handlers.py - Main handler class for /join workflow
@src/bot/messages/join_messages.py - User-facing messages and button helpers
@src/core/state_manager.py - JoinWorkflowState enum defining workflow states
@worker/statsbot_worker.py - Worker that executes join operations (execute_join_job function)
</context>

<research>
Before implementing, examine:

1. Current state flow in `JoinWorkflowState` enum (state_manager.py)
   - Current: CHOOSING_KEY_COLUMNS → CHOOSING_OUTPUT_PATH
   - New: CHOOSING_KEY_COLUMNS → CHOOSING_FILTER → CHOOSING_OUTPUT_PATH

2. How `handle_key_column_selection()` transitions to output path step
   - Line ~549: After confirm, goes to CHOOSING_OUTPUT_PATH
   - Need to insert filter step here

3. Message template patterns in `join_messages.py`
   - Follow existing static method patterns
   - Button creation helper patterns

4. Worker job params in `execute_join_job()` in statsbot_worker.py
   - Currently receives: operation, file_paths, key_columns, output_path
   - Need to add: filters (list of filter expressions)
</research>

<requirements>
## State Machine Changes (state_manager.py)
Add new state to `JoinWorkflowState` enum:
```python
# After CHOOSING_KEY_COLUMNS, before CHOOSING_OUTPUT_PATH
CHOOSING_FILTER = "choosing_filter"
AWAITING_FILTER_INPUT = "awaiting_filter_input"  # If user types filter text
```

## Message Templates (join_messages.py)
Add to `JoinMessages` class:

1. `filter_prompt()` - Ask if user wants filters
   ```
   *Optional Filters*

   You can add filters to apply during the join operation.

   To add a filter, type it in this format:
     `column_name = value`

   Examples:
     `month = 1`
     `status = 'active'`
     `amount > 100`

   Supported operators: =, !=, >, <, >=, <=

   Or click "No Filters" to proceed without filtering.
   ```

2. `filter_added_message(filter_expr)` - Confirm filter was added
   ```
   Filter added: `{filter_expr}`

   Add another filter, or click "Done with Filters" to continue.
   ```

3. `filter_error_message(error)` - Invalid filter format
   ```
   Invalid filter format: {error}

   Expected format: `column = value`
   Example: `month = 1` or `status = 'active'`
   ```

4. Button helper: `create_filter_buttons(filters_added=[])`
   - If no filters: [["No Filters"]]
   - If filters exist: [["Add Another Filter"], ["Done with Filters"]]

## Handler Changes (join_handlers.py)

1. Modify `handle_key_column_selection()`:
   - After confirm, instead of going to output path, go to CHOOSING_FILTER state
   - Call `_show_filter_prompt()` instead of `_show_output_path_selection()`

2. Add `_show_filter_prompt(update, session)`:
   - Show filter prompt message with "No Filters" button
   - Set state to CHOOSING_FILTER

3. Add `handle_filter_selection(update, context)`:
   - Handle callback_data: `join_filter_skip`, `join_filter_done`, `join_filter_add_more`
   - If skip/done: proceed to output path selection
   - If add_more: set state to AWAITING_FILTER_INPUT

4. Add `handle_filter_input(update, context)`:
   - Handle text message with filter expression
   - Parse and validate filter: `column operator value`
   - Validate column exists in at least one dataframe
   - Store in `session.selections["filters"]` list
   - Show confirmation and updated buttons

5. For operations that skip key columns (union, concat, cross_join):
   - After all dataframes collected, go to filter step (not directly to output)
   - Modify `_advance_to_next_step()` to check if filter step should be shown

## Worker Changes (statsbot_worker.py)

1. Modify `execute_join_job()` to accept `filters` parameter:
   ```python
   filters = params.get("filters", [])
   ```

2. Apply filters to dataframes BEFORE join:
   ```python
   def apply_filters(df, filters):
       for filter_expr in filters:
           # Parse: "column op value"
           # Use df.query() or boolean indexing
           df = df.query(filter_expr)
       return df
   ```

## Job Parameters Update (join_handlers.py)

In `_execute_join()`, add filters to job_params:
```python
job_params = {
    "operation": operation,
    "file_paths": file_paths,
    "key_columns": key_columns,
    "output_path": output_path,
    "filters": session.selections.get("filters", []),  # NEW
}
```
</requirements>

<constraints>
- Filter step is OPTIONAL - users can skip with "No Filters" button
- Filters apply to ALL dataframes before the join (not just one)
- Invalid filter expressions should show helpful error message, not crash
- Keep workflow stateless between steps (all data in session.selections)
- Follow existing i18n patterns if present (check for I18nManager usage)
- Filter validation should check column exists in at least one dataframe
- Support common operators: =, !=, >, <, >=, <=
- String values should be quoted: `status = 'active'`
- Numeric values should not be quoted: `amount > 100`
</constraints>

<output>
Files to modify:
1. `src/core/state_manager.py` - Add CHOOSING_FILTER and AWAITING_FILTER_INPUT states
2. `src/bot/messages/join_messages.py` - Add filter-related message methods and button helper
3. `src/bot/handlers/join_handlers.py` - Add filter handlers and modify key column flow
4. `worker/statsbot_worker.py` - Add filter application logic in execute_join_job
5. `src/bot/telegram_bot.py` - Register new callback handlers for filter patterns

New callback patterns to register:
- `^join_filter_` - For filter selection callbacks
</output>

<verification>
Before declaring complete:
1. Verify state transitions: key_columns → filter → output_path works
2. Verify "No Filters" button skips to output path
3. Verify filter text input is parsed and validated correctly
4. Verify filters are passed to worker in job_params
5. Verify worker applies filters before join operation
6. Test with invalid filter expression - should show error, not crash
7. Test full workflow: /join → operation → count → dataframes → keys → filters → output → execute
8. Check syntax: `python3 -m py_compile src/bot/handlers/join_handlers.py`
</verification>

<success_criteria>
- User can type `month = 1` as filter and it's applied to join
- User can click "No Filters" to skip filter step
- User can add multiple filters before clicking "Done with Filters"
- Invalid filter format shows helpful error message
- Filters are applied to dataframes before join/union/concat operation
- Existing tests still pass (if any exist for join workflow)
</success_criteria>
</content>
