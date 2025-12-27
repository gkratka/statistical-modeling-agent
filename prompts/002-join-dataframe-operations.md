<objective>
Implement a new `/join` command for the Telegram bot that allows users to join, union, concat, or merge multiple dataframes (2-4). This is a data mining/manipulation capability similar to `/train` and `/predict` workflows.

The feature must handle large dataframes using the worker pattern (like /predict) to avoid loading data into Telegram messages.
</objective>

<context>
This is a Telegram bot that performs ML training and predictions. The bot uses:
- State machine for conversation workflows (`src/core/state_manager.py`)
- Handler pattern for commands (`src/bot/handlers/`, `src/bot/ml_handlers/`)
- Worker execution for heavy operations (`worker/statsbot_worker.py`)
- Local path validation for file loading (`src/utils/path_validator.py`)
- Messages/prompts organization (`src/bot/messages/`)

Key reference implementations to follow:
@src/bot/ml_handlers/ml_training_local_path.py - Local path workflow pattern
@src/bot/ml_handlers/prediction_handlers.py - Worker execution pattern
@worker/statsbot_worker.py - Job execution on worker
@src/core/state_manager.py - State management pattern
</context>

<research>
Before implementing, thoroughly analyze these existing patterns:
1. How `/train` handles data source selection (Upload File vs Local Path buttons)
2. How `/predict` executes jobs on worker and saves results to local CSV
3. How state machine tracks multi-step workflows
4. How messages are organized in `src/bot/messages/`
</research>

<requirements>
## Supported Operations
1. **Left Join** - Keep all rows from left dataframe
2. **Right Join** - Keep all rows from right dataframe
3. **Inner Join** - Only matching rows
4. **Outer Join** - All rows from both dataframes
5. **Cross Join** - Cartesian product
6. **Union** - Vertical stack (same columns required)
7. **Concat** - Vertical stack (handles different columns)
8. **Merge** - Pandas merge with configurable parameters

## Workflow Steps
1. User sends `/join`
2. Bot shows intro message + operation buttons (Left Join, Right Join, Inner Join, Outer Join, Cross Join, Union, Concat, Merge)
3. User selects operation
4. Bot asks how many dataframes (buttons: 2, 3, 4 - with "2" marked as "most common")
5. For each dataframe:
   - Bot shows [📤 Upload File] [📁 Use Local Path] buttons
   - If Local Path: show allowed directories, formats, examples (like /train)
   - User provides file path or uploads file
   - Bot validates and loads dataframe schema (column names)
6. For join operations (not union/concat): Bot asks for key column(s)
   - Show available columns from loaded dataframes
   - Allow single or multiple key columns
7. Bot asks for output location:
   - [📁 Default (same as first input)] button
   - Or user types custom path
8. Bot executes operation via worker
9. Worker saves result to specified path, returns summary
10. Bot displays completion message with row count, output path

## State Machine States (new)
- `CHOOSING_JOIN_OPERATION` - Selecting operation type
- `CHOOSING_DATAFRAME_COUNT` - Selecting 2-4 dataframes
- `AWAITING_DATAFRAME_N_SOURCE` - Choosing upload vs local path for dataframe N
- `AWAITING_DATAFRAME_N_PATH` - Waiting for local path input for dataframe N
- `AWAITING_DATAFRAME_N_UPLOAD` - Waiting for file upload for dataframe N
- `CHOOSING_KEY_COLUMNS` - Selecting join key(s)
- `CHOOSING_OUTPUT_PATH` - Selecting output location
- `EXECUTING_JOIN` - Worker executing operation

## Key Column Selection
- For joins: Required - show columns common to all dataframes
- For union: Not needed (must have same columns)
- For concat: Not needed (handles different columns)
- For merge: Required - allow left_on/right_on if columns differ

## Output Handling
- Worker executes pandas operation
- Saves result to CSV at specified path
- Returns only summary (row count, columns, file path) - NOT the dataframe
- Pattern: Same as /predict truncation to prevent OOM
</requirements>

<implementation>
## Files to Create
1. `src/bot/handlers/join_handlers.py` - Main /join command and workflow handlers
2. `src/bot/messages/join_messages.py` - User-facing messages and button labels
3. Add new job type `execute_join_job` in `worker/statsbot_worker.py`

## Files to Modify
1. `src/core/state_manager.py` - Add new workflow type and states
2. `src/bot/telegram_bot.py` - Register /join command handler
3. `worker/statsbot_worker.py` - Add join job execution

## Handler Structure
```python
# Pattern from ml_training_local_path.py
async def join_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /join command"""
    # Show intro + operation buttons

async def handle_operation_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle operation button click (callback query)"""
    # Store selected operation, ask for dataframe count

async def handle_dataframe_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle dataframe count selection"""
    # Store count, start dataframe collection loop

async def handle_dataframe_source(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Upload/Local Path selection for current dataframe"""

async def handle_local_path_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle local path text input"""
    # Validate path, load schema, move to next dataframe or key selection

async def handle_key_column_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle key column selection for joins"""

async def handle_output_path(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle output path selection"""
    # Execute join via worker
```

## Worker Job Execution
```python
def execute_join_job(job_id: str, params: Dict[str, Any], ws_send_callback) -> str:
    """
    Execute dataframe join/union/concat/merge operation.

    params:
        operation: str - "left_join", "right_join", "inner_join", "outer_join",
                        "cross_join", "union", "concat", "merge"
        file_paths: List[str] - Paths to input dataframes
        key_columns: Optional[List[str]] - Join key(s)
        output_path: str - Where to save result
    """
    import pandas as pd

    # Load all dataframes
    dfs = [pd.read_csv(path) for path in file_paths]

    # Execute operation
    if operation == "left_join":
        result = dfs[0].merge(dfs[1], on=key_columns, how="left")
        for df in dfs[2:]:
            result = result.merge(df, on=key_columns, how="left")
    elif operation == "union":
        result = pd.concat(dfs, ignore_index=True)
    # ... other operations

    # Save result (truncation pattern from /predict)
    result.to_csv(output_path, index=False)

    return create_result_message(job_id, True, data={
        "output_file": output_path,
        "row_count": len(result),
        "column_count": len(result.columns),
        "columns": list(result.columns)[:20],  # Sample for display
    })
```

## Button Callbacks
Use consistent callback_data patterns:
- `join_op_left`, `join_op_right`, `join_op_inner`, etc.
- `join_count_2`, `join_count_3`, `join_count_4`
- `join_source_upload_1`, `join_source_local_1`, etc.
- `join_output_default`, `join_output_custom`
</implementation>

<constraints>
- NEVER load full dataframes into bot memory - use worker pattern
- MUST validate all file paths using existing path_validator.py
- MUST handle large files (10M+ rows) - save to disk, return summary only
- MUST support CSV, Excel, Parquet formats (like /train)
- MUST track state properly for multi-step workflow
- Follow existing code patterns and naming conventions
</constraints>

<output>
Create/modify these files:
- `./src/bot/handlers/join_handlers.py` - Main handler implementation
- `./src/bot/messages/join_messages.py` - Messages and button labels
- Modify `./src/core/state_manager.py` - Add JOIN workflow and states
- Modify `./src/bot/telegram_bot.py` - Register handlers
- Modify `./worker/statsbot_worker.py` - Add execute_join_job function
</output>

<verification>
Before declaring complete:
1. Verify /join command is registered and responds
2. Verify all operation buttons work (callback handlers registered)
3. Verify state transitions work correctly through full workflow
4. Verify worker job executes and saves result file
5. Verify large dataframe handling (no OOM on 10M+ rows)
6. Run syntax check on all modified files: `python3 -m py_compile <file>`
</verification>

<success_criteria>
- User can run `/join` and complete full workflow
- All 8 operations work correctly (left, right, inner, outer, cross, union, concat, merge)
- 2-4 dataframes supported
- Both upload and local path work
- Result saved to specified/default location
- Summary message shows row count and output path
- No OOM errors on large dataframes
</success_criteria>
</content>
</invoke>