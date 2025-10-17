# Feature: Save Prediction Results to Local File Path

**Status:** Planning Complete ✅ | Implementation: Pending
**Created:** 2025-10-16
**Complexity:** Medium
**Estimated LOC:** ~850 lines (implementation + tests)

---

## Feature Overview

Add optional local file system save capability to the prediction workflow. After predictions complete, users can choose to save results to a local directory path instead of (or in addition to) downloading via Telegram.

### Key Benefits
- **No File Size Limits:** Bypass Telegram's 10MB file limit
- **Direct File Access:** Save directly to working directories
- **Flexible Naming:** Default descriptive names with rename option
- **Backward Compatible:** Existing Telegram download workflow unchanged

---

## User Experience Flow

```
[Predictions Complete + Stats Shown]
         ↓
[Output Options: 💾 Save Local | 📥 Telegram | ✅ Done]
         ↓ (if Save Local chosen)
[Prompt: "Enter directory path"]
         ↓
[User: /Users/me/data]
         ↓
[Validate: Path OK ✓]
         ↓
[Show: "Default filename: predictions_keras_binary_20251016.csv"]
[Buttons: ✅ Accept | ✏️ Rename]
         ↓ (if Accept)
[Save file to: /Users/me/data/predictions_keras_binary_20251016.csv]
         ↓
[Success: "✅ File saved: /Users/me/data/predictions_keras_binary_20251016.csv (200 rows)"]
         ↓
[Workflow Complete]
```

### User Interaction Points

1. **Output Method Selection** (after predictions complete)
   - 💾 Save to Local Path
   - 📥 Download via Telegram (existing)
   - ✅ Done (skip both)

2. **Directory Path Input**
   - User provides absolute path: `/Users/username/data`
   - Validation: security checks, whitelist enforcement, write permissions
   - Error recovery: retry or back to options

3. **Filename Confirmation**
   - Bot suggests: `predictions_keras_binary_classification_20251016_143022.csv`
   - User can: Accept default OR Provide custom name
   - Validation: no special chars, no path traversal, no conflicts

4. **Save Confirmation**
   - Success: Show full path and row count
   - Error: Clear error message with recovery options

---

## Technical Architecture

### State Machine Changes

#### New States (`MLPredictionState` enum)
```python
class MLPredictionState(str, Enum):
    # ... existing states ...
    COMPLETE = "COMPLETE"

    # NEW STATES
    AWAITING_SAVE_PATH = "AWAITING_SAVE_PATH"
    CONFIRMING_SAVE_FILENAME = "CONFIRMING_SAVE_FILENAME"
```

#### State Transition Graph
```
RUNNING_PREDICTION → COMPLETE
                      ↓
[Output Options Presented]
                      ↓
           ┌──────────┼──────────┐
           ↓          ↓          ↓
    Save Local  Telegram    Done
           ↓          ↓          ↓
  AWAITING_    [Send     [Reset
   SAVE_PATH   File]     Session]
           ↓
 CONFIRMING_SAVE_FILENAME
           ↓
      [Save File]
           ↓
       COMPLETE
```

### Default Filename Strategy

**Format:** `predictions_{model_name}_{timestamp}.csv`

**Example:** `predictions_keras_binary_classification_20251016_143022.csv`

**Benefits:**
- Descriptive: includes model type
- Collision-resistant: timestamp ensures uniqueness
- Sortable: chronological ordering
- Parseable: consistent structure

### Security Architecture

**Reuse Existing Infrastructure:**
- `PathValidator` class (already battle-tested for training)
- Same `allowed_directories` whitelist from config
- Multi-layer validation (8 security checks)

**New Validations for Output:**
```python
validate_output_path(directory_path: str, filename: str) -> dict:
    """
    Validate output path for saving predictions.

    Checks:
    1. Directory exists
    2. Directory is writable
    3. Parent directory in whitelist
    4. Filename is valid (no special chars)
    5. No path traversal in filename
    6. File doesn't exist (or prompt overwrite)
    7. Sufficient disk space (estimate from DataFrame size)
    8. Extension is .csv
    """
```

### Session Data Storage

**Store predictions after execution:**
```python
# In _execute_prediction() after ml_engine.predict()
session.predictions_result = df  # Full dataframe with predictions
session.selections['save_directory'] = None  # Set when user chooses path
session.selections['save_filename'] = None  # Set when user confirms name
```

**Memory Considerations:**
- DataFrame already in memory for statistics calculation
- No additional memory overhead
- Cleaned up when session ends or user chooses "Done"

---

## Implementation Plan

### Phase 1: State Machine (30 lines)
**File:** `src/core/state_manager.py`

**Changes:**
1. Add 2 new enum values to `MLPredictionState`
2. Update state transition graph validation
3. Add transition rules:
   - `COMPLETE → AWAITING_SAVE_PATH`
   - `AWAITING_SAVE_PATH → CONFIRMING_SAVE_FILENAME`
   - `CONFIRMING_SAVE_FILENAME → COMPLETE`

**Testing:** `tests/unit/test_prediction_save_workflow.py`

---

### Phase 2: Path Validation (80 lines)
**File:** `src/utils/path_validator.py`

**New Method:**
```python
def validate_output_path(
    self,
    directory_path: str,
    filename: str
) -> dict:
    """
    Validate output directory and filename for saving predictions.

    Returns:
        {
            'is_valid': bool,
            'resolved_path': Path,
            'error': str (if invalid),
            'warnings': list[str]
        }
    """
```

**Helper Methods:**
```python
def sanitize_filename(self, filename: str) -> str:
    """Remove/replace invalid filename characters."""

def check_disk_space(self, path: Path, required_mb: int) -> bool:
    """Verify sufficient disk space available."""
```

**Testing:** `tests/unit/test_path_validator_output.py`

---

### Phase 3: Message Templates (120 lines)
**File:** `src/bot/messages/prediction_messages.py`

**New Message Functions:**
```python
@staticmethod
def output_options_prompt() -> str:
    """Present output method choices after predictions complete."""

@staticmethod
def save_path_input_prompt(allowed_dirs: list[str]) -> str:
    """Prompt user for output directory path."""

@staticmethod
def filename_confirmation_prompt(
    default_name: str,
    directory: str
) -> str:
    """Show default filename with rename option."""

@staticmethod
def file_save_success_message(
    full_path: str,
    n_rows: int
) -> str:
    """Confirm successful file save."""

@staticmethod
def file_save_error_message(
    error_type: str,
    details: str
) -> str:
    """Error messages for various save failures."""
```

**Button Helpers:**
```python
def create_output_option_buttons() -> list:
    """Buttons: Save Local | Telegram | Done"""

def create_filename_confirmation_buttons() -> list:
    """Buttons: Accept Default | Custom Name"""
```

**Testing:** Unit tests in `tests/unit/test_prediction_messages.py`

---

### Phase 4: Handler Implementation (350 lines)
**File:** `src/bot/ml_handlers/prediction_handlers.py`

**Modify Existing:**
```python
async def _execute_prediction(...):
    # After predictions complete:
    # 1. Store df in session.predictions_result
    # 2. Show statistics message
    # 3. Call _show_output_options() instead of auto-sending file
```

**New Handler Methods:**
```python
async def _show_output_options(update, context, session):
    """Present output method choices with inline buttons."""

async def handle_output_option_selection(update, context):
    """Route based on: local_path | telegram | done"""

async def handle_save_directory_input(update, context):
    """Validate directory path and transition to filename confirmation."""

async def handle_filename_confirmation(update, context):
    """Handle default filename acceptance."""

async def handle_filename_custom_input(update, context):
    """Process custom filename from user."""

async def _execute_file_save(update, context, session):
    """
    Perform actual file write operation.

    Steps:
    1. Combine directory + filename
    2. Final validation
    3. df.to_csv(path, index=False)
    4. Send success/error message
    5. Transition back to COMPLETE
    """

def _generate_default_filename(model_id: str, timestamp: float) -> str:
    """Generate descriptive default filename."""
```

**Update Routing:**
```python
async def handle_text_input(update, context):
    # Add routing for new states:
    if current_state == MLPredictionState.AWAITING_SAVE_PATH.value:
        await self.handle_save_directory_input(update, context)
    elif current_state == MLPredictionState.CONFIRMING_SAVE_FILENAME.value:
        await self.handle_filename_custom_input(update, context)
```

**Register Callbacks:**
```python
def register_prediction_handlers(...):
    # Add new callback patterns:
    CallbackQueryHandler(
        handler.handle_output_option_selection,
        pattern=r"^pred_output_(local|telegram|done)$"
    )

    CallbackQueryHandler(
        handler.handle_filename_confirmation,
        pattern=r"^pred_filename_(default|custom)$"
    )
```

**Testing:**
- `tests/unit/test_prediction_handlers.py` (unit tests)
- `tests/integration/test_prediction_local_save_e2e.py` (E2E workflow)

---

### Phase 5: Testing Strategy (450 lines total)

#### Unit Tests: Path Validation
**File:** `tests/unit/test_path_validator_output.py` (~150 lines)

```python
class TestPathValidatorOutput:
    def test_valid_directory_and_filename(path_validator):
        """Happy path: valid directory and filename."""

    def test_directory_not_writable(path_validator, tmp_path):
        """Error: directory exists but no write permissions."""

    def test_directory_not_in_whitelist(path_validator):
        """Error: directory outside allowed paths."""

    def test_filename_with_path_traversal(path_validator):
        """Error: filename contains '../' or similar."""

    def test_filename_with_special_chars(path_validator):
        """Sanitization: replace invalid filename characters."""

    def test_file_already_exists(path_validator, tmp_path):
        """Warning: file exists, offer overwrite option."""

    def test_insufficient_disk_space(path_validator, monkeypatch):
        """Error: not enough disk space for predicted file size."""

    def test_invalid_file_extension(path_validator):
        """Error: must be .csv extension."""
```

#### Unit Tests: State Transitions
**File:** `tests/unit/test_prediction_save_workflow.py` (~120 lines)

```python
class TestPredictionSaveStateTransitions:
    def test_complete_to_awaiting_save_path(state_manager):
        """Valid transition from COMPLETE to AWAITING_SAVE_PATH."""

    def test_awaiting_save_path_to_confirming_filename(state_manager):
        """Valid transition after directory path validation."""

    def test_confirming_filename_to_complete(state_manager):
        """Valid transition after successful file save."""

    def test_invalid_transitions(state_manager):
        """Ensure invalid state jumps are rejected."""
```

#### Integration Tests: E2E Workflow
**File:** `tests/integration/test_prediction_local_save_e2e.py` (~180 lines)

```python
class TestPredictionLocalSaveE2E:
    async def test_full_local_save_workflow(mock_bot, tmp_path):
        """
        Complete workflow:
        1. /predict command
        2. Upload file
        3. Select features
        4. Select model
        5. Run predictions
        6. Choose "Save Local"
        7. Provide directory path
        8. Accept default filename
        9. Verify file saved correctly
        """

    async def test_local_save_with_custom_filename(mock_bot, tmp_path):
        """Workflow with custom filename renaming."""

    async def test_telegram_download_still_works(mock_bot):
        """Ensure existing Telegram download path unaffected."""

    async def test_done_option_skips_save(mock_bot):
        """Choosing 'Done' completes workflow without saving."""

    async def test_error_recovery_paths(mock_bot):
        """
        Test error scenarios:
        - Invalid directory → Retry
        - Permission denied → Back to options
        - Filename conflict → Rename or overwrite
        """
```

---

## File Modification Summary

| File | Lines Added | Lines Modified | Complexity |
|------|-------------|----------------|------------|
| `src/core/state_manager.py` | 30 | 10 | Low |
| `src/utils/path_validator.py` | 80 | 0 | Medium |
| `src/bot/ml_handlers/prediction_handlers.py` | 350 | 50 | High |
| `src/bot/messages/prediction_messages.py` | 120 | 0 | Low |
| **Tests (new files)** | 450 | - | Medium |
| **TOTAL** | **1030** | **60** | **Medium** |

---

## Security Considerations

### Reuse Existing Infrastructure
- Leverage `PathValidator` (350 lines, 39 tests, battle-tested)
- Same whitelist from `config.yaml`
- Same validation patterns as local training feature

### Additional Output-Specific Checks
1. **Write Permission Validation:** Check `os.access(dir, os.W_OK)`
2. **Disk Space Check:** `shutil.disk_usage()` before write
3. **File Existence Check:** Warn if overwriting existing file
4. **Filename Sanitization:** Remove/escape special characters
5. **Extension Enforcement:** Must be `.csv`

### Attack Vector Mitigation
- **Path Traversal:** Reject `../`, `..\\`, URL-encoded variants
- **Symlink Attacks:** Resolve symlinks before validation
- **Directory Creation:** Never auto-create directories
- **Overwrite Protection:** Require explicit confirmation

---

## Edge Cases & Error Handling

### Edge Case 1: Directory vs Full Path
**Scenario:** User provides `/Users/me/data/results.csv` instead of `/Users/me/data`

**Solution:**
```python
if os.path.isfile(user_input):
    directory = os.path.dirname(user_input)
    filename = os.path.basename(user_input)
    # Validate filename, use it instead of default
elif os.path.isdir(user_input):
    directory = user_input
    filename = generate_default()
```

### Edge Case 2: Filename Conflicts
**Scenario:** `predictions_model_123.csv` already exists

**Solution:**
```python
if os.path.exists(target_path):
    # Option 1: Auto-rename with suffix
    target_path = "predictions_model_123_(1).csv"

    # Option 2: Prompt user
    keyboard = [
        [InlineKeyboardButton("Overwrite", callback_data="overwrite")],
        [InlineKeyboardButton("Rename", callback_data="rename")]
    ]
```

### Edge Case 3: Insufficient Disk Space
**Scenario:** Large prediction file (500MB) but only 100MB free

**Solution:**
```python
df_size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
free_space_mb = shutil.disk_usage(directory).free / (1024 ** 2)

if df_size_mb > free_space_mb:
    await update.message.reply_text(
        f"❌ **Insufficient Disk Space**\n\n"
        f"Required: {df_size_mb:.1f} MB\n"
        f"Available: {free_space_mb:.1f} MB\n\n"
        f"Please choose a different directory or free up space."
    )
```

### Edge Case 4: Permission Denied
**Scenario:** Directory exists but user lacks write permission

**Solution:**
```python
if not os.access(directory, os.W_OK):
    await update.message.reply_text(
        "❌ **Permission Denied**\n\n"
        f"You don't have write permission for:\n"
        f"`{directory}`\n\n"
        "Please choose a directory where you have write access.",
        reply_markup=create_path_error_recovery_buttons()
    )
```

### Edge Case 5: Special Characters in Filename
**Scenario:** User provides filename: `my<predictions>2024.csv`

**Solution:**
```python
def sanitize_filename(filename: str) -> str:
    """Remove/replace invalid characters."""
    # Remove: < > : " / \ | ? *
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Ensure .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'

    return filename
```

---

## Backward Compatibility

### Existing Behavior Preserved
✅ **Telegram Download:** Unchanged, still default option
✅ **Temp File Creation:** Still creates temp file for Telegram
✅ **Statistics Display:** Same format and content
✅ **Workflow Complete Message:** Still shown after all operations

### New Behavior Optional
- **Local Save:** User must explicitly choose this option
- **"Done" Option:** Skip all file operations
- **Default Flow:** If user doesn't choose, workflow ends gracefully

### Migration Path
- **Existing Users:** No impact, existing workflow identical
- **New Users:** Discover new feature via inline buttons
- **Documentation:** Update CLAUDE.md with new workflow description

---

## Configuration

### No New Config Required
- Reuse existing `local_data.allowed_directories` setting
- Reuse existing `local_data.max_file_size_mb` setting
- No new feature flags needed

### Example Config (existing)
```yaml
local_data:
  enabled: true
  allowed_directories:
    - /Users/username/datasets
    - /home/user/data
    - ./data
  max_file_size_mb: 1000
  allowed_extensions: [.csv, .xlsx, .xls, .parquet]
```

**Output files automatically validated against same whitelist.**

---

## Success Criteria

### Functional Requirements
- [ ] User can save predictions to local file path
- [ ] Default filename is descriptive and unique
- [ ] User can rename file before saving
- [ ] File saves successfully to chosen location
- [ ] Errors are clear and actionable
- [ ] Existing Telegram workflow unaffected

### Security Requirements
- [ ] All path validation checks pass
- [ ] No path traversal vulnerabilities
- [ ] Whitelist enforcement working
- [ ] Write permission validation working
- [ ] Disk space checks prevent failures

### Testing Requirements
- [ ] 100% pass rate on unit tests
- [ ] 100% pass rate on integration tests
- [ ] No regressions in existing tests
- [ ] E2E workflow validated in Telegram
- [ ] Error recovery paths tested

### Code Quality Requirements
- [ ] Type annotations complete
- [ ] Docstrings for all new functions
- [ ] Error handling comprehensive
- [ ] Logging at appropriate levels
- [ ] Follows existing code patterns

---

## Implementation Timeline

### Phase 1: Foundation (Day 1)
- State machine modifications
- Path validator enhancements
- Unit tests for path validation

### Phase 2: Messages & UI (Day 1-2)
- Message templates
- Button helpers
- Message unit tests

### Phase 3: Handlers (Day 2-3)
- Handler method implementation
- Callback registration
- Handler unit tests

### Phase 4: Integration (Day 3)
- E2E workflow testing
- Error scenario testing
- Telegram bot testing

### Phase 5: Documentation (Day 3)
- Update CLAUDE.md
- Add usage examples
- Update README if needed

**Total Estimated Time:** 3 days

---

## Future Enhancements

### Potential Follow-ups
1. **Multiple Format Support:** Add JSON, Excel, Parquet outputs
2. **Compression Options:** Offer gzip compression for large files
3. **Cloud Storage Integration:** Save to S3, GCS, Azure Blob
4. **Batch Operations:** Save multiple prediction runs at once
5. **Email Results:** Send file via email instead of download

### User Feedback Opportunities
- Track usage: How often is local save used vs Telegram?
- Error patterns: Which validation errors occur most?
- UX improvements: Is filename rename used often?

---

## Notes

**Created:** 2025-10-16
**Author:** Claude Code (GPT-4.5)
**Review:** Pending
**Approval:** Plan approved by user

**Related Features:**
- Local File Path Training (implemented)
- Path Validation Infrastructure (existing)
- Prediction Workflow (existing)

**Dependencies:**
- PathValidator (src/utils/path_validator.py)
- StateManager (src/core/state_manager.py)
- PredictionHandler (src/bot/ml_handlers/prediction_handlers.py)
