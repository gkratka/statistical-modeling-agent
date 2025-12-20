# Drag-Drop File Upload Bug Fix

**Date**: 2025-12-12
**Status**: Fixed
**Bug**: Dragging/dropping files during prediction workflow causes bot to get stuck

---

## Root Cause Analysis

### The Problem
When users drag/drop a file to the chat during the `AWAITING_FILE_UPLOAD` state (ML prediction workflow), the bot sends a "Workflow active, cannot upload" message instead of processing the file. This happens even though the workflow is EXPECTING a file upload.

### Evidence Chain

1. **Handler Registration Conflict** (`telegram_bot.py`):
   - Line 385: General `document_handler` registered in group=0
   - Line 3347 (prediction_handlers.py): Prediction `handle_file_upload` also registered in group=0
   - **Problem**: Both handlers match `filters.Document.ALL`, but only the FIRST handler runs

2. **Handler Execution Order**:
   ```
   User uploads file
   ↓
   telegram_bot.py registers document_handler FIRST (line 385)
   ↓
   prediction_handlers.py registers handle_file_upload SECOND (line 3347)
   ↓
   When Update arrives, only FIRST handler executes (group=0 priority)
   ```

3. **General Document Handler Blocking Logic** (`main_handlers.py:675`):
   ```python
   if session.current_state is not None:
       # ❌ BLOCKS ALL uploads during ANY workflow
       await update.message.reply_text(
           Messages.workflow_active(...),
           parse_mode="Markdown"
       )
       return  # Stops propagation - prediction handler never runs
   ```

4. **Result**:
   - General handler sees `current_state == AWAITING_FILE_UPLOAD` → blocks upload
   - Prediction handler (which SHOULD process the file) never executes
   - User gets stuck with no way to proceed

---

## The Fix

### Part 1: Move Prediction Handler to Different Group

**File**: `src/bot/ml_handlers/prediction_handlers.py`
**Line**: 3346-3352

**Before**:
```python
# File upload handler
application.add_handler(
    MessageHandler(
        filters.Document.ALL,
        handler.handle_file_upload
    )
)  # Defaults to group=0 - CONFLICTS with general document_handler
```

**After**:
```python
# File upload handler (group=1 to avoid conflict with general document handler)
application.add_handler(
    MessageHandler(
        filters.Document.ALL,
        handler.handle_file_upload
    ),
    group=1  # ✅ Different from general document_handler (group=0)
)
```

### Part 2: Add State Filtering to General Document Handler

**File**: `src/bot/main_handlers.py`
**Line**: 663-684

**Before**:
```python
try:
    # Check if user has active workflow - prevent data upload during workflow
    state_manager = context.bot_data.get('state_manager')
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )

        # Get locale from session for i18n
        locale = session.language if session.language else None

        if session.current_state is not None:
            # ❌ BLOCKS ALL workflows without checking if they expect files
            await update.message.reply_text(
                Messages.workflow_active(
                    workflow_type=session.workflow_type.value if session.workflow_type else 'unknown',
                    current_state=session.current_state,
                    locale=locale
                ),
                parse_mode="Markdown"
            )
            return
```

**After**:
```python
try:
    # Check if user has active workflow
    state_manager = context.bot_data.get('state_manager')
    if state_manager:
        session = await state_manager.get_or_create_session(
            user_id,
            f"chat_{update.effective_chat.id}"
        )

        # Get locale from session for i18n
        locale = session.language if session.language else None

        # NEW: Skip blocking for prediction workflow file uploads
        # The prediction handler (group=1) will process these
        from src.core.state_manager import MLPredictionState
        file_upload_states = [
            MLPredictionState.AWAITING_FILE_UPLOAD.value,
        ]

        if session.current_state in file_upload_states:
            # ✅ Let prediction handler (group=1) process this file
            logger.info(
                f"📂 Skipping general document_handler - "
                f"prediction workflow expects file upload (state={session.current_state})"
            )
            return

        if session.current_state is not None:
            # Block uploads for non-file-expecting workflows
            await update.message.reply_text(
                Messages.workflow_active(
                    workflow_type=session.workflow_type.value if session.workflow_type else 'unknown',
                    current_state=session.current_state,
                    locale=locale
                ),
                parse_mode="Markdown"
            )
            return
```

---

## Handler Execution Flow After Fix

### Before Fix (BROKEN)
```
User drags file during AWAITING_FILE_UPLOAD
↓
Group 0 Handlers:
  1. document_handler (main_handlers.py) - FIRST
     → Checks: current_state != None ✓
     → Sends: "Workflow active, cannot upload" ❌
     → Returns (stops propagation)
  2. handle_file_upload (prediction_handlers.py) - SECOND
     → NEVER RUNS because first handler stopped propagation
```

### After Fix (WORKING)
```
User drags file during AWAITING_FILE_UPLOAD
↓
Group 0 Handlers:
  1. document_handler (main_handlers.py)
     → Checks: current_state in file_upload_states? ✓
     → Skips processing (returns early)
     → Falls through to next handler
↓
Group 1 Handlers:
  2. handle_file_upload (prediction_handlers.py)
     → Processes file correctly ✅
     → Transitions to CONFIRMING_SCHEMA ✅
     → Workflow continues ✅
```

---

## Testing

### Test Case 1: Prediction Workflow File Upload
```python
1. User: /predict
2. Bot: "Choose data source"
3. User: Clicks "Upload File"
4. Bot: "Please upload your CSV file"
5. User: Drags german_credit_data_train2.csv
6. ✅ BEFORE: Bot says "Workflow active, cannot upload"
7. ✅ AFTER: Bot processes file and shows schema confirmation
```

### Test Case 2: General File Upload (No Workflow)
```python
1. User: (no active workflow)
2. User: Drags german_credit_data.csv
3. ✅ Bot processes file normally
4. ✅ Bot stores data and shows summary
```

### Test Case 3: File Upload During Non-File-Expecting Workflow
```python
1. User: /train
2. Bot: "Select target column"
3. User: Drags file
4. ✅ Bot blocks upload: "Workflow active, please complete or /cancel"
```

---

## Verification Steps

1. **Check handler registration order**:
   ```bash
   grep -n "add_handler.*Document" src/bot/telegram_bot.py src/bot/ml_handlers/prediction_handlers.py
   ```

2. **Verify group assignments**:
   - General document handler: group=0 (default)
   - Prediction file upload handler: group=1 (explicit)
   - Text message handlers: group=1 and group=2 (no conflict)

3. **Test drag-drop during prediction**:
   - Start prediction workflow
   - Choose "Upload File"
   - Drag/drop CSV file
   - Verify bot processes file (not blocks it)

---

## Files Modified

1. `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/ml_handlers/prediction_handlers.py`
   - Line 3346-3352: Added `group=1` to file upload handler registration

2. `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/main_handlers.py`
   - Line 663-684: Added state filtering to skip blocking for file upload states

---

## Prevention

**Rule**: When registering multiple `MessageHandler` instances with overlapping filters:
1. Use different handler groups to control execution order
2. Add state filtering to prevent handler collisions
3. Document which states each handler is responsible for
4. Test with realistic user flows (drag-drop, not just upload button clicks)

**Code Review Checklist**:
- [ ] Do any handlers share the same filter in the same group?
- [ ] Are workflow-specific handlers isolated from general handlers?
- [ ] Does the general handler check workflow state before blocking?
- [ ] Are handler groups documented in registration comments?

---

## Related Issues

- Handler collision between workflow-specific and general handlers
- Workflow state not checked before blocking file uploads
- Missing group assignment in handler registration
- Drag-drop upload different from upload button (both use same handler)

**Tags**: #bug-fix #handler-collision #file-upload #prediction-workflow #state-filtering
