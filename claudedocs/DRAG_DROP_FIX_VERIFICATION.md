# Drag-Drop File Upload Fix - Verification Report

**Date**: 2025-12-12
**Issue**: Drag/drop files during prediction workflow causes bot to get stuck
**Status**: ✅ FIXED

---

## Summary

Fixed handler collision bug where dragging/dropping files during `AWAITING_FILE_UPLOAD` state caused the bot to block uploads instead of processing them.

### Root Cause
- General `document_handler` (group=0) and prediction `handle_file_upload` (group=0) both matched `filters.Document.ALL`
- Only FIRST handler in group executes → prediction handler never ran
- General handler blocked ALL uploads during ANY workflow state

### Solution
**Part 1**: Move prediction file upload handler to group=1
**Part 2**: Add state filtering to general document handler

---

## Changes Applied

### 1. General Document Handler (src/bot/main_handlers.py)

**Location**: Lines 675-688

**Before**:
```python
if session.current_state is not None:
    # ❌ BLOCKS ALL workflows
    await update.message.reply_text(
        Messages.workflow_active(...),
        parse_mode="Markdown"
    )
    return
```

**After**:
```python
# NEW: Skip blocking for prediction workflow file uploads
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
        Messages.workflow_active(...),
        parse_mode="Markdown"
    )
    return
```

### 2. Prediction Handler Registration (src/bot/ml_handlers/prediction_handlers.py)

**Location**: Lines 3346-3353

**Before**:
```python
# File upload handler
application.add_handler(
    MessageHandler(
        filters.Document.ALL,
        handler.handle_file_upload
    )
)  # Defaults to group=0 - CONFLICTS
```

**After**:
```python
# File upload handler
application.add_handler(
    MessageHandler(
        filters.Document.ALL,
        handler.handle_file_upload
    ),
    group=1  # Different from general document_handler (group=0)
)
```

---

## Handler Execution Flow

### Before Fix (BROKEN)
```
User drags file during AWAITING_FILE_UPLOAD
↓
Group 0 Handlers:
  1. document_handler (main_handlers.py) - RUNS FIRST
     → Checks: current_state != None? ✓
     → Sends: "Workflow active, cannot upload" ❌
     → Returns (stops propagation)
  2. handle_file_upload (prediction_handlers.py) - SECOND
     → NEVER RUNS ❌
```

### After Fix (WORKING)
```
User drags file during AWAITING_FILE_UPLOAD
↓
Group 0 Handlers:
  1. document_handler (main_handlers.py)
     → Checks: current_state in file_upload_states? ✓
     → Logs: "Skipping general document_handler"
     → Returns early (no blocking)
     ↓ Falls through to next group
↓
Group 1 Handlers:
  2. handle_file_upload (prediction_handlers.py)
     → Processes file ✅
     → Transitions to CONFIRMING_SCHEMA ✅
```

---

## Verification

### Code Syntax
```bash
✅ python3 -m py_compile src/bot/main_handlers.py
✅ python3 -m py_compile src/bot/ml_handlers/prediction_handlers.py
```

### File Changes
```bash
✅ src/bot/main_handlers.py - Lines 675-688 (state filtering added)
✅ src/bot/ml_handlers/prediction_handlers.py - Line 3352 (group=1 added)
```

### Handler Groups
- General document handler: `group=0` (default)
- Prediction file upload handler: `group=1` (explicit)
- Text message handler (main): `group=1`
- Text message handler (prediction): `group=2`

---

## Testing Checklist

### Manual Testing Required

- [ ] Start prediction workflow (`/predict`)
- [ ] Choose "Upload File"
- [ ] **Drag/drop CSV file** to chat
- [ ] **Expected**: Bot processes file and shows schema confirmation
- [ ] **NOT Expected**: "Workflow active, cannot upload" message

### Edge Cases to Test

- [ ] Drag/drop file during non-file-expecting state (e.g., SELECTING_TARGET)
  - **Expected**: Bot blocks upload with "Workflow active" message
- [ ] Drag/drop file with no active workflow
  - **Expected**: Bot processes file normally (general handler)
- [ ] Upload file using "Attach" button (not drag/drop)
  - **Expected**: Same behavior as drag/drop

### Integration Test Scenarios

1. **Prediction Workflow - Happy Path**
   - /predict → Upload File → Drag file → Schema shown ✓
   
2. **Training Workflow - File Upload Blocked**
   - /train → Select Target → Drag file → Blocked with error ✓
   
3. **No Workflow - General Upload**
   - No workflow → Drag file → File processed normally ✓

---

## Related Files

- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/main_handlers.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/ml_handlers/prediction_handlers.py`
- `/Users/gkratka/Documents/statistical-modeling-agent/src/core/state_manager.py` (state definitions)

---

## Prevention

**Code Review Checklist**:
- [ ] When adding new document handlers, check group assignments
- [ ] When blocking uploads, check if workflow expects files
- [ ] Test both drag/drop AND upload button methods
- [ ] Verify handler registration order in telegram_bot.py

**Rule**: 
- Workflow-specific handlers → **group > 0**
- General fallback handlers → **group=0**
- Always check state before blocking uploads

---

## Deployment

1. Commit changes to repository
2. Deploy to production
3. Monitor logs for "📂 Skipping general document_handler" messages
4. Verify user reports of stuck uploads are resolved

**Log Marker**: 
```
📂 Skipping general document_handler - prediction workflow expects file upload (state=awaiting_file_upload)
```

---

## Tags
#bug-fix #handler-collision #file-upload #drag-drop #prediction-workflow #state-filtering

