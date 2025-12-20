# Root Cause Analysis: Back Button "Cannot Go Back" Bug

## Problem Statement
Back buttons show "Cannot Go Back - You're at the beginning of the workflow" even when user is mid-workflow, specifically:
- After "Access Granted" for directory
- After "Schema Accepted"

## Investigation Timeline

### Evidence Collection

1. **Snapshot Saving Verification** ✅
   - Lines 222, 253, 452, 545, 664, 751, 840, 968, 1036, 1256, 4358, 4380 all call `save_state_snapshot()`
   - Test confirms: After first transition, history_depth=1 (snapshot WAS saved)
   - Prediction workflow (`/predict`) works correctly with same pattern

2. **Workflow Initialization Analysis** ⚠️
   - Line 124-128: Legacy workflow initialization (local_enabled=False)
   - Line 144-148: Modern workflow initialization (local_enabled=True)
   - **Both manually set state WITHOUT clearing old history**

### Root Cause Identified

**Location**: `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/ml_handlers/ml_training_local_path.py`

**Lines 144-148:**
```python
async def _show_data_source_selection(...):
    # Manually initialize workflow at CHOOSING_DATA_SOURCE state
    session.workflow_type = WorkflowType.ML_TRAINING
    session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
    await self.state_manager.update_session(session)  # NO history clear!
```

**Problem:**
When starting `/train`, the workflow manually sets the state but **does NOT clear the state_history** from any previous workflow. This causes:

1. User completes previous workflow → history depth = N
2. User starts `/train` → history still depth = N (old snapshots remain)
3. User progresses through new workflow → depth grows
4. User clicks back → Restores states from PREVIOUS workflow, not current one
5. Confusion: Back button behavior doesn't match current workflow

### Comparison with Working Code

**Prediction workflow (WORKS)** - `src/bot/ml_handlers/prediction_handlers.py:131`:
```python
session.current_state = MLPredictionState.STARTED.value
await self.state_manager.update_session(session)

# Show start message
await update.message.reply_text(...)

# IMMEDIATELY save snapshot before first transition
session.save_state_snapshot()  # Line 131
await self.state_manager.transition_state(...)
```

**Training workflow (BROKEN)** - `src/bot/ml_handlers/ml_training_local_path.py:144-148`:
```python
session.workflow_type = WorkflowType.ML_TRAINING
session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
await self.state_manager.update_session(session)
# NO snapshot save, NO history clear, directly shows UI
await update.message.reply_text(...)
```

### Why Snapshots ARE Being Saved But Bug Still Occurs

The snapshots ARE being saved at transition points (lines 222, 253, etc.). However:

1. **Old history persists**: Starting a new workflow doesn't clear old snapshots
2. **State mismatch**: Clicking back restores states from mixed workflows
3. **History pollution**: Depth accumulates across workflow restarts

### The Fix

**Required Changes:**

1. **Clear history when starting new workflow**:
   ```python
   # In _show_data_source_selection() and _start_telegram_upload_workflow()
   session.workflow_type = WorkflowType.ML_TRAINING
   session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
   session.clear_history()  # ADD THIS LINE
   await self.state_manager.update_session(session)
   ```

2. **Alternative: Use `start_workflow()` method** (recommended):
   ```python
   # Replace manual state assignment with:
   await self.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
   session.clear_history()  # Add history clear to start_workflow()
   ```

3. **Update `StateManager.start_workflow()`** to automatically clear history:
   ```python
   async def start_workflow(self, session: UserSession, workflow_type: WorkflowType) -> None:
       """Start a new workflow for session."""
       if session.workflow_type is not None:
           raise InvalidStateTransitionError(...)

       session.workflow_type = workflow_type
       session.current_state = list(initial_states)[0]
       session.clear_history()  # ADD THIS LINE
       await self._update_and_save(session)
   ```

4. **Update `cancel_workflow()`** to clear history:
   ```python
   async def cancel_workflow(self, session: UserSession) -> None:
       """Cancel active workflow."""
       session.workflow_type = None
       session.current_state = None
       session.selections.clear()
       session.clear_history()  # ADD THIS LINE
       await self._update_and_save(session)
   ```

## Test Strategy

1. Create test that starts multiple workflows in sequence
2. Verify history is cleared between workflows
3. Verify back button works correctly within each workflow
4. Verify "at beginning" message only shows at true beginning

## Files to Modify

1. `/Users/gkratka/Documents/statistical-modeling-agent/src/core/state_manager.py`
   - Add `session.clear_history()` to `start_workflow()` method
   - Add `session.clear_history()` to `cancel_workflow()` method

2. `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/ml_handlers/ml_training_local_path.py`
   - Add `session.clear_history()` in `_show_data_source_selection()` after line 147
   - Add `session.clear_history()` in `_start_telegram_upload_workflow()` after line 127

## Success Criteria

- ✅ Back button works at all mid-workflow stages
- ✅ "Cannot go back" only shows at true workflow beginning
- ✅ History cleared when starting new workflow
- ✅ No history pollution across workflow restarts
- ✅ All existing tests continue to pass
