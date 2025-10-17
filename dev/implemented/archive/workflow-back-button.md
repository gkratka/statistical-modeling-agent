# Workflow Back Button Implementation Plan

**Status**: Approved - Implementation in Progress
**Estimated Effort**: 15-21 hours
**Created**: 2025-10-11

## Executive Summary

This document outlines the implementation plan for adding back button navigation to bot workflows, allowing users to:
- Navigate backward in multi-step workflows
- Adjust parameters without restarting
- Load different datasets mid-workflow
- Switch model types before training

**Critical Requirement**: When users go back, previous choices are **not retained** (clean slate on rollback).

---

## Architecture Overview

### State History System

The implementation uses a **LIFO (Last-In-First-Out) stack** for state snapshots with memory optimization:

```python
class StateSnapshot:
    """Immutable snapshot of workflow state at a point in time."""

    def __init__(self, state: ConversationSession):
        # Core state
        self.step = state.step
        self.workflow = state.workflow
        self.timestamp = time.time()

        # Memory-optimized data handling
        # Shallow copy: DataFrame reference only (avoid memory explosion)
        self.data_ref = state.data
        self.data_hash = hash(id(state.data))  # Track mutations

        # Deep copy: Small objects (selections, parameters)
        self.selections = copy.deepcopy({
            'selected_target': state.selected_target,
            'selected_features': state.selected_features,
            'selected_model_type': state.selected_model_type,
            'selected_task_type': state.selected_task_type,
        })

        # Metadata
        self.file_path = state.file_path
        self.detected_schema = copy.deepcopy(state.detected_schema)


class StateHistory:
    """LIFO stack for state snapshots with depth limit."""

    def __init__(self, max_depth: int = 10):
        self.history: List[StateSnapshot] = []
        self.max_depth = max_depth

    def push(self, snapshot: StateSnapshot) -> None:
        """Push snapshot to history (circular buffer)."""
        self.history.append(snapshot)
        if len(self.history) > self.max_depth:
            self.history.pop(0)  # Remove oldest

    def pop(self) -> Optional[StateSnapshot]:
        """Pop and return previous state."""
        return self.history.pop() if self.history else None

    def can_go_back(self) -> bool:
        """Check if back navigation is possible."""
        return len(self.history) > 0

    def clear(self) -> None:
        """Clear all history."""
        self.history.clear()
```

### Memory Optimization Strategy

**Problem**: Deep copying large DataFrames (e.g., 799 rows × 20 columns) on every state transition with 10-level history = memory explosion.

**Solution**: Hybrid copying approach:
- **Shallow Copy** (reference): DataFrames, large objects
- **Deep Copy** (clone): Selections, parameters, small objects

**Result**: ~90% memory reduction, <5MB per session vs potential 50MB+

---

## State Transition Graph

### ML Training Workflow States

```
IDLE
  ↓ /train
CHOOSING_DATA_SOURCE
  ↓ (local)                    ↓ (upload)
AWAITING_FILE_PATH          AWAITING_FILE_UPLOAD
  ↓                              ↓
FILE_PATH_RECEIVED          FILE_UPLOADED
  ↓                              ↓
CONFIRMING_SCHEMA ←──────────────┘
  ↓ (accept)         ↓ (defer)
AWAITING_TARGET    DEFERRED_SCHEMA_PENDING
  ↓
AWAITING_FEATURES
  ↓
AWAITING_MODEL_TYPE
  ↓
TRAINING_IN_PROGRESS
  ↓
TRAINING_COMPLETE
```

### Back Button Eligibility

| State | Back Button | Restores To | Fields Cleared |
|-------|-------------|-------------|----------------|
| IDLE | ❌ No | - | - |
| CHOOSING_DATA_SOURCE | ❌ No (entry) | - | - |
| AWAITING_FILE_PATH | ✅ Yes | CHOOSING_DATA_SOURCE | file_path, data, ALL selections |
| AWAITING_FILE_UPLOAD | ✅ Yes | CHOOSING_DATA_SOURCE | file_path, data, ALL selections |
| CONFIRMING_SCHEMA | ✅ Yes | Previous (local/upload) | detected_schema |
| AWAITING_TARGET | ✅ Yes | CONFIRMING_SCHEMA | selected_target |
| AWAITING_FEATURES | ✅ Yes | AWAITING_TARGET | selected_features |
| AWAITING_MODEL_TYPE | ✅ Yes | AWAITING_FEATURES | selected_model_type |
| TRAINING_IN_PROGRESS | ❌ No (processing) | - | - |
| TRAINING_COMPLETE | ✅ Yes | IDLE (restart) | ALL workflow data |

---

## State Cleanup Strategy

Each backward transition defines which fields to clear to ensure "not retain previous choices":

```python
# State cleanup map
CLEANUP_MAP = {
    'CHOOSING_DATA_SOURCE': [
        'file_path', 'data', 'detected_schema',
        'selected_target', 'selected_features',
        'selected_model_type', 'selected_task_type'
    ],
    'AWAITING_FILE_PATH': ['file_path', 'data', 'detected_schema'],
    'AWAITING_FILE_UPLOAD': ['file_path', 'data', 'detected_schema'],
    'CONFIRMING_SCHEMA': ['detected_schema'],
    'AWAITING_TARGET': ['selected_target'],
    'AWAITING_FEATURES': ['selected_features'],
    'AWAITING_MODEL_TYPE': ['selected_model_type'],
}

def restore_previous_state(self) -> bool:
    """Restore previous state and clear downstream fields."""
    previous = self.state_history.pop()
    if not previous:
        return False

    # Restore state
    self.step = previous.step
    self.workflow = previous.workflow

    # Restore selections (from deep copy)
    for key, value in previous.selections.items():
        setattr(self, key, value)

    # Clear fields set AFTER this state
    fields_to_clear = CLEANUP_MAP.get(self.step, [])
    self.clear_fields(fields_to_clear)

    return True
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (4-6 hours)

**Goal**: Implement state history system foundation

**Tasks**:
1. Create `src/core/state_history.py` with StateSnapshot and StateHistory classes
2. Modify `src/core/state_manager.py`:
   - Add `state_history: StateHistory` to ConversationSession
   - Implement `save_state_snapshot()` method
   - Implement `restore_previous_state()` method
   - Implement `clear_fields()` method
3. Create `tests/unit/test_state_history.py`:
   - Test snapshot creation
   - Test push/pop operations
   - Test depth limiting (max 10)
   - Test serialization
   - Test memory optimization

**Deliverables**:
- ✅ StateHistory system functional
- ✅ Unit tests passing (100% coverage)
- ✅ Memory usage validated (<5MB per session)

**Code Example**:
```python
# src/core/state_manager.py modification
class ConversationSession:
    def __init__(self, user_id: int, conversation_id: str):
        # ... existing attributes ...
        self.state_history = StateHistory(max_depth=10)  # NEW
        self.last_back_action = None  # For debouncing

    def save_state_snapshot(self) -> None:
        """Save current state before transition."""
        snapshot = StateSnapshot(self)
        self.state_history.push(snapshot)

    def restore_previous_state(self) -> bool:
        """Restore previous state and clear downstream fields."""
        previous = self.state_history.pop()
        if not previous:
            return False

        # Restore core state
        self.step = previous.step
        self.workflow = previous.workflow

        # Restore selections
        for key, value in previous.selections.items():
            setattr(self, key, value)

        # Clear fields set after this state
        fields_to_clear = CLEANUP_MAP.get(self.step, [])
        self.clear_fields(fields_to_clear)

        return True

    def clear_fields(self, field_list: List[str]) -> None:
        """Clear specified state fields."""
        for field in field_list:
            if hasattr(self, field):
                setattr(self, field, None)
```

---

### Phase 2: UI Integration (3-4 hours)

**Goal**: Add back buttons to all workflow UIs

**Tasks**:
1. Modify `src/bot/messages/local_path_messages.py`:
   - Create `create_back_button()` utility
   - Create `add_back_button()` helper
2. Modify `src/bot/handlers.py`:
   - Create universal `handle_workflow_back()` callback handler
   - Add callback routing for `callback_data="workflow_back"`
3. Modify `src/bot/ml_handlers/ml_training_local_path.py`:
   - Add back buttons to all eligible state keyboards
   - Call `session.save_state_snapshot()` before transitions

**Deliverables**:
- ✅ Back button visible on all eligible states
- ✅ Callback routing functional
- ✅ State snapshots saved on transitions

**Code Example**:
```python
# src/bot/messages/local_path_messages.py
def create_back_button() -> InlineKeyboardButton:
    """Create standardized back button."""
    return InlineKeyboardButton("⬅️ Back", callback_data="workflow_back")

def add_back_button(keyboard: List[List[InlineKeyboardButton]]) -> None:
    """Add back button to keyboard layout."""
    keyboard.append([create_back_button()])

# src/bot/handlers.py
async def handle_workflow_back(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Universal back button handler with debouncing."""
    query = update.callback_query
    user_id = query.from_user.id

    # Get session
    session = await state_manager.get_session(user_id)

    # Debouncing check (500ms cooldown)
    if session.last_back_action:
        elapsed = time.time() - session.last_back_action
        if elapsed < 0.5:
            await query.answer("⚠️ Please wait...")
            return

    session.last_back_action = time.time()

    # Restore previous state
    if session.restore_previous_state():
        # Route to appropriate workflow renderer
        if session.workflow == "ml_training":
            await ml_training_handler.render_current_state(update, context)
        await query.answer("↩️ Returned to previous step")
    else:
        await query.answer("❌ Cannot go back further")

# src/bot/ml_handlers/ml_training_local_path.py
async def handle_target_selection(update: Update, context):
    """Handle target column selection."""
    session = state_manager.get_session(user_id)

    # Save state snapshot BEFORE transition
    session.save_state_snapshot()

    # Process selection
    session.selected_target = selected_column
    session.step = ConversationState.AWAITING_FEATURE_SELECTION

    # Render next step with back button
    keyboard = create_feature_selection_keyboard(...)
    add_back_button(keyboard)  # Add back button
    await update.message.reply_text("Select features:", reply_markup=keyboard)
```

---

### Phase 3: State Cleanup Logic (3-4 hours)

**Goal**: Implement field clearing to ensure clean slate

**Tasks**:
1. Define `CLEANUP_MAP` in `src/core/state_manager.py`
2. Implement cleanup logic in `restore_previous_state()`
3. Create tests for state cleanup validation
4. Test multi-level back navigation (e.g., back 3 steps)

**Deliverables**:
- ✅ Cleanup map complete for all states
- ✅ Fields properly cleared on rollback
- ✅ Multi-level navigation validated

**Test Cases**:
```python
# tests/unit/test_state_cleanup.py
def test_back_from_target_clears_target():
    """Back from AWAITING_FEATURES clears selected_target."""
    session = create_session()
    session.step = ConversationState.AWAITING_FEATURES
    session.selected_target = "price"
    session.save_state_snapshot()

    session.step = ConversationState.AWAITING_MODEL_TYPE
    session.selected_features = ["sqft", "bedrooms"]

    # Go back
    session.restore_previous_state()

    assert session.step == ConversationState.AWAITING_FEATURES
    assert session.selected_target == "price"  # Restored
    assert session.selected_features is None  # Cleared

def test_back_to_source_clears_all():
    """Back to CHOOSING_DATA_SOURCE clears all downstream data."""
    session = create_session()
    session.step = ConversationState.CHOOSING_DATA_SOURCE
    session.save_state_snapshot()

    # Simulate full workflow
    session.file_path = "/data/housing.csv"
    session.data = pd.DataFrame(...)
    session.selected_target = "price"
    session.selected_features = ["sqft"]

    # Go back
    session.restore_previous_state()

    assert session.step == ConversationState.CHOOSING_DATA_SOURCE
    assert session.file_path is None
    assert session.data is None
    assert session.selected_target is None
    assert session.selected_features is None
```

---

### Phase 4: Error Handling (2-3 hours)

**Goal**: Add robustness and user safety

**Tasks**:
1. Implement debouncing (500ms cooldown on back button)
2. Add state locking during transitions
3. Handle edge cases:
   - Empty history stack
   - Corrupted state data
   - Concurrent back requests
4. Add user-friendly error messages
5. Implement workflow restart on critical errors

**Deliverables**:
- ✅ Debouncing prevents race conditions
- ✅ State locking prevents corruption
- ✅ Graceful error handling
- ✅ User feedback for all error cases

**Code Example**:
```python
# Debouncing implementation
async def handle_workflow_back(update, context):
    session = state_manager.get_session(user_id)

    # Check cooldown
    if session.last_back_action:
        elapsed = time.time() - session.last_back_action
        if elapsed < 0.5:  # 500ms
            await query.answer("⚠️ Please wait before going back again...")
            return

    session.last_back_action = time.time()

    # State locking
    async with session.transition_lock:
        if session.restore_previous_state():
            await render_current_state(update, context)
        else:
            # Empty history - offer workflow restart
            await query.answer("❌ Cannot go back further")
            keyboard = [[
                InlineKeyboardButton("🔄 Restart Workflow",
                                     callback_data="workflow_restart")
            ]]
            await update.message.reply_text(
                "You're at the beginning of the workflow.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
```

---

### Phase 5: Testing & Validation (3-4 hours)

**Goal**: Comprehensive test coverage and validation

**Tasks**:
1. Create `tests/integration/test_workflow_back_button.py`
2. Test back navigation from each state
3. Test multi-level back (back 2-5 steps)
4. Test data cleanup validation
5. Test memory usage (load testing)
6. Test concurrent user scenarios
7. Manual testing with real Telegram bot

**Test Coverage Target**: >90%

**Deliverables**:
- ✅ Integration tests passing
- ✅ Manual testing completed
- ✅ Performance validated (<5MB/session)
- ✅ User acceptance testing passed

**Integration Test Example**:
```python
# tests/integration/test_workflow_back_button.py
@pytest.mark.asyncio
class TestWorkflowBackButton:
    """Integration tests for back button navigation."""

    async def test_back_from_feature_selection(self, mock_bot):
        """Test back from feature selection to target selection."""
        # Start workflow
        await mock_bot.send_message("/train")
        await mock_bot.click_button("data_source_local")
        await mock_bot.send_message("/path/to/data.csv")
        await mock_bot.click_button("accept_schema")
        await mock_bot.click_button("select_target_0")  # Select target

        # At feature selection
        response = await mock_bot.get_last_message()
        assert "Select features" in response

        # Click back button
        await mock_bot.click_button("workflow_back")

        # Should be back at target selection
        response = await mock_bot.get_last_message()
        assert "Select target" in response

        # Verify target was cleared
        session = state_manager.get_session(mock_bot.user_id)
        assert session.selected_target is None

    async def test_multi_level_back(self, mock_bot):
        """Test going back multiple steps."""
        # Progress to model type selection
        await mock_bot.complete_workflow_to_model_type()

        # Go back 3 times: model_type → features → target
        await mock_bot.click_button("workflow_back")  # Back to features
        await mock_bot.click_button("workflow_back")  # Back to target
        await mock_bot.click_button("workflow_back")  # Back to schema

        response = await mock_bot.get_last_message()
        assert "Detected schema" in response

        # Verify all selections cleared
        session = state_manager.get_session(mock_bot.user_id)
        assert session.selected_target is None
        assert session.selected_features is None
        assert session.selected_model_type is None
```

---

## Technical Decisions & Trade-offs

### Decision 1: LIFO Stack vs. Full History Graph

**Chosen**: LIFO Stack
**Rationale**: Workflows are primarily linear with occasional branching. Stack handles branching naturally by tracking actual path taken.
**Trade-off**: Cannot jump to arbitrary states, only sequential back navigation.
**Alternative Considered**: Full state graph with arbitrary jumps - rejected due to complexity and UX confusion.

### Decision 2: Shallow Copy for DataFrames

**Chosen**: Shallow copy (reference only)
**Rationale**: 799-row DataFrames × 10 snapshots = 40MB+ memory usage. Shallow copy reduces to <5MB.
**Trade-off**: If user modifies data mid-workflow, all snapshots affected. Mitigated by data immutability pattern.
**Alternative Considered**: Deep copy all data - rejected due to memory explosion.

### Decision 3: Max History Depth = 10

**Chosen**: 10 snapshots
**Rationale**: Typical workflow is 4-6 steps. 10 provides safety margin while preventing unbounded memory growth.
**Trade-off**: Cannot go back >10 steps (rare case).
**Alternative Considered**: Unlimited history - rejected due to memory concerns.

### Decision 4: Debouncing = 500ms

**Chosen**: 500ms cooldown
**Rationale**: Prevents accidental double-clicks, balances UX (not too slow) vs. safety.
**Trade-off**: Very fast users might notice slight delay.
**Alternative Considered**: 1000ms (too slow), 200ms (too permissive).

### Decision 5: State Cleanup Map

**Chosen**: Explicit cleanup map per state
**Rationale**: Clear, maintainable, ensures "not retain previous choices" requirement.
**Trade-off**: Requires manual maintenance when adding new workflow states.
**Alternative Considered**: Auto-detect downstream fields - rejected due to fragility.

---

## Success Criteria

### Functional Requirements

- ✅ Back button visible on all eligible workflow states
- ✅ Back navigation restores previous state correctly
- ✅ Previous choices are NOT retained (clean slate)
- ✅ Multi-level back navigation works (back 2-5 steps)
- ✅ Edge cases handled gracefully (empty history, errors)
- ✅ Debouncing prevents race conditions

### Performance Requirements

- ✅ Memory usage <5MB per active session
- ✅ Back navigation latency <200ms
- ✅ No memory leaks over extended sessions
- ✅ Supports 100+ concurrent users

### Quality Requirements

- ✅ Unit test coverage >90%
- ✅ Integration test coverage >85%
- ✅ Zero critical bugs in UAT
- ✅ User feedback positive (manual testing)

### Documentation Requirements

- ✅ Implementation plan documented (this file)
- ✅ Code comments for complex logic
- ✅ User-facing error messages clear
- ✅ CLAUDE.md updated with back button workflow

---

## Rollback Plan

If critical issues arise during implementation:

### Phase 1-2 Rollback
- Remove `state_history` from ConversationSession
- Remove back button UI elements
- System returns to pre-implementation state
- **Impact**: Low (no user-facing changes yet)

### Phase 3-4 Rollback
- Disable back button callback handler
- Hide back buttons from UI (comment out)
- Keep state_history code for future retry
- **Impact**: Medium (UI changes visible but disabled)

### Phase 5 Rollback
- Feature flag to disable back button system
- Gradual rollback: disable per workflow first
- Monitor for data corruption issues
- **Impact**: High (feature partially deployed)

---

## Future Enhancements

### Post-MVP Improvements

1. **Undo/Redo System**: Beyond back button, full undo/redo like code editors
2. **Workflow Branching Visualization**: Show user their path through workflow
3. **Save Workflow Progress**: Persist state across bot restarts
4. **Quick Jump**: Jump to specific workflow step (not just sequential back)
5. **Workflow Templates**: Save common workflows for quick replay

### Performance Optimizations

1. **Lazy Snapshot Creation**: Only snapshot when user might go back
2. **Compression**: Compress snapshots for long-running sessions
3. **Distributed State**: Redis for state storage in multi-instance setup

---

## Implementation Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| Phase 1 | 4-6 hours | None | StateHistory system |
| Phase 2 | 3-4 hours | Phase 1 | Back button UI |
| Phase 3 | 3-4 hours | Phase 2 | State cleanup logic |
| Phase 4 | 2-3 hours | Phase 3 | Error handling |
| Phase 5 | 3-4 hours | Phase 4 | Testing & validation |
| **Total** | **15-21 hours** | - | **Fully functional back button** |

---

## Conclusion

This implementation plan provides a robust, memory-efficient solution for back button navigation in bot workflows. The phased approach allows for incremental validation and rollback at each stage.

**Key Strengths**:
- ✅ Meets "clean slate" requirement with explicit cleanup
- ✅ Memory-optimized (<5MB per session)
- ✅ Handles edge cases and race conditions
- ✅ Comprehensive testing strategy
- ✅ Clear rollback plan

**Approval Status**: ✅ APPROVED by user on 2025-10-11

**Implementation Status**: 🔄 IN PROGRESS (starting Phase 1)
