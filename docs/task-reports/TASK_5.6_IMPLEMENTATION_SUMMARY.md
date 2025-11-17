# Task 5.6: State-Based Error Handling - Implementation Summary

## Overview
Implemented comprehensive error handling with ERROR_STATE transitions, automatic retry logic, escalation paths, and a /recover command for all workflows.

## Implementation Details

### 1. ERROR_STATE Added to All Workflow Enums
- **MLTrainingState.ERROR_STATE** = "error_state"
- **MLPredictionState.ERROR_STATE** = "error_state"
- **CloudTrainingState.ERROR_STATE** = "error_state"
- **CloudPredictionState.ERROR_STATE** = "error_state"
- **ScoreWorkflowState.ERROR_STATE** = "error_state"

### 2. Universal Error Transition Rules
**Any state → ERROR_STATE**: Allowed from all workflow states
**ERROR_STATE → Any state**: Allowed for recovery (retry to previous state, restart to initial state)

Implementation: Modified `StateMachine.validate_transition()` to bypass standard validation for error state transitions.

### 3. ErrorContext Dataclass
Located in `src/core/state_manager.py`

```python
@dataclass
class ErrorContext:
    error_message: str
    stack_trace: str
    user_id: int
    timestamp: datetime
    previous_state: str
    workflow_type: str
    retry_count: int = 0
    is_transient: bool = False
    escalation_path: Optional[str] = None
```

Methods:
- `to_dict()`: Serialize for storage
- `from_dict()`: Deserialize from storage

### 4. StateManager Error Handling Methods

#### `error_state(session, error_message, stack_trace, is_transient) -> List[str]`
- Creates ErrorContext
- Transitions to ERROR_STATE
- Logs error with full context
- Returns recovery options: ["retry", "restart", "cancel", "switch_to_local"]

#### `retry_transient_error(session) -> bool`
- Implements exponential backoff: 1s, 2s, 4s
- Max 3 retry attempts (configurable)
- Auto-escalates to persistent after max attempts
- Returns True if retry successful, False if max exceeded

#### `escalate_persistent_error(session) -> List[str]`
- Logs ERROR level for admin notification
- Offers cloud→local fallback for cloud workflows
- Stores escalation path in error_context
- Returns escalation options

#### `get_recovery_info(session) -> Optional[Dict]`
- Returns current error state and recovery options
- Used by /recover command
- Returns None if no error active

#### `execute_recovery_action(session, action) -> bool`
- Executes user-selected recovery action
- Actions: "Retry", "Restart Workflow", "Switch to Local", "Cancel"
- Clears error_context on successful recovery

### 5. Error Classification System
Located in `src/utils/exceptions.py`

**Transient Errors** (auto-retry with backoff):
- NetworkError
- APIRateLimitError
- TemporaryUnavailableError
- ConnectionError, TimeoutError
- Some CloudError/S3Error instances

**Persistent Errors** (require manual intervention):
- ValidationError
- AuthenticationError
- PermissionDeniedError
- ResourceNotFoundError
- InvalidConfigurationError
- DataCorruptionError
- ModelTrainingError

Function: `classify_error(error: Exception) -> str`
Returns "transient" or "persistent"

### 6. Configuration
Added to `StateManagerConfig`:
- `max_retry_attempts: int = 3`
- `retry_base_delay: float = 1.0` (seconds)

### 7. Bot Handlers
Created `src/bot/handlers/error_recovery_handlers.py`:

#### `/recover` Command Handler
- Shows current error state
- Displays recovery options as inline keyboard
- Handles recovery action execution

#### Helper Function: `handle_workflow_error()`
- Automatic error handling for workflow handlers
- Classifies errors
- Transitions to ERROR_STATE
- Sends user-friendly error notification

### 8. User Messages
Created `src/bot/messages/error_messages.py`:

Functions:
- `format_error_state_message()`: Error notification
- `format_recovery_options_message()`: /recover display
- `format_retry_message()`: Retry attempt status
- `format_escalation_message()`: Escalation notifications
- `format_recovery_success_message()`: Recovery completion
- `format_workflow_cancelled_message()`: Cancellation confirmation

## Test Coverage

### Unit Tests (40 tests)
File: `tests/unit/test_error_handling.py`

1. **TestErrorStateEnums** (5 tests): ERROR_STATE in all workflow enums
2. **TestErrorStateTransitions** (5 tests): Universal error transitions
3. **TestErrorContext** (4 tests): Dataclass serialization
4. **TestErrorStateMethod** (4 tests): error_state() functionality
5. **TestTransientErrorRetry** (5 tests): Retry logic and exponential backoff
6. **TestPersistentErrorEscalation** (4 tests): Escalation paths
7. **TestRecoverCommand** (8 tests): /recover command functionality
8. **TestErrorTypeClassification** (5 tests): Error classification

### Integration Tests (5 tests)
File: `tests/unit/test_error_handling_integration.py`

1. Complete transient error flow (error → retry → success)
2. Complete persistent error flow (error → restart)
3. Max retries escalation
4. Cancel recovery
5. Error classification integration

**Total: 45 tests, all passing**

## Workflow Examples

### Transient Error Recovery
```
1. User in ML_TRAINING workflow at TRAINING state
2. Network error occurs
3. System calls: await manager.error_state(session, "Network timeout", trace, is_transient=True)
4. Session transitions to ERROR_STATE
5. User receives error notification with /recover option
6. User runs /recover
7. System offers: [Retry, Restart Workflow, Cancel]
8. User selects "Retry"
9. System waits 1 second (exponential backoff)
10. Session returns to TRAINING state
11. User continues workflow
```

### Persistent Error Escalation
```
1. User encounters validation error
2. System transitions to ERROR_STATE (is_transient=False)
3. Admin notification logged (ERROR level)
4. User runs /recover
5. System offers: [Restart Workflow, Cancel]
6. User selects "Restart Workflow"
7. Session resets to initial state
8. Error context cleared
9. User starts fresh workflow
```

### Cloud to Local Fallback
```
1. User in CLOUD_TRAINING workflow
2. Cloud launch fails (persistent error)
3. System escalates with cloud→local option
4. User runs /recover
5. System offers: [Restart Workflow, Switch to Local, Cancel]
6. User selects "Switch to Local"
7. Workflow switches to LOCAL_TRAINING
8. Session resets to local training initial state
9. User continues with local training
```

## Files Created
- `tests/unit/test_error_handling.py` (40 tests)
- `tests/unit/test_error_handling_integration.py` (5 tests)
- `src/bot/handlers/error_recovery_handlers.py` (handlers)
- `src/bot/messages/error_messages.py` (message templates)
- `TASK_5.6_IMPLEMENTATION_SUMMARY.md` (this file)

## Files Modified
- `src/core/state_manager.py`:
  - Added ERROR_STATE to all workflow state enums
  - Added ErrorContext dataclass
  - Added error_context field to UserSession
  - Added error handling methods (6 new methods)
  - Modified StateMachine.validate_transition() for error transitions
  - Added configuration fields to StateManagerConfig

- `src/utils/exceptions.py`:
  - Added 9 new exception classes
  - Added TRANSIENT_ERRORS and PERSISTENT_ERRORS sets
  - Added classify_error() function

## Task Completion Checklist

- [x] Add ERROR_STATE enum value to all workflow state enums
- [x] Update workflow transitions to allow: any state → ERROR_STATE
- [x] ERROR_STATE can transition to: previous state (retry) or initial state (restart)
- [x] Create ErrorContext dataclass with all required fields
- [x] ErrorContext.to_dict() and from_dict() methods
- [x] Store ErrorContext in UserSession.error_context
- [x] Add error_state() method to StateManager
- [x] error_state() creates ErrorContext
- [x] error_state() transitions to ERROR_STATE
- [x] error_state() logs error with full context
- [x] error_state() returns recovery options
- [x] Implement retry_transient_error() with exponential backoff
- [x] Max retry attempts: 3 (configurable)
- [x] Exponential backoff: 1s, 2s, 4s
- [x] Auto-escalate after max retries
- [x] Implement escalate_persistent_error()
- [x] Escalation: admin notification (ERROR logging)
- [x] Escalation: cloud→local fallback option
- [x] Escalation: manual intervention option
- [x] Create /recover command handler
- [x] /recover shows error context
- [x] /recover displays recovery options
- [x] /recover allows user to select recovery action
- [x] Implement execute_recovery_action()
- [x] Recovery actions: Retry, Restart, Switch to Local, Cancel
- [x] Error classification: transient vs persistent
- [x] Add new exception classes for error types
- [x] Write comprehensive unit tests (40 tests)
- [x] Write integration tests (5 tests)
- [x] All tests passing (45/45)

## Next Steps
After Task 5.6 completion:
1. Run all Section 5.0 tests
2. Commit changes with message: "feat: Complete state-based error handling (Task 5.6)"
3. Section 5.0 is COMPLETE
