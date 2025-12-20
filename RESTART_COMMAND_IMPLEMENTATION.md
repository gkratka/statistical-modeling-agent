# /restart Command Implementation

## Overview
Implemented a `/restart` command that performs a complete session reset when the bot gets stuck or the user wants to start fresh.

## Files Modified

### 1. src/core/state_manager.py
**Added:** `reset_session()` method (lines 836-900)

**Functionality:**
- Completely clears ALL session state (more thorough than `cancel_workflow()`)
- Clears workflow state (workflow_type, current_state, selections)
- Clears conversation history
- Clears uploaded data
- Clears state history (back button navigation)
- Clears local path workflow data (file_path, detected_schema, load_deferred, manual_schema)
- Clears password authentication state (dynamic_allowed_directories, pending_auth_path, password_attempts)
- Clears back button debouncing
- Clears prediction workflow data (compatible_models)
- **Does NOT** delete the session from memory (allows immediate restart)
- **Preserves** user_id and conversation_id

### 2. src/bot/main_handlers.py
**Added:** `restart_handler()` function (lines 859-906)

**Functionality:**
- Handles `/restart` command
- Calls `state_manager.reset_session()`
- Sends confirmation message to user
- Logs the reset action

### 3. src/bot/telegram_bot.py
**Modified:**
- Imported `restart_handler` (line 54)
- Registered command handler (line 233)

### 4. tests/unit/core/test_state_manager.py
**Added:** Three comprehensive tests (lines 408-516)

**Tests:**
1. `test_reset_session_clears_all_state` - Verifies all state is cleared
2. `test_reset_session_nonexistent` - Handles non-existent sessions gracefully
3. `test_reset_session_preserves_user_info` - Ensures user_id/conversation_id preserved

## Differences: /restart vs /cancel

| Feature | /cancel | /restart |
|---------|---------|----------|
| Clears workflow state | ✓ | ✓ |
| Clears selections | ✓ | ✓ |
| Clears state history | ✓ | ✓ |
| Clears uploaded data | ✗ | ✓ |
| Clears conversation history | ✗ | ✓ |
| Clears local path data | ✗ | ✓ |
| Clears password auth state | ✗ | ✓ |
| Clears prediction data | ✗ | ✓ |
| Clears model IDs | ✗ | ✓ |

## Usage

### User Experience
```
User: /restart

Bot: 🔄 **Session Reset Complete**

All session data has been cleared:
• Workflow state
• Uploaded data
• Conversation history
• Authentication state

Use /start to begin again.
```

### When to Use
- Bot is stuck in a workflow state
- User wants to start completely fresh
- After password entry workflow gets stuck
- Local file path workflow has errors
- Any workflow state corruption

## Testing

All tests pass:
```bash
pytest tests/unit/core/test_state_manager.py::TestStateManager::test_reset_session_clears_all_state -v
pytest tests/unit/core/test_state_manager.py::TestStateManager::test_reset_session_nonexistent -v
pytest tests/unit/core/test_state_manager.py::TestStateManager::test_reset_session_preserves_user_info -v
```

## Verification
```python
# Import check
from src.core.state_manager import StateManager
from src.bot.main_handlers import restart_handler

# Session exists after reset
state_manager.reset_session(user_id, conversation_id)
session = await state_manager.get_session(user_id, conversation_id)
assert session is not None
assert session.workflow_type is None  # All state cleared
```

## Security Considerations
- Does NOT persist password authentication state (session-scoped only)
- Password attempts counter is reset (rate limiting resets)
- Dynamic allowed directories are cleared (security whitelist expansion undone)
- Session signing/integrity maintained (if SESSION_SIGNING_KEY configured)

## Future Enhancements
- Add i18n support for confirmation message
- Add analytics/logging for reset frequency
- Consider adding a "confirm" step for destructive reset
- Add /restart to help menu

## Implementation Date
2024-12-13
