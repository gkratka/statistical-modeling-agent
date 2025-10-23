# Password-Protected File Path Access - Implementation Plan

**Date:** 2025-10-19
**Status:** Planning Phase
**Priority:** Medium (Security Enhancement)
**Complexity:** Medium-High (Security + State Management + Multi-Workflow)

---

## Executive Summary

This plan adds password protection to local file path access in both ML training and prediction workflows. Users can access paths in the whitelist freely, but paths outside the whitelist require password authentication ('senha123'). Upon successful authentication, the path is temporarily added to the allowed directories list for the session.

**Key Benefits:**
- Controlled access to sensitive file system areas
- Maintains existing security validation layers
- Seamless UX for whitelisted paths (no change)
- Simple password-based expansion of access
- Works across both /train and /predict workflows

**Security Approach:**
- Multi-layer validation (existing 8 checks + new password layer)
- In-memory whitelist expansion (no persistent config modification)
- Rate limiting to prevent brute force attacks
- Audit logging for all access attempts
- Session-scoped permissions (isolated per user)

---

## Current Architecture Analysis

### Existing Local Path Implementation

**File Path Training Workflow (Implemented):**
1. User starts `/train` → Data source selection
2. User selects "Local Path" → State: `AWAITING_FILE_PATH`
3. User provides path → `PathValidator.validate_path()` with 8 security layers
4. If valid and in whitelist → Load options (immediate/defer)
5. Continue with ML training workflow

**Security Validation Layers (path_validator.py:10-121):**
1. Auto-fix missing leading slash
2. Path normalization (resolve symlinks, relative paths)
3. Path traversal detection (`../`, encoded variants)
4. Directory whitelist enforcement (`is_path_in_allowed_directory()`)
5. File existence and type checks
6. Extension validation (`.csv`, `.xlsx`, `.parquet`)
7. File readability check
8. Size validation (max 1GB for local paths)

**Configuration (config/config.yaml:64-78):**
```yaml
local_data:
  enabled: true
  allowed_directories:
    - /Users/gkratka/Documents/datasets
    - /Users/gkratka/Documents/statistical-modeling-agent/data
    - ./data
    - ./tests/fixtures/test_datasets
    - /tmp
  max_file_size_mb: 1000
  allowed_extensions: [.csv, .xlsx, .xls, .parquet]
```

**State Flow (state_manager.py:275-350):**
```
CHOOSING_DATA_SOURCE → AWAITING_FILE_PATH → CHOOSING_LOAD_OPTION → ...
```

**Prediction Workflow (prediction_handlers.py:255-351):**
- Similar flow: CHOOSING_DATA_SOURCE → AWAITING_FILE_PATH → CHOOSING_LOAD_OPTION
- Uses same PathValidator and DataLoader
- Parallel implementation to training workflow

### Integration Points

**Where Password Check Fits:**
```
Current: Path provided → PathValidator.validate_path() → Whitelist check
New:     Path provided → PathValidator.validate_path() → Whitelist check
                                                           ↓ FAIL
                                                        Password prompt
                                                           ↓ SUCCESS
                                                        Add to session whitelist
                                                           ↓
                                                        Re-validate (pass)
```

**Affected Components:**
1. `PathValidator.validate_path()` - Add password check branch
2. `StateManager` - Add `AWAITING_PASSWORD` state + session whitelist
3. Handlers - Add password input handling for both workflows
4. Messages - Add password prompts and error messages
5. Config - Add password configuration (future: encrypted storage)

---

## Security Considerations

### Threat Model

**Attack Vectors:**
1. **Brute Force:** Automated password guessing
2. **Session Hijacking:** Steal expanded permissions
3. **Path Injection:** Bypass whitelist with malicious paths
4. **Replay Attacks:** Reuse intercepted password
5. **Denial of Service:** Flood with failed attempts

**Existing Mitigations:**
- Path traversal detection (Layer 3)
- Symlink resolution (Layer 2)
- Extension validation (Layer 6)
- Size limits (Layer 8)

**New Mitigations Needed:**
- Rate limiting (max 3 attempts per session)
- Exponential backoff (2s, 5s, 10s delays)
- Audit logging (all attempts, success/failure)
- Session isolation (permissions per user_id)
- Timeout (password attempt expires after 5 minutes)

### Password Security

**Current Plan (MVP):**
- Hardcoded password: `'senha123'`
- Plain text comparison
- No persistence

**Future Enhancements (Post-MVP):**
- Environment variable: `PASSWORD_FILE_PATH` in `.env`
- Hashed comparison: `bcrypt` or `argon2`
- Per-user passwords: Database-backed auth
- Two-factor: Require email/SMS confirmation
- Token-based: JWT with expiration

**Decision:** Start with hardcoded for simplicity. Add environment variable in Phase 5.

### Session Security

**Permission Scope:**
- Expanded whitelist stored in `UserSession.dynamic_allowed_directories`
- Isolated per `user_id + conversation_id` (existing session key)
- Cleared on workflow completion
- NOT persisted to disk (session save excludes dynamic whitelist)

**Concurrent Sessions:**
- Each user can have 1 session per conversation
- Dynamic whitelist does NOT propagate to other sessions
- Password required per session (no global unlock)

### Audit Requirements

**Log Events:**
1. Password prompt triggered (path, user_id)
2. Password attempt (success/failure, user_id, timestamp)
3. Rate limit exceeded (user_id, attempt count)
4. Dynamic whitelist expansion (path added, user_id)
5. Session cleanup (dynamic whitelist cleared)

**Log Format:**
```python
logger.info(f"[AUTH] Password prompt: user={user_id}, path={path}")
logger.warning(f"[AUTH] Failed attempt: user={user_id}, attempts={count}/3")
logger.info(f"[AUTH] Access granted: user={user_id}, path={parent_dir}")
```

**Retention:**
- Store in `data/logs/auth.log` (separate from agent.log)
- Rotate daily, keep 30 days
- Include timestamp, user_id, action, result

---

## Detailed Implementation Phases

### Phase 1: Password Validation Module

**File:** `src/utils/password_validator.py` (NEW - 120 lines)

**Purpose:** Centralized password validation with rate limiting and audit logging.

**Implementation:**

```python
"""Password validation for file path access control."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Audit logger for security events (separate from main logger)
auth_logger = logging.getLogger('auth')
auth_logger.setLevel(logging.INFO)
auth_handler = logging.FileHandler('data/logs/auth.log')
auth_handler.setFormatter(logging.Formatter(
    '%(asctime)s - [AUTH] - %(levelname)s - %(message)s'
))
auth_logger.addHandler(auth_handler)


@dataclass
class PasswordAttempt:
    """Track password attempts for rate limiting."""
    user_id: int
    attempt_count: int = 0
    last_attempt: Optional[float] = None
    locked_until: Optional[float] = None
    session_start: float = field(default_factory=time.time)


class PasswordValidator:
    """
    Password validation with rate limiting and audit logging.

    Security Features:
    - Configurable password (default: 'senha123', overridable via env)
    - Rate limiting: Max 3 attempts per session
    - Exponential backoff: 2s, 5s, 10s delays
    - Audit logging: All attempts logged
    - Session timeout: 5 minutes for password prompt
    """

    # Configuration
    DEFAULT_PASSWORD = "senha123"
    MAX_ATTEMPTS = 3
    BACKOFF_DELAYS = [2, 5, 10]  # seconds
    SESSION_TIMEOUT = 300  # 5 minutes

    def __init__(self, password: Optional[str] = None):
        """
        Initialize password validator.

        Args:
            password: Password to validate against (default: 'senha123')
        """
        self.password = password or self.DEFAULT_PASSWORD
        self._attempts: Dict[int, PasswordAttempt] = {}

    def validate_password(
        self,
        user_id: int,
        password_input: str,
        path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate password with rate limiting.

        Args:
            user_id: User ID attempting access
            password_input: Password provided by user
            path: File path being accessed (for logging)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get or create attempt record
        if user_id not in self._attempts:
            self._attempts[user_id] = PasswordAttempt(user_id=user_id)

        attempt = self._attempts[user_id]
        current_time = time.time()

        # Check if session expired
        if current_time - attempt.session_start > self.SESSION_TIMEOUT:
            auth_logger.warning(
                f"Session timeout: user={user_id}, elapsed={current_time - attempt.session_start:.1f}s"
            )
            self._attempts[user_id] = PasswordAttempt(user_id=user_id)
            return False, "Password prompt expired. Please try again with /train or /predict."

        # Check if locked due to rate limit
        if attempt.locked_until and current_time < attempt.locked_until:
            wait_seconds = int(attempt.locked_until - current_time)
            auth_logger.warning(
                f"Rate limit active: user={user_id}, wait={wait_seconds}s"
            )
            return False, f"Too many failed attempts. Please wait {wait_seconds} seconds."

        # Check backoff delay
        if attempt.last_attempt:
            delay_index = min(attempt.attempt_count - 1, len(self.BACKOFF_DELAYS) - 1)
            required_delay = self.BACKOFF_DELAYS[delay_index]
            elapsed = current_time - attempt.last_attempt

            if elapsed < required_delay:
                wait_seconds = int(required_delay - elapsed)
                return False, f"Please wait {wait_seconds} seconds before trying again."

        # Increment attempt counter
        attempt.attempt_count += 1
        attempt.last_attempt = current_time

        # Validate password
        is_valid = password_input == self.password

        if is_valid:
            auth_logger.info(
                f"Access granted: user={user_id}, path={path}, attempts={attempt.attempt_count}"
            )
            # Reset attempts on success
            del self._attempts[user_id]
            return True, None
        else:
            auth_logger.warning(
                f"Failed attempt: user={user_id}, path={path}, "
                f"attempts={attempt.attempt_count}/{self.MAX_ATTEMPTS}"
            )

            # Check if max attempts reached
            if attempt.attempt_count >= self.MAX_ATTEMPTS:
                attempt.locked_until = current_time + 60  # 60 second lockout
                auth_logger.error(
                    f"Rate limit exceeded: user={user_id}, locked_for=60s"
                )
                return False, (
                    f"Maximum attempts ({self.MAX_ATTEMPTS}) exceeded. "
                    f"Access locked for 60 seconds."
                )

            remaining = self.MAX_ATTEMPTS - attempt.attempt_count
            return False, (
                f"Incorrect password. {remaining} attempt(s) remaining."
            )

    def reset_attempts(self, user_id: int) -> None:
        """Reset attempt counter for user (e.g., on workflow restart)."""
        if user_id in self._attempts:
            del self._attempts[user_id]

    def get_attempt_count(self, user_id: int) -> int:
        """Get current attempt count for user."""
        return self._attempts.get(user_id, PasswordAttempt(user_id=user_id)).attempt_count
```

**Testing Requirements:**
- Test successful password validation
- Test failed password with correct retry flow
- Test rate limiting (3 attempts → lockout)
- Test exponential backoff delays
- Test session timeout (5 minutes)
- Test concurrent users (isolated attempts)

---

### Phase 2: StateManager Integration

**File:** `src/core/state_manager.py` (Modifications: ~100 lines)

**Changes:**

1. **Add Password State to Enums (Line 36-37, 69-70):**
```python
class MLTrainingState(Enum):
    # ... existing states ...
    AWAITING_FILE_PATH = "awaiting_file_path"
    AWAITING_PASSWORD = "awaiting_password"  # NEW: Password entry for non-whitelisted path
    CHOOSING_LOAD_OPTION = "choosing_load_option"
    # ... rest of states ...

class MLPredictionState(Enum):
    # ... existing states ...
    AWAITING_FILE_PATH = "awaiting_file_path"
    AWAITING_PASSWORD = "awaiting_password"  # NEW: Password entry for non-whitelisted path
    CHOOSING_LOAD_OPTION = "choosing_load_option"
    # ... rest of states ...
```

2. **Add Session Fields for Password Flow (Line 114-116):**
```python
@dataclass
class UserSession:
    # ... existing fields ...

    # NEW: Password-protected path access
    dynamic_allowed_directories: List[str] = field(default_factory=list)  # Session-scoped whitelist expansion
    pending_auth_path: Optional[str] = None  # Path waiting for password authentication
    password_attempts: int = 0  # Track attempts for rate limiting
```

3. **Update State Transitions (Line 275-350, 352-423):**
```python
ML_TRAINING_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # ... existing transitions ...

    # NEW: Password flow transitions
    MLTrainingState.AWAITING_FILE_PATH.value: {
        MLTrainingState.AWAITING_PASSWORD.value,  # NEW: Path not in whitelist
        MLTrainingState.CHOOSING_LOAD_OPTION.value  # Existing: Path in whitelist
    },
    MLTrainingState.AWAITING_PASSWORD.value: {
        MLTrainingState.CHOOSING_LOAD_OPTION.value,  # NEW: Password correct
        MLTrainingState.AWAITING_FILE_PATH.value      # NEW: Password incorrect, retry
    },
    # ... rest of transitions ...
}

ML_PREDICTION_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # ... existing transitions ...

    # NEW: Password flow transitions
    MLPredictionState.AWAITING_FILE_PATH.value: {
        MLPredictionState.AWAITING_PASSWORD.value,  # NEW: Path not in whitelist
        MLPredictionState.CHOOSING_LOAD_OPTION.value  # Existing: Path in whitelist
    },
    MLPredictionState.AWAITING_PASSWORD.value: {
        MLPredictionState.CHOOSING_LOAD_OPTION.value,  # NEW: Password correct
        MLPredictionState.AWAITING_FILE_PATH.value      # NEW: Password incorrect, retry
    },
    # ... rest of transitions ...
}
```

4. **Add Dynamic Whitelist Methods (Line 940+):**
```python
    def add_dynamic_directory(self, session: UserSession, directory: str) -> None:
        """
        Add directory to session-scoped whitelist.

        This allows temporary expansion of allowed directories after
        password authentication, without modifying the global config.

        Args:
            session: User session to modify
            directory: Directory path to add
        """
        if directory not in session.dynamic_allowed_directories:
            session.dynamic_allowed_directories.append(directory)
            auth_logger.info(
                f"Dynamic whitelist expanded: user={session.user_id}, "
                f"dir={directory}, session={session.session_key}"
            )

    def get_effective_allowed_directories(
        self,
        session: UserSession,
        base_allowed: List[str]
    ) -> List[str]:
        """
        Get combined whitelist (base + dynamic).

        Args:
            session: User session
            base_allowed: Base allowed directories from config

        Returns:
            Combined list of allowed directories
        """
        return list(set(base_allowed + session.dynamic_allowed_directories))
```

5. **Exclude Dynamic Whitelist from Persistence (Line 822-843):**
```python
    def _session_to_dict(self, session: UserSession) -> Dict[str, Any]:
        """Convert session to JSON-serializable dict."""
        data = {
            # ... existing fields ...
            # NOTE: dynamic_allowed_directories is NOT persisted (security)
            # NOTE: pending_auth_path is NOT persisted (session state)
            # NOTE: password_attempts is NOT persisted (rate limiting reset on reload)
        }
        return data
```

**Testing Requirements:**
- Test state transitions (file_path → password → load_option)
- Test dynamic whitelist isolation (separate sessions)
- Test session cleanup (dynamic whitelist cleared)
- Test persistence exclusion (reload doesn't restore dynamic dirs)

---

### Phase 3: Handler Modifications (Training)

**File:** `src/bot/ml_handlers/ml_training_local_path.py` (Modifications: ~150 lines)

**Changes:**

1. **Import PasswordValidator (Line 26):**
```python
from src.utils.password_validator import PasswordValidator
```

2. **Initialize PasswordValidator in Handler (Line 78):**
```python
class LocalPathMLTrainingHandler:
    def __init__(self, ...):
        # ... existing initialization ...

        # NEW: Initialize password validator
        self.password_validator = PasswordValidator()
```

3. **Modify Path Processing to Check Whitelist (Line 359-458, replace):**
```python
    async def _process_file_path_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        file_path: str
    ) -> None:
        """Process file path input from user. Validates path and checks whitelist."""
        print(f"🔍 DEBUG: Processing file path: {file_path}")

        validating_msg = await update.message.reply_text(
            "🔍 **Validating path...**"
        )

        try:
            # Validate path structure (security checks 1-8, EXCEPT whitelist check)
            from src.utils.path_validator import validate_local_path, get_file_size_mb
            from pathlib import Path

            is_valid, error_msg, resolved_path = validate_local_path(
                path=file_path,
                allowed_dirs=self.data_loader.allowed_directories,
                max_size_mb=self.data_loader.local_max_size_mb,
                allowed_extensions=self.data_loader.local_extensions
            )

            if not is_valid:
                # Check if error is specifically whitelist failure
                if "not in allowed directories" in error_msg.lower():
                    # Whitelist check failed - prompt for password
                    await validating_msg.delete()
                    await self._prompt_for_password(
                        update, context, session, file_path, resolved_path
                    )
                    raise ApplicationHandlerStop
                else:
                    # Other validation error (path traversal, size, etc.)
                    await validating_msg.delete()
                    error_display = LocalPathMessages.format_path_error(
                        error_type="security_validation",
                        path=file_path,
                        error_details=error_msg
                    )
                    await update.message.reply_text(error_display)
                    return

            # Path valid and in whitelist - continue to load options
            session.file_path = str(resolved_path)
            size_mb = get_file_size_mb(resolved_path)

            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )

            await validating_msg.delete()

            # Show load option selection
            keyboard = [
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(str(resolved_path), size_mb),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            raise ApplicationHandlerStop

        except ApplicationHandlerStop:
            raise
        except Exception as e:
            print(f"❌ DEBUG: Unexpected error validating path: {e}")
            import traceback
            traceback.print_exc()

            try:
                await validating_msg.delete()
            except Exception:
                pass

            await update.message.reply_text(
                LocalPathMessages.format_path_error(
                    error_type="unexpected",
                    path=file_path,
                    error_details=str(e)
                )
            )
```

4. **Add Password Prompt Method (NEW - Line 460+):**
```python
    async def _prompt_for_password(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        session,
        original_path: str,
        resolved_path: Path
    ) -> None:
        """
        Prompt user for password to access non-whitelisted path.

        Args:
            update: Telegram update
            context: Telegram context
            session: User session
            original_path: Original path string from user
            resolved_path: Resolved absolute path
        """
        # Store pending path for later validation
        session.pending_auth_path = str(resolved_path)

        # Transition to password state
        session.save_state_snapshot()
        success, error_msg, missing = await self.state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_PASSWORD.value
        )

        if not success:
            await update.message.reply_text(
                f"❌ **State Transition Failed**\n\n{error_msg}",
                parse_mode="Markdown"
            )
            return

        # Get parent directory for display
        parent_dir = str(resolved_path.parent)

        # Show password prompt
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="password:cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"🔐 **Password Required**\n\n"
            f"The path you entered is not in the allowed directories:\n"
            f"`{original_path}`\n\n"
            f"**Resolved to:** `{parent_dir}`\n\n"
            f"To access this directory, please enter the password.\n\n"
            f"⚠️ **Security Note:** This will grant access to ALL files in "
            f"this directory for this session.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
```

5. **Add Password Input Handler (NEW - Line 510+):**
```python
    async def handle_password_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle password input for path access."""
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            password_input = update.message.text.strip()
        except AttributeError as e:
            logger.error(f"Malformed update in handle_password_input: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Validate we're in password state
        if session.current_state != MLTrainingState.AWAITING_PASSWORD.value:
            return

        # Get pending path
        pending_path = session.pending_auth_path
        if not pending_path:
            await update.message.reply_text(
                "❌ **Session Error**\n\nNo pending path found. Please try /train again.",
                parse_mode="Markdown"
            )
            return

        # Validate password
        is_valid, error_msg = self.password_validator.validate_password(
            user_id=user_id,
            password_input=password_input,
            path=pending_path
        )

        if is_valid:
            # Password correct - add directory to dynamic whitelist
            from pathlib import Path
            parent_dir = str(Path(pending_path).parent)
            self.state_manager.add_dynamic_directory(session, parent_dir)

            # Clear pending auth
            session.pending_auth_path = None
            session.password_attempts = 0

            # Store file path and get size
            session.file_path = pending_path
            from src.utils.path_validator import get_file_size_mb
            size_mb = get_file_size_mb(Path(pending_path))

            # Transition to load options
            session.save_state_snapshot()
            await self.state_manager.transition_state(
                session,
                MLTrainingState.CHOOSING_LOAD_OPTION.value
            )

            await update.message.reply_text(
                f"✅ **Access Granted**\n\n"
                f"Directory added to allowed paths for this session:\n"
                f"`{parent_dir}`",
                parse_mode="Markdown"
            )

            # Show load options
            keyboard = [
                [InlineKeyboardButton("🔄 Load Now", callback_data="load_option:immediate")],
                [InlineKeyboardButton("⏳ Defer Loading", callback_data="load_option:defer")]
            ]
            add_back_button(keyboard)
            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                LocalPathMessages.load_option_prompt(pending_path, size_mb),
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            raise ApplicationHandlerStop

        else:
            # Password incorrect or rate limited
            session.password_attempts += 1

            if "locked" in error_msg.lower() or "maximum attempts" in error_msg.lower():
                # Rate limit exceeded - reset to file path input
                session.pending_auth_path = None
                session.password_attempts = 0

                await self.state_manager.transition_state(
                    session,
                    MLTrainingState.AWAITING_FILE_PATH.value
                )

                await update.message.reply_text(
                    f"❌ {error_msg}\n\n"
                    f"Please try again or choose a different path.",
                    parse_mode="Markdown"
                )
            else:
                # Failed attempt, allow retry
                await update.message.reply_text(
                    f"❌ {error_msg}",
                    parse_mode="Markdown"
                )
```

6. **Add Password Cancel Handler (NEW - Line 610+):**
```python
    async def handle_password_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle password cancel callback."""
        query = update.callback_query
        await query.answer()

        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update in handle_password_cancel: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Clear pending auth
        session.pending_auth_path = None
        session.password_attempts = 0

        # Reset password validator attempts
        self.password_validator.reset_attempts(user_id)

        # Go back to file path input
        session.save_state_snapshot()
        await self.state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_FILE_PATH.value
        )

        allowed_dirs = self.data_loader.allowed_directories
        await query.edit_message_text(
            LocalPathMessages.file_path_input_prompt(allowed_dirs),
            parse_mode="Markdown"
        )
```

7. **Update Text Router to Handle Password State (Line 324-340, modify):**
```python
    async def handle_text_input(self, ...):
        # ... existing code ...

        # Route based on current state
        if current_state == MLTrainingState.AWAITING_FILE_PATH.value:
            await self._process_file_path_input(update, context, session, text_input)
        elif current_state == MLTrainingState.AWAITING_PASSWORD.value:  # NEW
            await self.handle_password_input(update, context)
        elif current_state == MLTrainingState.AWAITING_SCHEMA_INPUT.value:
            await self._process_schema_input(update, context, session, text_input)
        # ... rest of routing ...
```

8. **Register Password Handlers (Line 790+, add):**
```python
def register_local_path_training_handlers(application, ...):
    # ... existing handlers ...

    # NEW: Password callback handler
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_password_cancel,
            pattern=r"^password:cancel$"
        )
    )
```

**Testing Requirements:**
- Test whitelist path (no password prompt)
- Test non-whitelist path (password prompt shown)
- Test correct password (access granted, dynamic whitelist expanded)
- Test incorrect password (retry flow, rate limiting)
- Test password cancel (back to file path input)

---

### Phase 4: Handler Modifications (Prediction)

**File:** `src/bot/ml_handlers/prediction_handlers.py` (Modifications: ~150 lines)

**Changes:** Nearly identical to Phase 3, adapted for prediction workflow.

1. **Import PasswordValidator (Line 32):**
```python
from src.utils.password_validator import PasswordValidator
```

2. **Initialize PasswordValidator (Line 79):**
```python
class PredictionHandler:
    def __init__(self, ...):
        # ... existing initialization ...

        # NEW: Initialize password validator
        self.password_validator = PasswordValidator()
```

3. **Modify `handle_file_path_input()` (Line 255-351, replace with whitelist check):**
```python
    async def handle_file_path_input(self, ...):
        # ... validation code ...

        if not result['is_valid']:
            # Check if whitelist error
            if "not in allowed directories" in result['error'].lower():
                await safe_delete_message(validating_msg)
                await self._prompt_for_password(
                    update, context, session, file_path, result['resolved_path']
                )
                raise ApplicationHandlerStop
            else:
                # Other validation error
                # ... existing error handling ...

        # ... rest of existing code ...
```

4. **Add Password Prompt, Input, Cancel Handlers (NEW - 3 methods, ~160 lines):**
   - Copy implementations from Phase 3, adapt state names:
     - `MLTrainingState.AWAITING_PASSWORD` → `MLPredictionState.AWAITING_PASSWORD`
     - `MLTrainingState.CHOOSING_LOAD_OPTION` → `MLPredictionState.CHOOSING_LOAD_OPTION`
     - `MLTrainingState.AWAITING_FILE_PATH` → `MLPredictionState.AWAITING_FILE_PATH`

5. **Update Text Router (Line 1766-1780, modify):**
```python
    async def handle_text_input(self, ...):
        # ... existing code ...

        # Route based on current state
        if current_state == MLPredictionState.AWAITING_FILE_PATH.value:
            await self.handle_file_path_input(update, context)
        elif current_state == MLPredictionState.AWAITING_PASSWORD.value:  # NEW
            await self.handle_password_input(update, context)
        elif current_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value:
            await self.handle_feature_selection_input(update, context)
        # ... rest of routing ...
```

6. **Register Password Handler (Line 1869+, add):**
```python
def register_prediction_handlers(application, ...):
    # ... existing handlers ...

    # NEW: Password callback handler
    application.add_handler(
        CallbackQueryHandler(
            handler.handle_password_cancel,
            pattern=r"^pred_password:cancel$"  # Note: different prefix for prediction
        )
    )
```

**Testing Requirements:**
- Same as Phase 3, but for prediction workflow
- Test workflow isolation (training vs prediction passwords don't interfere)

---

### Phase 5: Configuration & Environment Variables

**File:** `config/config.yaml` (Modifications: ~10 lines)

**Add Password Configuration (Line 78+):**
```yaml
# Local File Path Training (Existing + NEW password config)
local_data:
  enabled: true
  allowed_directories:
    - /Users/gkratka/Documents/datasets
    - /Users/gkratka/Documents/statistical-modeling-agent/data
    - ./data
    - ./tests/fixtures/test_datasets
    - /tmp
  max_file_size_mb: 1000
  allowed_extensions: [.csv, .xlsx, .xls, .parquet]

  # NEW: Password-based access control
  password_protection:
    enabled: true                    # Feature flag: enable/disable password prompts
    password: ${FILE_PATH_PASSWORD}  # Environment variable for password (default: 'senha123')
    max_attempts: 3                  # Maximum password attempts per session
    lockout_duration_seconds: 60     # Lockout duration after max attempts
    session_timeout_seconds: 300     # Password prompt expires after 5 minutes
```

**File:** `.env.example` (Add line)

```bash
# File Path Access Password (optional, defaults to 'senha123' if not set)
FILE_PATH_PASSWORD=senha123
```

**File:** `src/utils/password_validator.py` (Modify Line 47-48)

```python
    def __init__(self, password: Optional[str] = None):
        """
        Initialize password validator.

        Args:
            password: Password to validate against (default: from env or 'senha123')
        """
        # Priority: 1) Explicit param, 2) Environment variable, 3) Default
        if password is None:
            password = os.getenv('FILE_PATH_PASSWORD', self.DEFAULT_PASSWORD)

        self.password = password
        self._attempts: Dict[int, PasswordAttempt] = {}
```

**Testing Requirements:**
- Test default password ('senha123')
- Test environment variable override
- Test config flag (enabled/disabled)
- Test config values (max_attempts, lockout, timeout)

---

### Phase 6: Message Templates

**File:** `src/bot/messages/local_path_messages.py` (Add ~50 lines)

**Add Password Messages (Line 200+):**
```python
    @staticmethod
    def password_prompt(original_path: str, resolved_dir: str) -> str:
        """
        Password prompt for non-whitelisted path.

        Args:
            original_path: Original path string from user
            resolved_dir: Resolved parent directory

        Returns:
            Markdown-formatted password prompt
        """
        return (
            f"🔐 **Password Required**\n\n"
            f"The path you entered is not in the allowed directories:\n"
            f"`{original_path}`\n\n"
            f"**Resolved to:** `{resolved_dir}`\n\n"
            f"To access this directory, please enter the password.\n\n"
            f"⚠️ **Security Note:** This will grant access to ALL files in "
            f"this directory for the current session only."
        )

    @staticmethod
    def password_success(directory: str) -> str:
        """Message shown after successful password entry."""
        return (
            f"✅ **Access Granted**\n\n"
            f"Directory added to allowed paths for this session:\n"
            f"`{directory}`\n\n"
            f"You can now access any file in this directory."
        )

    @staticmethod
    def password_failure(error_message: str) -> str:
        """Message shown after failed password attempt."""
        return f"❌ {error_message}"

    @staticmethod
    def password_lockout(wait_seconds: int) -> str:
        """Message shown when user is locked out."""
        return (
            f"🔒 **Access Locked**\n\n"
            f"Too many failed password attempts.\n\n"
            f"Please wait {wait_seconds} seconds before trying again, "
            f"or choose a different path from the whitelist."
        )

    @staticmethod
    def password_timeout() -> str:
        """Message shown when password prompt expires."""
        return (
            f"⏱️ **Session Timeout**\n\n"
            f"The password prompt has expired (5 minute limit).\n\n"
            f"Please restart with /train or /predict to try again."
        )
```

**Testing Requirements:**
- Test message rendering (no markdown errors)
- Test special character escaping (paths with underscores, etc.)

---

### Phase 7: Testing Strategy

**Test Coverage Requirements:**

1. **Unit Tests (120 tests total):**

   a. **PasswordValidator Tests (src/utils/test_password_validator.py - 40 tests):**
   - Test correct password validation
   - Test incorrect password validation
   - Test rate limiting (3 attempts → lockout)
   - Test exponential backoff (2s, 5s, 10s)
   - Test session timeout (5 minutes)
   - Test concurrent users (isolated attempts)
   - Test attempt reset on success
   - Test attempt reset on workflow restart
   - Test audit logging (success, failure, lockout)
   - Test environment variable password loading

   b. **StateManager Tests (tests/unit/test_state_manager.py - 30 tests):**
   - Test dynamic whitelist addition
   - Test effective allowed directories (base + dynamic)
   - Test session isolation (separate dynamic whitelists)
   - Test state transitions (file_path → password → load_option)
   - Test password state prerequisites
   - Test session persistence (dynamic whitelist NOT saved)
   - Test session cleanup (dynamic whitelist cleared)
   - Test concurrent sessions (no cross-contamination)

   c. **Handler Tests (tests/unit/test_password_handlers.py - 50 tests):**
   - **Training Workflow:**
     - Test whitelisted path (no password prompt)
     - Test non-whitelisted path (password prompt shown)
     - Test correct password (access granted)
     - Test incorrect password (retry flow)
     - Test password cancel (back to file path)
     - Test rate limit (lockout after 3 failures)
   - **Prediction Workflow:**
     - Same tests as training workflow
   - **Workflow Isolation:**
     - Test training password doesn't affect prediction
     - Test prediction password doesn't affect training

2. **Integration Tests (30 tests total):**

   a. **End-to-End Workflow Tests (tests/integration/test_password_workflow.py - 20 tests):**
   - **Training E2E:**
     - Test full flow: /train → local path → password → load → train
     - Test whitelisted path bypass
     - Test password failure recovery
     - Test dynamic whitelist reuse (second file in same dir)
   - **Prediction E2E:**
     - Test full flow: /predict → local path → password → load → predict
     - Test template loading with password-protected paths
   - **Cross-Workflow:**
     - Test switching between training and prediction
     - Test session isolation across conversations

   b. **Security Tests (tests/integration/test_password_security.py - 10 tests):**
   - Test path traversal blocked (even with password)
   - Test symlink resolution (password required for actual location)
   - Test concurrent users (rate limiting isolated)
   - Test session expiry (dynamic whitelist cleared)
   - Test audit log completeness

3. **Manual Testing Checklist:**
   - [ ] Test with real file system paths
   - [ ] Test with network paths (if applicable)
   - [ ] Test with very long paths (>256 chars)
   - [ ] Test with special characters in paths
   - [ ] Test with multiple users simultaneously
   - [ ] Test telegram message formatting (markdown)
   - [ ] Test error recovery (bot restart mid-workflow)
   - [ ] Verify audit logs are written correctly
   - [ ] Test password change (env variable update)
   - [ ] Test config flag (disable password feature)

**Test Execution Plan:**
1. Phase 1-2: Unit tests for PasswordValidator + StateManager
2. Phase 3-4: Unit tests for handlers
3. Phase 5-6: Integration tests (config + messages)
4. Phase 7: Security tests + manual testing
5. Regression: Run full test suite (127 existing tests + 150 new = 277 total)

---

## Risk Analysis & Mitigation

### Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Backwards Compatibility:** Existing users lose access to paths | Medium | High | Add config flag `password_protection.enabled`, default to `false` for smooth migration |
| **State Transition Bugs:** Password state breaks existing workflows | Medium | High | Comprehensive state machine tests, manual testing across all workflows |
| **Security Bypass:** Path traversal still possible with password | Low | Critical | Password check happens AFTER all security checks (Layer 9, not Layer 4) |
| **Rate Limit DOS:** Users locked out unfairly | Low | Medium | Exponential backoff + manual unlock mechanism (admin command) |
| **Password Leakage:** Password visible in logs/messages | Low | High | Never log password input, only log success/failure |
| **Session Hijacking:** Steal expanded permissions | Low | Medium | Session-scoped whitelist, cleared on workflow end |
| **Concurrent Access:** Race conditions in whitelist expansion | Low | Medium | Use existing session lock (`_global_lock` in StateManager) |

### Deployment Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Config Migration:** Existing deployments break | Low | High | Provide migration script, backward-compatible defaults |
| **Environment Variables:** Password not set in .env | Medium | Low | Default to 'senha123' with warning in logs |
| **Audit Log Storage:** Disk fills up with logs | Low | Medium | Log rotation (30 days), size limits (100MB) |
| **Performance:** Password validation slows down workflow | Low | Low | In-memory validation, no I/O, <10ms overhead |

### Rollback Plan

**If Critical Bug Found:**
1. Set `password_protection.enabled: false` in config
2. Restart bot (workflows continue with whitelist-only validation)
3. Fix bug, redeploy, re-enable feature

**Data Safety:**
- No persistent state changes (dynamic whitelist in-memory only)
- Audit logs can be archived/deleted without affecting functionality
- Password change doesn't invalidate existing sessions

---

## Performance Considerations

**Expected Overhead:**
- Password validation: <5ms (in-memory hash comparison)
- Whitelist lookup: <1ms (list iteration, <100 directories)
- State transition: <2ms (existing overhead)
- Audit logging: <10ms (async file write)

**Total Additional Latency:** <20ms per path validation (negligible)

**Memory Usage:**
- PasswordAttempt object: ~100 bytes per user
- Dynamic whitelist: ~50 bytes per directory
- Expected: <10KB per active user

**Scalability:**
- Concurrent users: No contention (session-scoped)
- Rate limiting: O(1) lookup per user
- Audit logging: Batched writes (10s intervals)

---

## Future Enhancements (Post-MVP)

1. **Per-User Passwords:** Database-backed authentication
2. **Two-Factor Auth:** Email/SMS confirmation for sensitive paths
3. **Temporary Access:** Time-limited permissions (e.g., 1 hour)
4. **Path Patterns:** Whitelist with wildcards (`/data/*/public`)
5. **Admin Dashboard:** View/revoke dynamic permissions
6. **Encrypted Audit Logs:** PGP-signed log entries
7. **OAuth Integration:** Use existing SSO providers
8. **Path Quotas:** Limit disk space usage per user
9. **Access History:** Show user what paths they've accessed
10. **Collaborative Access:** Share permissions between team members

---

## Implementation Timeline

| Phase | Component | Lines of Code | Est. Time | Dependencies |
|-------|-----------|---------------|-----------|--------------|
| **Phase 1** | PasswordValidator | 120 | 4 hours | None |
| **Phase 2** | StateManager Integration | 100 | 3 hours | Phase 1 |
| **Phase 3** | Training Handler | 150 | 5 hours | Phase 1-2 |
| **Phase 4** | Prediction Handler | 150 | 4 hours | Phase 1-2 |
| **Phase 5** | Config & Environment | 20 | 1 hour | Phase 1-4 |
| **Phase 6** | Message Templates | 50 | 1 hour | Phase 1-4 |
| **Phase 7** | Testing | 600 (test code) | 8 hours | Phase 1-6 |
| **Total** | | 1,190 lines | 26 hours | |

**Recommended Approach:** Implement phases 1-2 first (foundation), then 3-4 in parallel (workflows), then 5-6 (polish), then 7 (validation).

---

## Success Criteria

**Functional Requirements:**
- [ ] User can access whitelisted paths without password (no UX change)
- [ ] User is prompted for password on non-whitelisted path
- [ ] Correct password grants access to entire directory
- [ ] Incorrect password allows retry (up to 3 attempts)
- [ ] Rate limiting prevents brute force attacks
- [ ] Dynamic whitelist works in both /train and /predict workflows
- [ ] Session cleanup removes dynamic permissions
- [ ] Workflow isolation (separate sessions don't share permissions)

**Non-Functional Requirements:**
- [ ] All 150 new tests pass (100% coverage for new code)
- [ ] No regression in existing 127 tests
- [ ] Performance overhead <20ms per path validation
- [ ] Audit logs written for all access attempts
- [ ] Documentation updated (README, CLAUDE.md)
- [ ] Security review completed (no vulnerabilities)

**User Experience:**
- [ ] Error messages are clear and actionable
- [ ] Password prompt is obvious and well-explained
- [ ] Rate limiting messages explain wait time
- [ ] Workflow feels seamless for valid passwords
- [ ] No confusion between workflows (training vs prediction)

---

## Open Questions & Decisions Needed

1. **Password Storage:**
   - **Q:** Should password be stored in environment variable or database?
   - **A:** Start with environment variable (simpler), migrate to database if needed.

2. **Whitelist Scope:**
   - **Q:** Should dynamic whitelist apply to entire directory or just the specific file?
   - **A:** Entire directory (more usable, still secure with password gate).

3. **Permission Persistence:**
   - **Q:** Should dynamic whitelist persist across sessions (saved to disk)?
   - **A:** No (security: require password each session).

4. **Admin Override:**
   - **Q:** Should admins be able to bypass password or view audit logs?
   - **A:** Future enhancement (not in MVP).

5. **Password Complexity:**
   - **Q:** Should we enforce password strength requirements?
   - **A:** No (admin-configurable password, not user-generated).

6. **Multiple Passwords:**
   - **Q:** Should different directories have different passwords?
   - **A:** Future enhancement (single password for MVP).

7. **Expiration:**
   - **Q:** Should dynamic permissions expire after time period (e.g., 1 hour)?
   - **A:** No (session-scoped is sufficient, expires on workflow end).

8. **Notification:**
   - **Q:** Should admin be notified of successful password entries?
   - **A:** Yes (audit log includes all events, optional webhook in future).

---

## Appendix: File Modification Summary

| File | Type | Lines Added | Lines Modified | Lines Deleted | Total Changes |
|------|------|-------------|----------------|---------------|---------------|
| `src/utils/password_validator.py` | NEW | 120 | 0 | 0 | 120 |
| `src/core/state_manager.py` | MOD | 50 | 30 | 0 | 80 |
| `src/bot/ml_handlers/ml_training_local_path.py` | MOD | 200 | 50 | 10 | 240 |
| `src/bot/ml_handlers/prediction_handlers.py` | MOD | 200 | 50 | 10 | 240 |
| `src/bot/messages/local_path_messages.py` | MOD | 50 | 0 | 0 | 50 |
| `config/config.yaml` | MOD | 10 | 0 | 0 | 10 |
| `.env.example` | MOD | 2 | 0 | 0 | 2 |
| `tests/unit/test_password_validator.py` | NEW | 200 | 0 | 0 | 200 |
| `tests/unit/test_password_handlers.py` | NEW | 250 | 0 | 0 | 250 |
| `tests/integration/test_password_workflow.py` | NEW | 150 | 0 | 0 | 150 |
| **TOTAL** | | **1,232** | **130** | **20** | **1,382** |

---

## Appendix: Security Validation Flow Diagram

```
User provides path
       ↓
[Layer 1-2] Path normalization & symlink resolution
       ↓
[Layer 3] Path traversal detection
       ↓ FAIL → Reject (no password prompt)
       ↓ PASS
[Layer 4] Whitelist check (base + dynamic)
       ↓ PASS → Continue to load options
       ↓ FAIL
[Layer 9] NEW: Password prompt
       ↓ Cancel → Back to file input
       ↓ Incorrect (3x) → Rate limit lockout
       ↓ Correct
Add parent dir to dynamic whitelist
       ↓
[Layer 4 Retry] Whitelist check (now passes)
       ↓
[Layer 5-8] Extension, size, readability checks
       ↓ FAIL → Reject
       ↓ PASS
Continue to load options
```

**Key Insight:** Password check is Layer 9 (AFTER security checks 1-3), ensuring malicious paths are rejected before password prompt.

---

## Appendix: State Transition Diagrams

**Training Workflow (with Password):**
```
/train
  ↓
CHOOSING_DATA_SOURCE
  ↓ (local_path)
AWAITING_FILE_PATH
  ↓ (path provided)
  ├─ In whitelist → CHOOSING_LOAD_OPTION
  └─ Not in whitelist → AWAITING_PASSWORD
                           ↓ (correct password)
                         CHOOSING_LOAD_OPTION
                           ↓ (incorrect password)
                         AWAITING_PASSWORD (retry)
                           ↓ (3x failures)
                         AWAITING_FILE_PATH (reset)
```

**Prediction Workflow (with Password):**
```
/predict
  ↓
STARTED
  ↓
CHOOSING_DATA_SOURCE
  ↓ (local_path)
AWAITING_FILE_PATH
  ↓ (path provided)
  ├─ In whitelist → CHOOSING_LOAD_OPTION
  └─ Not in whitelist → AWAITING_PASSWORD
                           ↓ (correct password)
                         CHOOSING_LOAD_OPTION
                           ↓ (incorrect password)
                         AWAITING_PASSWORD (retry)
                           ↓ (3x failures)
                         AWAITING_FILE_PATH (reset)
```

---

**End of Implementation Plan**

This comprehensive plan provides a complete roadmap for adding password-protected file path access to the ML training and prediction workflows. The phased approach allows for incremental development and testing, with clear rollback options if issues arise. The security-first design ensures that the password layer adds controlled flexibility without compromising the existing 8-layer validation system.
