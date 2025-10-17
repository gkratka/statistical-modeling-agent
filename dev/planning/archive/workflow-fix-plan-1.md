# Workflow Fix Plan 1 - Comprehensive TDD-Driven Repair

**Created**: 2025-10-10
**Author**: Claude Code
**Status**: Planning Phase
**Estimated Timeline**: 7-8 hours over 2 days

---

## Executive Summary

### Critical Issues Identified

1. **🔴 CRITICAL - Multiple Bot Instances**
   - **Impact**: Blocks ALL workflows
   - **Evidence**: 2 active processes + 8 zombie background processes
   - **Symptom**: "Conflict: terminated by other getUpdates request" errors
   - **Root Cause**: Incomplete cleanup, no PID file locking mechanism
   - **Fix Priority**: IMMEDIATE (Phase 1)

2. **🔴 HIGH - "Load Now" Workflow Failure**
   - **Impact**: Local path training workflow unusable
   - **Evidence**: Screenshots showing "Unexpected Error" at load_now button press
   - **Symptom**: Generic error message, actual exception not captured
   - **Root Cause**: Unknown - defensive error handling masks real error
   - **Fix Priority**: After Phase 1 (Phase 2)

3. **🟡 MEDIUM - Session Persistence Issues**
   - **Impact**: Users lose workflow progress, poor UX
   - **Evidence**: User report "workflow continues to session persist, not working as it should"
   - **Symptom**: State not resuming correctly across interactions
   - **Root Cause**: StateManager not properly integrated or missing save/load calls
   - **Fix Priority**: After Phase 2 (Phase 3)

4. **🟢 LOW - Missing Planned Architecture**
   - **Impact**: Code maintainability, future scalability
   - **Evidence**: integration-layer.md defines IntegrationLayer, ResponseBuilder, CommandHandlers not implemented
   - **Gap**: 800+ LOC of planned but unimplemented code
   - **Fix Priority**: DEFER to Phase 5 (optional)

### Success Metrics

✅ **Primary Goal**: All workflows work end-to-end without errors
✅ **Phase 1 Success**: Single bot instance, zero conflicts for 24 hours
✅ **Phase 2 Success**: Load Now completes successfully with real data
✅ **Phase 3 Success**: Session persists across bot restarts
✅ **Phase 4 Success**: All E2E tests pass (100% coverage of user workflows)

---

## Phase 1: Fix Bot Instance Conflicts (CRITICAL)

**Priority**: 🔴 CRITICAL - Blocks everything
**Timeline**: 30 minutes
**Files Modified**:
- `scripts/start_bot_clean.sh`
- `scripts/check_bot_health.sh`
- `src/bot/telegram_bot.py`

### 1.1 TDD: Write Failing Tests First

**Test File**: `tests/integration/test_bot_instance_management.py`

```python
import pytest
import subprocess
import time
import os
from pathlib import Path

class TestBotInstanceManagement:
    """TDD tests for bot process lifecycle management."""

    def test_pid_file_created_on_startup(self):
        """Test 1: Bot creates PID file at startup."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        if pid_file.exists():
            pid_file.unlink()

        # ACT
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ASSERT
        assert pid_file.exists(), "PID file not created"
        pid = int(pid_file.read_text())
        assert pid == proc.pid, "PID file contains wrong PID"

        # CLEANUP
        proc.terminate()
        proc.wait()

    def test_second_instance_refuses_to_start(self):
        """Test 2: Second bot instance detects existing PID and exits."""
        # ARRANGE
        proc1 = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ACT
        proc2 = subprocess.Popen(
            ["python3", "src/bot/telegram_bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc2.communicate(timeout=5)

        # ASSERT
        assert proc2.returncode != 0, "Second instance should fail"
        assert b"already running" in stderr.lower(), "Wrong error message"

        # CLEANUP
        proc1.terminate()
        proc1.wait()

    def test_pid_file_cleanup_on_graceful_shutdown(self):
        """Test 3: PID file removed on clean shutdown."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)
        assert pid_file.exists()

        # ACT
        proc.terminate()
        proc.wait(timeout=10)

        # ASSERT
        assert not pid_file.exists(), "PID file not cleaned up"

    def test_stale_pid_detection_and_override(self):
        """Test 4: Stale PID (dead process) allows new instance."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        pid_file.write_text("99999")  # Non-existent PID

        # ACT
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ASSERT
        assert proc.poll() is None, "Bot should start with stale PID"
        new_pid = int(pid_file.read_text())
        assert new_pid == proc.pid, "PID file not updated"

        # CLEANUP
        proc.terminate()
        proc.wait()

    def test_clean_startup_script_kills_zombies(self):
        """Test 5: start_bot_clean.sh cleans up zombie processes."""
        # ARRANGE - Create 3 zombie background processes
        for i in range(3):
            subprocess.Popen(
                ["python3", "src/bot/telegram_bot.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        time.sleep(2)

        # ACT
        result = subprocess.run(
            ["./scripts/start_bot_clean.sh"],
            capture_output=True,
            text=True
        )

        # ASSERT
        assert result.returncode == 0, "Startup script failed"
        assert "TEST 2 PASSED" in result.stdout, "Cleanup verification failed"

        # Verify only 1 process running
        ps_result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        bot_processes = [
            line for line in ps_result.stdout.split("\n")
            if "telegram_bot" in line and "grep" not in line
        ]
        assert len(bot_processes) == 1, f"Expected 1 process, found {len(bot_processes)}"
```

**Expected Result**: All 5 tests FAIL (not implemented yet)

### 1.2 Implementation: PID File Locking

**File**: `src/bot/telegram_bot.py`

Add at the top of `main()` function:

```python
import os
import sys
import signal
from pathlib import Path

# PID file management
PID_FILE = Path(".bot.pid")

def cleanup_pid_file():
    """Remove PID file on shutdown."""
    if PID_FILE.exists():
        PID_FILE.unlink()
        logger.info("PID file removed")

def check_running_instance() -> bool:
    """Check if another instance is running."""
    if not PID_FILE.exists():
        return False

    try:
        existing_pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        return False

    # Check if process is actually running
    try:
        os.kill(existing_pid, 0)  # Signal 0 checks existence
        return True  # Process exists
    except OSError:
        # Process doesn't exist (stale PID)
        logger.warning(f"Stale PID file detected (PID {existing_pid} dead)")
        PID_FILE.unlink()
        return False

def create_pid_file():
    """Create PID file with current process ID."""
    PID_FILE.write_text(str(os.getpid()))
    logger.info(f"PID file created: {os.getpid()}")

async def main():
    """Bot entry point with instance management."""

    # INSTANCE CHECK
    if check_running_instance():
        logger.error("❌ Bot instance already running!")
        logger.error("   Use ./scripts/start_bot_clean.sh to restart")
        sys.exit(1)

    # CREATE PID FILE
    create_pid_file()

    # REGISTER CLEANUP
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_pid_file())
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_pid_file())

    try:
        # ... existing bot initialization code ...
        application = Application.builder().token(TOKEN).build()
        # ... register handlers ...

        logger.info("🤖 Bot started successfully")
        await application.run_polling(allowed_updates=Update.ALL_TYPES)

    finally:
        # ALWAYS CLEANUP
        cleanup_pid_file()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        cleanup_pid_file()
```

### 1.3 Enhancement: Startup Script with PID Check

**File**: `scripts/start_bot_clean.sh`

Add after TEST 2 (line 35):

```bash
# Test 2.5: Remove stale PID file
echo ""
echo "🗑️  TEST 2.5: Removing stale PID file..."
if [ -f ".bot.pid" ]; then
    PID_IN_FILE=$(cat .bot.pid)
    if ! ps -p $PID_IN_FILE > /dev/null 2>&1; then
        echo "   Stale PID file found (PID $PID_IN_FILE dead)"
        rm .bot.pid
        echo "✅ TEST 2.5 PASSED: Stale PID file removed"
    else
        echo "❌ TEST 2.5 FAILED: Process $PID_IN_FILE still running"
        exit 1
    fi
else
    echo "✅ TEST 2.5 PASSED: No PID file found"
fi
```

### 1.4 Validation: Run Tests

```bash
# Run TDD tests
python3 -m pytest tests/integration/test_bot_instance_management.py -v

# Expected: All 5 tests PASS

# Run health check
./scripts/check_bot_health.sh

# Expected:
# ✅ TEST 1 PASSED: Bot process running (PID: XXXXX)
# ✅ TEST 2 PASSED: Exactly 1 instance running
# ✅ TEST 3 PASSED: No conflicts in recent logs
# ✅ TEST 4 PASSED: Handlers registered
# ✅ TEST 5 PASSED: No recent exceptions
```

### 1.5 Success Criteria

✅ All 5 integration tests pass
✅ `./scripts/start_bot_clean.sh` completes without errors
✅ `./scripts/check_bot_health.sh` passes all tests
✅ Single bot process running: `ps aux | grep -E "[Pp]ython.*telegram_bot" | wc -l` returns 1
✅ Zero conflicts in `bot_output.log` for 24 hours

---

## Phase 2: Fix "Load Now" Root Cause (HIGH)

**Priority**: 🔴 HIGH - Core workflow broken
**Timeline**: 1 hour
**Files Modified**:
- `src/bot/ml_handlers/ml_training_local_path.py`
- `tests/integration/test_load_now_workflow.py`

### 2.1 TDD: Write Failing Test First

**Test File**: `tests/integration/test_load_now_workflow.py`

```python
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, CallbackQuery
from src.bot.ml_handlers.ml_training_local_path import MLTrainingLocalPathHandler
from src.core.state_manager import StateManager, MLTrainingState

@pytest.fixture
def test_data_file(tmp_path):
    """Create temporary CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "feature1,feature2,target\n"
        "1.0,2.0,100\n"
        "2.0,3.0,200\n"
        "3.0,4.0,300\n"
    )
    return str(csv_file)

@pytest.fixture
def mock_update():
    """Create mock Telegram update with callback query."""
    update = MagicMock(spec=Update)
    update.effective_user.id = 12345
    update.callback_query = MagicMock(spec=CallbackQuery)
    update.callback_query.data = "load_now"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.message.reply_text = AsyncMock()
    return update

@pytest.mark.asyncio
class TestLoadNowWorkflow:
    """TDD tests for Load Now workflow."""

    async def test_load_now_with_valid_path(self, mock_update, test_data_file):
        """Test 1: Load Now with valid CSV path completes successfully."""
        # ARRANGE
        handler = MLTrainingLocalPathHandler()
        state_manager = StateManager()

        # Set up state as if user just validated path
        state_manager.set_workflow_state(
            user_id=12345,
            state=MLTrainingState.CHOOSING_LOAD_OPTION,
            data={
                "validated_path": test_data_file,
                "file_size_mb": 0.001
            }
        )

        # ACT
        await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        mock_update.callback_query.answer.assert_called_once()

        # Check state transitioned to CONFIRMING_SCHEMA
        session = state_manager.get_session(12345)
        assert session.current_workflow_state == MLTrainingState.CONFIRMING_SCHEMA.value

        # Check data was loaded
        assert session.uploaded_data is not None
        assert "feature1" in session.uploaded_data.columns
        assert len(session.uploaded_data) == 3

        # Check schema detection ran
        assert "detected_target" in session.selections
        assert "detected_features" in session.selections

    async def test_load_now_with_missing_file(self, mock_update):
        """Test 2: Load Now with non-existent file shows proper error."""
        # ARRANGE
        handler = MLTrainingLocalPathHandler()
        state_manager = StateManager()

        state_manager.set_workflow_state(
            user_id=12345,
            state=MLTrainingState.CHOOSING_LOAD_OPTION,
            data={
                "validated_path": "/nonexistent/file.csv",
                "file_size_mb": 1.0
            }
        )

        # ACT
        await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        # Should show error message
        args = mock_update.callback_query.message.reply_text.call_args
        error_message = args[0][0]
        assert "File Not Found" in error_message
        assert "/nonexistent/file.csv" in error_message

        # State should revert to AWAITING_FILE_PATH
        session = state_manager.get_session(12345)
        assert session.current_workflow_state == MLTrainingState.AWAITING_FILE_PATH.value

    async def test_load_now_with_corrupted_csv(self, mock_update, tmp_path):
        """Test 3: Load Now with corrupted CSV shows proper error."""
        # ARRANGE
        corrupted_file = tmp_path / "corrupted.csv"
        corrupted_file.write_text("not,valid,csv\nwith\nmismatched\ncolumns")

        handler = MLTrainingLocalPathHandler()
        state_manager = StateManager()

        state_manager.set_workflow_state(
            user_id=12345,
            state=MLTrainingState.CHOOSING_LOAD_OPTION,
            data={
                "validated_path": str(corrupted_file),
                "file_size_mb": 0.001
            }
        )

        # ACT
        await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        args = mock_update.callback_query.message.reply_text.call_args
        error_message = args[0][0]
        assert "Loading Error" in error_message
        assert "corrupted.csv" in error_message

        # Should show actual pandas error details
        assert "ParserError" in error_message or "ValueError" in error_message

    async def test_load_now_exception_handling(self, mock_update, test_data_file):
        """Test 4: Unexpected exceptions are logged with full traceback."""
        # ARRANGE
        handler = MLTrainingLocalPathHandler()
        state_manager = StateManager()

        state_manager.set_workflow_state(
            user_id=12345,
            state=MLTrainingState.CHOOSING_LOAD_OPTION,
            data={
                "validated_path": test_data_file,
                "file_size_mb": 0.001
            }
        )

        # Force an unexpected exception by patching data loader
        with patch.object(
            handler.data_loader,
            "load_from_local_path",
            side_effect=RuntimeError("Simulated crash")
        ):
            # ACT
            await handler.handle_load_option_selection(mock_update, None)

        # ASSERT
        args = mock_update.callback_query.message.reply_text.call_args
        error_message = args[0][0]

        # Should show actual exception type and message
        assert "RuntimeError" in error_message
        assert "Simulated crash" in error_message
```

**Expected Result**: Tests 1-4 FAIL (current error handling is broken)

### 2.2 Root Cause Analysis with Logging

**File**: `src/bot/ml_handlers/ml_training_local_path.py`

Update `handle_load_option_selection()` at line 357:

```python
async def handle_load_option_selection(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle load option selection (immediate or defer)."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    session = self.state_manager.get_session(user_id)

    # ADD DETAILED LOGGING
    logger.info(f"[LOAD_OPTION] User {user_id} selected: {query.data}")
    logger.info(f"[LOAD_OPTION] Current state: {session.current_workflow_state}")
    logger.info(f"[LOAD_OPTION] Session data keys: {list(session.selections.keys())}")

    try:
        if query.data == "load_now":
            logger.info("[LOAD_NOW] Starting immediate load workflow")

            # Get validated path from session
            validated_path = session.selections.get("validated_path")
            if not validated_path:
                logger.error("[LOAD_NOW] ❌ No validated_path in session!")
                logger.error(f"[LOAD_NOW] Session selections: {session.selections}")
                raise ValueError("Session missing validated_path - state corruption detected")

            logger.info(f"[LOAD_NOW] Loading from: {validated_path}")

            # REPLACE DEFENSIVE TRY-EXCEPT WITH SPECIFIC ERROR HANDLING
            try:
                # Load data with detailed error propagation
                df = self.data_loader.load_from_local_path(validated_path)
                logger.info(f"[LOAD_NOW] ✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

            except FileNotFoundError as e:
                logger.error(f"[LOAD_NOW] ❌ File not found: {e}")
                error_msg = LocalPathMessages.format_path_error(
                    error_type="not_found",
                    path=validated_path,
                    error_details=f"FileNotFoundError: {str(e)}"
                )
                await query.message.reply_text(error_msg)
                self.state_manager.set_workflow_state(
                    user_id=user_id,
                    state=MLTrainingState.AWAITING_FILE_PATH
                )
                return

            except pd.errors.ParserError as e:
                logger.error(f"[LOAD_NOW] ❌ CSV parsing failed: {e}")
                error_msg = LocalPathMessages.format_path_error(
                    error_type="loading_error",
                    path=validated_path,
                    error_details=f"CSV ParserError: {str(e)}\n\nThe file may be corrupted or not a valid CSV."
                )
                await query.message.reply_text(error_msg)
                self.state_manager.set_workflow_state(
                    user_id=user_id,
                    state=MLTrainingState.AWAITING_FILE_PATH
                )
                return

            except PermissionError as e:
                logger.error(f"[LOAD_NOW] ❌ Permission denied: {e}")
                error_msg = LocalPathMessages.format_path_error(
                    error_type="loading_error",
                    path=validated_path,
                    error_details=f"PermissionError: {str(e)}\n\nCheck file permissions."
                )
                await query.message.reply_text(error_msg)
                self.state_manager.set_workflow_state(
                    user_id=user_id,
                    state=MLTrainingState.AWAITING_FILE_PATH
                )
                return

            except Exception as e:
                # LOG FULL TRACEBACK for unexpected errors
                logger.exception(f"[LOAD_NOW] ❌ Unexpected error loading data:")
                logger.error(f"[LOAD_NOW] Exception type: {type(e).__name__}")
                logger.error(f"[LOAD_NOW] Exception message: {str(e)}")

                # Show detailed error to user
                error_msg = LocalPathMessages.format_path_error(
                    error_type="unexpected",
                    path=validated_path,
                    error_details=f"{type(e).__name__}: {str(e)}"
                )
                await query.message.reply_text(error_msg)
                self.state_manager.set_workflow_state(
                    user_id=user_id,
                    state=MLTrainingState.AWAITING_FILE_PATH
                )
                return

            # AUTO-DETECT SCHEMA
            logger.info("[LOAD_NOW] Running schema detection...")
            try:
                detection_result = self.schema_detector.detect_schema(
                    df=df,
                    auto_suggest=True
                )
                logger.info(f"[LOAD_NOW] ✅ Schema detected: {detection_result.suggested_target}")

            except Exception as e:
                logger.exception("[LOAD_NOW] ❌ Schema detection failed:")
                error_msg = (
                    f"❌ **Schema Detection Failed**\n\n"
                    f"{type(e).__name__}: {str(e)}\n\n"
                    "Try manual schema input instead (Defer Loading)."
                )
                await query.message.reply_text(error_msg)
                self.state_manager.set_workflow_state(
                    user_id=user_id,
                    state=MLTrainingState.CHOOSING_LOAD_OPTION
                )
                return

            # SAVE TO SESSION
            self.state_manager.update_user_data(
                user_id=user_id,
                data=df
            )
            self.state_manager.update_selection(
                user_id=user_id,
                key="detected_target",
                value=detection_result.suggested_target
            )
            self.state_manager.update_selection(
                user_id=user_id,
                key="detected_features",
                value=detection_result.suggested_features
            )
            self.state_manager.update_selection(
                user_id=user_id,
                key="detected_task_type",
                value=detection_result.suggested_task_type
            )

            # SHOW CONFIRMATION
            summary = detection_result.summary
            confirmation_msg = LocalPathMessages.schema_confirmation_prompt(
                summary=summary,
                suggested_target=detection_result.suggested_target,
                suggested_features=detection_result.suggested_features,
                task_type=detection_result.suggested_task_type
            )

            # CREATE INLINE KEYBOARD
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [
                    InlineKeyboardButton("✅ Accept", callback_data="accept_schema"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject_schema")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.message.reply_text(
                confirmation_msg,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

            # TRANSITION STATE
            self.state_manager.set_workflow_state(
                user_id=user_id,
                state=MLTrainingState.CONFIRMING_SCHEMA
            )

            logger.info("[LOAD_NOW] ✅ Workflow completed successfully")

        elif query.data == "defer_load":
            # ... existing defer logic ...
            pass

    except Exception as e:
        # TOP-LEVEL EXCEPTION HANDLER (should never reach here)
        logger.exception(f"[LOAD_OPTION] ❌ TOP-LEVEL EXCEPTION (THIS IS A BUG):")
        logger.error(f"[LOAD_OPTION] User: {user_id}, Data: {query.data}")
        logger.error(f"[LOAD_OPTION] Session state: {session.current_workflow_state}")

        # Show error to user
        error_msg = (
            f"❌ **Critical Error**\n\n"
            f"An unexpected error occurred in the workflow handler.\n\n"
            f"**Error Type**: {type(e).__name__}\n"
            f"**Message**: {str(e)}\n\n"
            "Please report this to the developer with screenshot."
        )
        await query.message.reply_text(error_msg)
```

### 2.3 Validation: Reproduce and Fix

**Step 1**: Run with enhanced logging
```bash
# Start bot with logging
./scripts/start_bot_clean.sh

# In another terminal, watch logs
tail -f bot_output.log | grep "\[LOAD"
```

**Step 2**: Trigger the error
1. Send `/train` to bot
2. Select "📂 Use Local Path"
3. Enter path: `/tmp/test.csv` (create file first with test data)
4. Click "🔄 Load Now"

**Step 3**: Analyze logs
```bash
# Look for the actual error
grep -A 20 "\[LOAD_NOW\] ❌" bot_output.log
```

**Expected Findings** (one of these):
- Missing `validated_path` in session (state corruption)
- File permission issue
- Data loader import error
- Schema detector crash
- Pandas parsing error

**Step 4**: Fix the root cause based on logs

**Step 5**: Run TDD tests
```bash
pytest tests/integration/test_load_now_workflow.py -v
# Expected: All 4 tests PASS
```

### 2.4 Success Criteria

✅ All 4 integration tests pass
✅ "Load Now" button successfully loads `/tmp/test.csv`
✅ Schema detection displays correctly
✅ Accept/Reject buttons appear
✅ No "Unexpected Error" messages
✅ All errors show specific error type and message

---

## Phase 3: Fix Session Persistence (MEDIUM)

**Priority**: 🟡 MEDIUM - UX issue
**Timeline**: 2 hours
**Files Modified**:
- `src/core/state_manager.py`
- `src/bot/telegram_bot.py`
- `tests/integration/test_session_persistence.py`

### 3.1 TDD: Write Failing Tests First

**Test File**: `tests/integration/test_session_persistence.py`

```python
import pytest
import asyncio
from pathlib import Path
from src.core.state_manager import StateManager, MLTrainingState
from src.bot.telegram_bot import main as bot_main

@pytest.mark.asyncio
class TestSessionPersistence:
    """TDD tests for workflow state persistence."""

    async def test_session_survives_bot_restart(self):
        """Test 1: Session state persists across bot restarts."""
        # ARRANGE
        state_manager = StateManager()
        user_id = 12345

        # Create session with data
        state_manager.set_workflow_state(
            user_id=user_id,
            state=MLTrainingState.AWAITING_FILE_PATH,
            data={
                "validated_path": "/tmp/test.csv",
                "file_size_mb": 1.5
            }
        )

        # Save session to disk
        session_file = Path(f".sessions/user_{user_id}.json")
        state_manager.save_session_to_disk(user_id)

        # ACT - Simulate bot restart
        del state_manager  # Destroy in-memory state
        new_state_manager = StateManager()
        new_state_manager.load_session_from_disk(user_id)

        # ASSERT
        session = new_state_manager.get_session(user_id)
        assert session.current_workflow_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert session.selections["validated_path"] == "/tmp/test.csv"
        assert session.selections["file_size_mb"] == 1.5

    async def test_session_auto_saves_on_state_change(self):
        """Test 2: Session auto-saves when state changes."""
        # ARRANGE
        state_manager = StateManager()
        user_id = 12345
        session_file = Path(f".sessions/user_{user_id}.json")

        # ACT
        state_manager.set_workflow_state(
            user_id=user_id,
            state=MLTrainingState.CHOOSING_DATA_SOURCE
        )

        # ASSERT
        assert session_file.exists(), "Session file not created"

        # Load from disk and verify
        import json
        session_data = json.loads(session_file.read_text())
        assert session_data["current_workflow_state"] == MLTrainingState.CHOOSING_DATA_SOURCE.value

    async def test_session_cleanup_on_workflow_complete(self):
        """Test 3: Session cleaned up when workflow completes."""
        # ARRANGE
        state_manager = StateManager()
        user_id = 12345
        session_file = Path(f".sessions/user_{user_id}.json")

        state_manager.set_workflow_state(
            user_id=user_id,
            state=MLTrainingState.CONFIRMING_SCHEMA
        )
        assert session_file.exists()

        # ACT - Complete workflow
        state_manager.complete_workflow(user_id)

        # ASSERT
        assert not session_file.exists(), "Session file not cleaned up"
        session = state_manager.get_session(user_id)
        assert session.current_workflow_state is None

    async def test_session_loaded_on_first_message_after_restart(self):
        """Test 4: Session auto-loads when user sends message after restart."""
        # ARRANGE
        state_manager = StateManager()
        user_id = 12345

        # Create saved session
        state_manager.set_workflow_state(
            user_id=user_id,
            state=MLTrainingState.AWAITING_FILE_PATH,
            data={"previous_data": "preserved"}
        )
        state_manager.save_session_to_disk(user_id)

        # Simulate bot restart - clear memory
        del state_manager
        new_state_manager = StateManager()

        # ACT - User sends message (should auto-load)
        session = new_state_manager.get_session(user_id, auto_load=True)

        # ASSERT
        assert session.current_workflow_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert session.selections["previous_data"] == "preserved"
```

**Expected Result**: Tests 1-4 FAIL (persistence not implemented)

### 3.2 Implementation: Session Persistence

**File**: `src/core/state_manager.py`

Add persistence methods:

```python
import json
from pathlib import Path
from typing import Optional

class StateManager:
    """Manage workflow state with disk persistence."""

    SESSIONS_DIR = Path(".sessions")

    def __init__(self):
        self._sessions: Dict[int, UserSession] = {}
        self.SESSIONS_DIR.mkdir(exist_ok=True)

    def _get_session_file(self, user_id: int) -> Path:
        """Get session file path for user."""
        return self.SESSIONS_DIR / f"user_{user_id}.json"

    def save_session_to_disk(self, user_id: int) -> None:
        """Save user session to disk."""
        session = self._sessions.get(user_id)
        if not session:
            return

        session_file = self._get_session_file(user_id)

        # Convert to JSON-serializable dict
        session_data = {
            "user_id": session.user_id,
            "current_workflow_state": session.current_workflow_state,
            "selections": session.selections,
            "history": [
                {
                    "state": entry["state"],
                    "timestamp": entry["timestamp"].isoformat() if entry.get("timestamp") else None,
                    "data": entry.get("data", {})
                }
                for entry in session.history
            ]
        }

        # Write atomically
        temp_file = session_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(session_data, indent=2))
        temp_file.replace(session_file)

        logger.debug(f"Session saved to disk: user {user_id}")

    def load_session_from_disk(self, user_id: int) -> Optional[UserSession]:
        """Load user session from disk."""
        session_file = self._get_session_file(user_id)

        if not session_file.exists():
            return None

        try:
            session_data = json.loads(session_file.read_text())

            # Reconstruct session
            session = UserSession(
                user_id=session_data["user_id"],
                current_workflow_state=session_data.get("current_workflow_state"),
                uploaded_data=None,  # Data not persisted (too large)
                selections=session_data.get("selections", {}),
                history=session_data.get("history", [])
            )

            self._sessions[user_id] = session
            logger.info(f"Session loaded from disk: user {user_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session for user {user_id}: {e}")
            return None

    def complete_workflow(self, user_id: int) -> None:
        """Complete workflow and clean up session."""
        # Clear in-memory
        if user_id in self._sessions:
            del self._sessions[user_id]

        # Remove from disk
        session_file = self._get_session_file(user_id)
        if session_file.exists():
            session_file.unlink()

        logger.info(f"Workflow completed, session cleaned up: user {user_id}")

    def get_session(self, user_id: int, auto_load: bool = True) -> UserSession:
        """Get or create user session with auto-load."""
        # Check in-memory first
        if user_id in self._sessions:
            return self._sessions[user_id]

        # Try loading from disk if auto_load enabled
        if auto_load:
            session = self.load_session_from_disk(user_id)
            if session:
                return session

        # Create new session
        session = UserSession(user_id=user_id)
        self._sessions[user_id] = session
        return session

    def set_workflow_state(
        self,
        user_id: int,
        state: MLTrainingState,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set workflow state with auto-save."""
        session = self.get_session(user_id)

        # Validate transition
        if not self.is_valid_transition(session.current_workflow_state, state.value):
            raise ValueError(f"Invalid transition: {session.current_workflow_state} -> {state.value}")

        # Update state
        session.current_workflow_state = state.value

        # Add to history
        session.history.append({
            "state": state.value,
            "timestamp": datetime.now(),
            "data": data or {}
        })

        # Merge data into selections
        if data:
            session.selections.update(data)

        # AUTO-SAVE TO DISK
        self.save_session_to_disk(user_id)

        logger.info(f"State transition: user {user_id} -> {state.value}")
```

### 3.3 Integration: Bot Auto-Load

**File**: `src/bot/telegram_bot.py`

Update message handler:

```python
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all text messages with session auto-load."""
    user_id = update.effective_user.id

    # AUTO-LOAD SESSION (if exists on disk)
    session = state_manager.get_session(user_id, auto_load=True)

    if session.current_workflow_state:
        logger.info(
            f"Resuming workflow for user {user_id}: "
            f"state={session.current_workflow_state}"
        )

    # ... rest of handler logic ...
```

### 3.4 Validation: Run Tests

```bash
# Run TDD tests
pytest tests/integration/test_session_persistence.py -v

# Expected: All 4 tests PASS

# Manual test
1. Start bot: ./scripts/start_bot_clean.sh
2. Send /train, select local path
3. Enter path: /tmp/test.csv
4. STOP bot: Ctrl+C
5. Restart bot: ./scripts/start_bot_clean.sh
6. Send any message
7. Verify: Bot remembers you were at "awaiting file path" state
```

### 3.5 Success Criteria

✅ All 4 integration tests pass
✅ `.sessions/` directory created with user files
✅ Session persists across bot restarts
✅ Session auto-loads on first message after restart
✅ Session cleaned up when workflow completes

---

## Phase 4: End-to-End Workflow Testing (VALIDATION)

**Priority**: 🟢 VALIDATION - Ensure everything works
**Timeline**: 3 hours
**Files Created**:
- `tests/e2e/test_full_local_path_workflow.py`
- `tests/e2e/test_full_telegram_upload_workflow.py`
- `tests/e2e/test_full_prediction_workflow.py`

### 4.1 E2E Test: Local Path Workflow

**Test File**: `tests/e2e/test_full_local_path_workflow.py`

```python
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, CallbackQuery
from src.bot.telegram_bot import application

@pytest.fixture
def test_csv(tmp_path):
    """Create test CSV with realistic data."""
    csv_file = tmp_path / "housing.csv"
    csv_file.write_text(
        "sqft,bedrooms,bathrooms,price\n"
        "1500,3,2,300000\n"
        "2000,4,3,450000\n"
        "1200,2,1,250000\n"
        "1800,3,2,380000\n"
        "2200,4,3,520000\n"
    )

    # Add to allowed directories in config
    import yaml
    config_file = Path("config/config.yaml")
    config = yaml.safe_load(config_file.read_text())
    if str(tmp_path) not in config["local_data"]["allowed_directories"]:
        config["local_data"]["allowed_directories"].append(str(tmp_path))
        config_file.write_text(yaml.dump(config))

    return str(csv_file)

@pytest.mark.asyncio
@pytest.mark.e2e
class TestFullLocalPathWorkflow:
    """E2E test for complete local path ML training workflow."""

    async def test_complete_local_path_workflow(self, test_csv):
        """
        Test complete workflow from /train to model trained.

        Steps:
        1. User sends /train
        2. Bot shows data source selection
        3. User selects "Local Path"
        4. Bot prompts for file path
        5. User enters path
        6. Bot validates and shows load options
        7. User selects "Load Now"
        8. Bot loads data, detects schema, shows confirmation
        9. User accepts schema
        10. Bot proceeds to model selection
        11. User selects model type
        12. Bot trains model
        13. Model saved successfully
        """

        user_id = 99999  # Test user

        # STEP 1: /train command
        update1 = create_mock_update(user_id, "/train", is_command=True)
        await application.process_update(update1)

        response1 = get_last_message_text(update1)
        assert "How would you like to provide your training data?" in response1
        assert "📤 Upload File" in response1
        assert "📂 Use Local Path" in response1

        # STEP 2: Select "Local Path"
        update2 = create_mock_callback_update(user_id, "use_local_path")
        await application.process_update(update2)

        response2 = get_last_message_text(update2)
        assert "Local File Path" in response2
        assert "Allowed directories:" in response2

        # STEP 3: Enter file path
        update3 = create_mock_update(user_id, test_csv)
        await application.process_update(update3)

        response3 = get_last_message_text(update3)
        assert "Path Validated" in response3
        assert "Choose Loading Strategy" in response3
        assert "🔄 Load Now" in response3
        assert "⏳ Defer Loading" in response3

        # STEP 4: Select "Load Now"
        update4 = create_mock_callback_update(user_id, "load_now")
        await application.process_update(update4)

        response4 = get_last_message_text(update4)
        assert "Dataset Summary" in response4
        assert "Auto-Detected" in response4
        assert "Target:" in response4
        assert "price" in response4  # Detected target
        assert "Features:" in response4

        # STEP 5: Accept schema
        update5 = create_mock_callback_update(user_id, "accept_schema")
        await application.process_update(update5)

        response5 = get_last_message_text(update5)
        assert "Schema Accepted" in response5
        assert "Proceeding to target selection" in response5

        # STEP 6: Confirm target (price)
        update6 = create_mock_update(user_id, "price")
        await application.process_update(update6)

        response6 = get_last_message_text(update6)
        assert "Select features" in response6

        # STEP 7: Select features (sqft, bedrooms, bathrooms)
        update7 = create_mock_update(user_id, "sqft,bedrooms,bathrooms")
        await application.process_update(update7)

        response7 = get_last_message_text(update7)
        assert "Model type" in response7 or "task type" in response7.lower()

        # STEP 8: Select model (linear regression)
        update8 = create_mock_update(user_id, "linear")
        await application.process_update(update8)

        response8 = get_last_message_text(update8)
        assert "Training started" in response8 or "Training" in response8

        # Wait for training to complete (async)
        await asyncio.sleep(5)

        # STEP 9: Check training completion
        final_update = create_mock_update(user_id, "/models")
        await application.process_update(final_update)

        final_response = get_last_message_text(final_update)
        assert "model_99999_linear" in final_response
        assert "R²" in final_response or "MSE" in final_response

        # CLEANUP
        from src.engines.ml_engine import MLEngine
        ml_engine = MLEngine.get_default()
        ml_engine.delete_model(user_id, "model_99999_linear")

def create_mock_update(user_id: int, text: str, is_command: bool = False):
    """Helper to create mock update."""
    update = MagicMock(spec=Update)
    update.effective_user.id = user_id
    update.message = MagicMock(spec=Message)
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update

def create_mock_callback_update(user_id: int, callback_data: str):
    """Helper to create mock callback update."""
    update = MagicMock(spec=Update)
    update.effective_user.id = user_id
    update.callback_query = MagicMock(spec=CallbackQuery)
    update.callback_query.data = callback_data
    update.callback_query.answer = AsyncMock()
    update.callback_query.message.reply_text = AsyncMock()
    return update

def get_last_message_text(update):
    """Extract last message text from mock."""
    if hasattr(update, 'callback_query'):
        calls = update.callback_query.message.reply_text.call_args_list
    else:
        calls = update.message.reply_text.call_args_list

    if calls:
        return calls[-1][0][0]  # First positional arg of last call
    return ""
```

### 4.2 E2E Test: Deferred Loading Workflow

**Test File**: `tests/e2e/test_deferred_loading_workflow.py`

```python
@pytest.mark.asyncio
@pytest.mark.e2e
class TestDeferredLoadingWorkflow:
    """E2E test for deferred loading with manual schema."""

    async def test_deferred_loading_with_manual_schema(self, test_csv):
        """
        Test deferred loading workflow.

        Steps:
        1-3. Same as local path workflow
        4. User selects "Defer Loading"
        5. Bot prompts for manual schema
        6. User provides schema in key-value format
        7. Bot validates and accepts schema
        8. Bot proceeds to model selection
        9. Data loads at training time (not during schema input)
        """

        user_id = 88888

        # ... steps 1-3 same as test_complete_local_path_workflow ...

        # STEP 4: Select "Defer Loading"
        update4 = create_mock_callback_update(user_id, "defer_load")
        await application.process_update(update4)

        response4 = get_last_message_text(update4)
        assert "Manual Schema Input" in response4
        assert "Format 1 - Key-Value" in response4
        assert "target:" in response4
        assert "features:" in response4

        # STEP 5: Provide manual schema
        schema_input = "target: price\nfeatures: sqft, bedrooms, bathrooms"
        update5 = create_mock_update(user_id, schema_input)
        await application.process_update(update5)

        response5 = get_last_message_text(update5)
        assert "Schema Accepted" in response5
        assert "Target: price" in response5
        assert "Features: 3" in response5
        assert "Data will load at training time" in response5

        # STEP 6: Proceed to model selection
        # ... continue with model training ...

        # VERIFY: Data loaded during training, not during schema input
        from src.core.state_manager import StateManager
        state_manager = StateManager()
        session = state_manager.get_session(user_id)

        # At this point, uploaded_data should be None
        assert session.uploaded_data is None, "Data should not be loaded yet"

        # Trigger training
        update_train = create_mock_update(user_id, "linear")
        await application.process_update(update_train)

        # Now data should be loaded
        await asyncio.sleep(2)
        session = state_manager.get_session(user_id)
        assert session.uploaded_data is not None, "Data should be loaded during training"
```

### 4.3 Validation: Run All E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -v -m e2e

# Expected output:
# test_full_local_path_workflow.py::test_complete_local_path_workflow PASSED
# test_deferred_loading_workflow.py::test_deferred_loading_with_manual_schema PASSED
# test_full_telegram_upload_workflow.py::test_upload_csv_and_train PASSED
# test_full_prediction_workflow.py::test_train_and_predict PASSED
```

### 4.4 Success Criteria

✅ All E2E tests pass (4/4)
✅ Local path workflow works end-to-end (10+ steps)
✅ Deferred loading workflow works end-to-end
✅ Telegram upload workflow still works (regression test)
✅ Train + predict workflow works end-to-end

---

## Phase 5: Missing Architecture Components (OPTIONAL)

**Priority**: 🟢 LOW - Optional refactoring
**Timeline**: 4 hours (if pursued)
**Status**: DEFERRED

### 5.1 Gap Analysis

From `dev/planning/integration-layer.md`, the following components were planned but NOT implemented:

1. **IntegrationLayer** (300 LOC) - Orchestrates workflow, delegates to handlers
2. **ResponseBuilder** (250 LOC) - Consistent message formatting
3. **CommandHandlers** (200 LOC) - /train, /models, /predict handlers
4. **Orchestrator Refactoring** - Remove ML-specific logic

**Decision**: DEFER to future iteration

**Rationale**:
- Core workflows are functional (after Phases 1-4)
- Refactoring introduces risk without immediate user value
- Better to validate current implementation first
- Can be addressed in future PR focused on architecture

### 5.2 If Pursued (Future Work)

Create separate PR with:
1. TDD tests for IntegrationLayer
2. Incremental refactoring (one component at a time)
3. Preserve existing functionality (no breaking changes)
4. Comprehensive regression testing

**Estimated Timeline**: 1-2 weeks as dedicated project

---

## Implementation Schedule

### Day 1 (4 hours)
- **Hour 1**: Phase 1 - Fix bot instance conflicts
  - Write 5 TDD tests
  - Implement PID file locking
  - Validate with health check
- **Hour 2**: Phase 2 - Root cause analysis
  - Add detailed logging
  - Reproduce error with logs
  - Identify actual exception
- **Hour 3**: Phase 2 - Fix Load Now
  - Implement specific error handling
  - Write 4 TDD tests
  - Validate with real data
- **Hour 4**: Phase 3 - Session persistence (part 1)
  - Write 4 TDD tests
  - Implement disk save/load

### Day 2 (4 hours)
- **Hour 1**: Phase 3 - Session persistence (part 2)
  - Integrate auto-load in bot
  - Validate persistence works
- **Hour 2**: Phase 4 - E2E testing (part 1)
  - Write local path E2E test
  - Write deferred loading E2E test
- **Hour 3**: Phase 4 - E2E testing (part 2)
  - Write upload E2E test
  - Write prediction E2E test
- **Hour 4**: Validation & Documentation
  - Run full test suite
  - Document all fixes
  - Update CLAUDE.md

---

## Testing Strategy

### Test Pyramid

```
        E2E Tests (4)
       /           \
      /   Integration  \
     /    Tests (13)    \
    /___________________\
    Unit Tests (127 existing)
```

### Test Coverage Goals

- **Unit Tests**: 95%+ coverage (already achieved)
- **Integration Tests**: 90%+ coverage (add 13 new tests)
- **E2E Tests**: 100% critical workflows (add 4 new tests)

### Test Execution

```bash
# Quick validation (unit only)
pytest tests/unit/ -v

# Full validation (unit + integration)
pytest tests/ --ignore=tests/e2e/ -v

# Complete validation (all tests)
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Rollback Plan

### If Phase 1 Fails
- Revert `src/bot/telegram_bot.py` changes
- Use manual process management: `killall python3 && ./start_bot_clean.sh`
- Continue with Phases 2-3

### If Phase 2 Fails
- Keep enhanced logging (useful for debugging)
- Document root cause in GitHub issue
- Continue with Phase 3 (independent)

### If Phase 3 Fails
- Session persistence is UX enhancement, not critical
- Document failure, defer to future iteration
- Proceed with Phase 4

### If Phase 4 Fails
- Review test setup (mocking issues likely)
- Manual E2E testing as fallback
- Document test patterns for future

---

## Success Validation Checklist

### Phase 1 Complete
- [ ] 5 integration tests pass
- [ ] `start_bot_clean.sh` runs without errors
- [ ] `check_bot_health.sh` passes all tests
- [ ] Single process running: `ps aux | grep telegram_bot | wc -l` = 1
- [ ] Zero conflicts in logs for 24 hours

### Phase 2 Complete
- [ ] 4 integration tests pass
- [ ] "Load Now" button works with test data
- [ ] Schema detection displays correctly
- [ ] Accept/Reject buttons functional
- [ ] No generic "Unexpected Error" messages

### Phase 3 Complete
- [ ] 4 integration tests pass
- [ ] `.sessions/` directory exists with user files
- [ ] Session persists across restarts
- [ ] Session auto-loads on message
- [ ] Session cleans up on completion

### Phase 4 Complete
- [ ] 4 E2E tests pass
- [ ] Local path workflow (10 steps) completes
- [ ] Deferred loading workflow completes
- [ ] Upload workflow still works (regression)
- [ ] Train + predict workflow completes

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 1 breaks bot startup | Low | High | Rollback plan, test in dev first |
| Phase 2 reveals complex bug | Medium | High | Enhanced logging identifies root cause |
| Phase 3 affects performance | Low | Low | Small file sizes, atomic writes |
| Phase 4 tests are flaky | Medium | Low | Mock external dependencies properly |
| Time estimate exceeded | Medium | Low | Phases are independent, can defer |

---

## Post-Implementation Tasks

### Documentation Updates
1. Update `CLAUDE.md` with session persistence details
2. Add troubleshooting section for common errors
3. Document test execution procedures

### Monitoring
1. Set up log monitoring for conflicts
2. Track session file sizes (.sessions/ directory)
3. Monitor test execution times

### Future Improvements
1. Session expiration (auto-cleanup after 7 days)
2. Session compression (reduce disk usage)
3. Implement Phase 5 architecture (if needed)
4. Performance optimization based on logs

---

## Appendix: Key Files Reference

### Core Implementation
- `src/bot/telegram_bot.py` - Bot entry point, main() function
- `src/bot/ml_handlers/ml_training_local_path.py` - Local path workflow (1807 lines)
- `src/core/state_manager.py` - State machine management (459 lines)
- `src/utils/path_validator.py` - Security validation (350 lines)
- `src/utils/schema_detector.py` - Auto-schema detection (550 lines)

### Scripts
- `scripts/start_bot_clean.sh` - TDD-driven clean startup (108 lines)
- `scripts/check_bot_health.sh` - TDD-driven health check (114 lines)

### Configuration
- `config/config.yaml` - Bot configuration
- `.env` - Environment variables (TELEGRAM_BOT_TOKEN)

### Planning Documentation
- `dev/planning/file-path-training.md` - Phase 1 plan (1910 lines)
- `dev/planning/file-path-training-2.md` - Phase 2 plan (682 lines)
- `dev/planning/state-manager.md` - State machine design (1032 lines)
- `dev/planning/integration-layer.md` - Future architecture (1536 lines)

---

**End of Workflow Fix Plan 1**
