"""Integration tests for session persistence across bot restarts."""
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

@pytest.mark.asyncio
class TestSessionPersistence:
    """TDD tests for session persistence functionality."""

    async def test_persist_session_across_restart(self, tmp_path):
        """Test 1: Session persists to disk and loads correctly after restart."""
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType

        # ARRANGE - Create session with data
        sessions_dir = tmp_path / ".sessions"
        state_manager = StateManager(sessions_dir=str(sessions_dir))

        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="conv_123"
        )
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.file_path = "/tmp/data.csv"
        session.selections = {"target": "price", "features": ["sqft", "bedrooms"]}
        await state_manager.update_session(session)

        # ACT - Save session to disk
        await state_manager.save_session_to_disk(user_id=12345)

        # Simulate restart: create new StateManager instance
        state_manager_after_restart = StateManager(sessions_dir=str(sessions_dir))

        # Load session from disk
        loaded_session = await state_manager_after_restart.load_session_from_disk(user_id=12345)

        # ASSERT - Session data restored correctly
        assert loaded_session is not None
        assert loaded_session.user_id == 12345
        assert loaded_session.conversation_id == "conv_123"
        assert loaded_session.workflow_type == WorkflowType.ML_TRAINING
        assert loaded_session.current_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert loaded_session.file_path == "/tmp/data.csv"
        assert loaded_session.selections["target"] == "price"
        assert "sqft" in loaded_session.selections["features"]

        # Verify session file exists on disk
        session_file = sessions_dir / "user_12345.json"
        assert session_file.exists()

    async def test_auto_save_on_state_change(self, tmp_path):
        """Test 2: Session automatically saves to disk when state changes."""
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType

        # ARRANGE
        sessions_dir = tmp_path / ".sessions"
        state_manager = StateManager(sessions_dir=str(sessions_dir), auto_save=True)

        session = await state_manager.get_or_create_session(
            user_id=67890,
            conversation_id="conv_456"
        )
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        await state_manager.update_session(session)

        # ACT - Change state (should trigger auto-save)
        session.current_state = MLTrainingState.CHOOSING_LOAD_OPTION.value
        await state_manager.update_session(session)

        # ASSERT - Session file created without explicit save call
        session_file = sessions_dir / "user_67890.json"
        assert session_file.exists()

        # Verify saved data matches current state
        saved_data = json.loads(session_file.read_text())
        assert saved_data["current_state"] == MLTrainingState.CHOOSING_LOAD_OPTION.value

    async def test_cleanup_on_workflow_completion(self, tmp_path):
        """Test 3: Session cleaned up when workflow completes."""
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType

        # ARRANGE - Create and save session
        sessions_dir = tmp_path / ".sessions"
        state_manager = StateManager(sessions_dir=str(sessions_dir))

        session = await state_manager.get_or_create_session(
            user_id=11111,
            conversation_id="conv_789"
        )
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.TRAINING.value
        await state_manager.update_session(session)
        await state_manager.save_session_to_disk(user_id=11111)

        # Verify session exists
        session_file = sessions_dir / "user_11111.json"
        assert session_file.exists()

        # ACT - Complete workflow
        await state_manager.complete_workflow(user_id=11111)

        # ASSERT - Session removed from disk
        assert not session_file.exists()

        # Session also removed from memory
        memory_session = await state_manager.get_session(11111, "conv_789")
        assert memory_session is None or memory_session.workflow_type is None

    async def test_auto_load_on_first_message(self, tmp_path):
        """Test 4: Session auto-loads when user sends message after restart."""
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType

        # ARRANGE - Create session and save to disk
        sessions_dir = tmp_path / ".sessions"
        state_manager_before = StateManager(sessions_dir=str(sessions_dir))

        session = await state_manager_before.get_or_create_session(
            user_id=22222,
            conversation_id="conv_999"
        )
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value
        session.detected_schema = {"target": "price", "features": ["sqft"]}
        await state_manager_before.update_session(session)
        await state_manager_before.save_session_to_disk(user_id=22222)

        # Simulate restart: new StateManager instance (empty memory)
        state_manager_after = StateManager(sessions_dir=str(sessions_dir))

        # ACT - Get session with auto_load=True (simulates user sending message)
        loaded_session = await state_manager_after.get_session(
            user_id=22222,
            conversation_id="conv_999",
            auto_load=True
        )

        # ASSERT - Session loaded from disk into memory
        assert loaded_session is not None
        assert loaded_session.workflow_type == WorkflowType.ML_TRAINING
        assert loaded_session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value
        assert loaded_session.detected_schema["target"] == "price"

        # Verify session now in memory for future calls
        cached_session = await state_manager_after.get_session(
            user_id=22222,
            conversation_id="conv_999",
            auto_load=False
        )
        assert cached_session is not None
        assert cached_session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value

    async def test_session_file_not_found_returns_none(self, tmp_path):
        """Test 5: Loading non-existent session returns None gracefully."""
        from src.core.state_manager import StateManager

        # ARRANGE
        sessions_dir = tmp_path / ".sessions"
        state_manager = StateManager(sessions_dir=str(sessions_dir))

        # ACT - Try to load session that doesn't exist
        loaded_session = await state_manager.load_session_from_disk(user_id=99999)

        # ASSERT - Returns None without raising exception
        assert loaded_session is None

    async def test_session_persistence_excludes_large_data(self, tmp_path):
        """Test 6: uploaded_data (DataFrame) not persisted to disk."""
        import pandas as pd
        from src.core.state_manager import StateManager, MLTrainingState, WorkflowType

        # ARRANGE
        sessions_dir = tmp_path / ".sessions"
        state_manager = StateManager(sessions_dir=str(sessions_dir))

        session = await state_manager.get_or_create_session(
            user_id=33333,
            conversation_id="conv_data"
        )
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value

        # Add large DataFrame (should not be persisted)
        session.uploaded_data = pd.DataFrame({
            'col1': range(10000),
            'col2': range(10000)
        })

        await state_manager.update_session(session)

        # ACT - Save to disk
        await state_manager.save_session_to_disk(user_id=33333)

        # ASSERT - Session file exists but is small (no DataFrame)
        session_file = sessions_dir / "user_33333.json"
        assert session_file.exists()

        # File should be small (< 10KB, not ~1MB from DataFrame)
        file_size = session_file.stat().st_size
        assert file_size < 10000  # Less than 10KB

        # Load and verify DataFrame is not in loaded session
        state_manager_new = StateManager(sessions_dir=str(sessions_dir))
        loaded_session = await state_manager_new.load_session_from_disk(user_id=33333)

        assert loaded_session.uploaded_data is None
        # But other data is present
        assert loaded_session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value
