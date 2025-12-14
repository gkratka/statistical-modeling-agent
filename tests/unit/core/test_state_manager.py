"""
Unit tests for StateManager class.

Tests session management, workflow operations, and data storage.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.core.state_manager import (
    StateManager,
    StateManagerConfig,
    UserSession,
    WorkflowType,
    MLTrainingState
)
from src.utils.exceptions import (
    SessionNotFoundError,
    SessionLimitError,
    InvalidStateTransitionError,
    DataSizeLimitError
)


@pytest.mark.asyncio
class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create state manager with default config."""
        return StateManager()

    @pytest.fixture
    def custom_manager(self):
        """Create state manager with custom config."""
        config = StateManagerConfig(
            session_timeout_minutes=10,
            max_data_size_mb=50,
            max_concurrent_sessions=5
        )
        return StateManager(config)

    async def test_get_or_create_session_new(self, manager):
        """Test creating new session."""
        session = await manager.get_or_create_session(
            user_id=123,
            conversation_id="conv_1"
        )

        assert session is not None
        assert session.user_id == 123
        assert session.conversation_id == "conv_1"
        assert session.workflow_type is None

    async def test_get_or_create_session_existing(self, manager):
        """Test getting existing session."""
        # Create session
        session1 = await manager.get_or_create_session(123, "conv_1")
        original_created_at = session1.created_at

        # Get same session
        session2 = await manager.get_or_create_session(123, "conv_1")

        assert session2.session_key == session1.session_key
        assert session2.created_at == original_created_at

    async def test_get_or_create_session_updates_activity(self, manager):
        """Test that getting session updates last activity."""
        session1 = await manager.get_or_create_session(123, "conv_1")
        first_activity = session1.last_activity

        # Wait a bit and get again
        import asyncio
        await asyncio.sleep(0.01)

        session2 = await manager.get_or_create_session(123, "conv_1")

        assert session2.last_activity > first_activity

    async def test_get_or_create_session_limit(self, custom_manager):
        """Test session limit enforcement."""
        # Create max sessions (5 in custom config)
        for i in range(1, 6):  # Start at 1, not 0
            await custom_manager.get_or_create_session(
                user_id=i,
                conversation_id=f"conv_{i}"
            )

        # Trying to create 6th should fail
        with pytest.raises(SessionLimitError) as exc_info:
            await custom_manager.get_or_create_session(999, "conv_999")

        assert exc_info.value.current_count == 5
        assert exc_info.value.limit == 5

    async def test_get_session_exists(self, manager):
        """Test getting existing session."""
        await manager.get_or_create_session(123, "conv_1")

        session = await manager.get_session(123, "conv_1")

        assert session is not None
        assert session.user_id == 123

    async def test_get_session_not_exists(self, manager):
        """Test getting non-existent session returns None."""
        session = await manager.get_session(999, "conv_999")

        assert session is None

    async def test_get_session_expired(self, manager):
        """Test getting expired session returns None."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Manually expire session
        session.last_activity = datetime.now() - timedelta(minutes=40)

        result = await manager.get_session(123, "conv_1")

        assert result is None

    async def test_update_session(self, manager):
        """Test updating session."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Modify session
        session.selections["key"] = "value"

        # Update
        await manager.update_session(session)

        # Get and verify
        retrieved = await manager.get_session(123, "conv_1")
        assert retrieved.selections["key"] == "value"

    async def test_update_session_not_found(self, manager):
        """Test updating non-existent session raises error."""
        session = UserSession(user_id=999, conversation_id="conv_999")

        with pytest.raises(SessionNotFoundError):
            await manager.update_session(session)

    async def test_delete_session(self, manager):
        """Test deleting session."""
        await manager.get_or_create_session(123, "conv_1")

        # Delete
        await manager.delete_session(123, "conv_1")

        # Verify deleted
        session = await manager.get_session(123, "conv_1")
        assert session is None

    async def test_delete_session_not_exists(self, manager):
        """Test deleting non-existent session doesn't error."""
        # Should not raise error
        await manager.delete_session(999, "conv_999")

    async def test_start_workflow(self, manager):
        """Test starting workflow."""
        session = await manager.get_or_create_session(123, "conv_1")

        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Verify workflow started
        assert session.workflow_type == WorkflowType.ML_TRAINING
        assert session.current_state == MLTrainingState.AWAITING_DATA.value

    async def test_start_workflow_already_active(self, manager):
        """Test starting workflow when one already active."""
        session = await manager.get_or_create_session(123, "conv_1")

        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Try to start another
        with pytest.raises(InvalidStateTransitionError):
            await manager.start_workflow(session, WorkflowType.ML_PREDICTION)

    async def test_transition_state_success(self, manager):
        """Test successful state transition."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Start workflow
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Upload data (prerequisite)
        session.uploaded_data = pd.DataFrame({"x": [1, 2, 3]})

        # Transition
        success, error_msg, missing = await manager.transition_state(
            session,
            MLTrainingState.SELECTING_TARGET.value
        )

        assert success
        assert session.current_state == MLTrainingState.SELECTING_TARGET.value

    async def test_transition_state_no_workflow(self, manager):
        """Test transition without active workflow."""
        session = await manager.get_or_create_session(123, "conv_1")

        with pytest.raises(InvalidStateTransitionError):
            await manager.transition_state(
                session,
                MLTrainingState.SELECTING_TARGET.value
            )

    async def test_transition_state_invalid(self, manager):
        """Test invalid state transition."""
        session = await manager.get_or_create_session(123, "conv_1")
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Try to skip states
        success, error_msg, missing = await manager.transition_state(
            session,
            MLTrainingState.TRAINING.value
        )

        assert not success
        assert "Invalid transition" in error_msg

    async def test_transition_state_prerequisites_not_met(self, manager):
        """Test transition when prerequisites not met."""
        session = await manager.get_or_create_session(123, "conv_1")
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Try to transition without uploading data
        success, error_msg, missing = await manager.transition_state(
            session,
            MLTrainingState.SELECTING_TARGET.value
        )

        assert not success
        assert "Prerequisites not met" in error_msg
        assert "uploaded_data" in missing

    async def test_cancel_workflow(self, manager):
        """Test canceling workflow."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Start and add some state
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.selections["key"] = "value"

        # Cancel
        await manager.cancel_workflow(session)

        assert session.workflow_type is None
        assert session.current_state is None
        assert len(session.selections) == 0

    async def test_store_data(self, manager):
        """Test storing DataFrame."""
        session = await manager.get_or_create_session(123, "conv_1")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        await manager.store_data(session, df)

        assert session.uploaded_data is not None
        assert len(session.uploaded_data) == 3

    async def test_store_data_size_limit(self, custom_manager):
        """Test data size limit enforcement."""
        session = await custom_manager.get_or_create_session(123, "conv_1")

        # Create large DataFrame that exceeds 50MB limit
        df = pd.DataFrame({
            f'col_{i}': range(1000000) for i in range(10)
        })

        with pytest.raises(DataSizeLimitError) as exc_info:
            await custom_manager.store_data(session, df)

        assert exc_info.value.limit_mb == 50

    async def test_get_data(self, manager):
        """Test getting DataFrame."""
        session = await manager.get_or_create_session(123, "conv_1")
        df = pd.DataFrame({"x": [1, 2, 3]})

        await manager.store_data(session, df)

        retrieved = await manager.get_data(session)

        assert retrieved is not None
        assert len(retrieved) == 3

    async def test_get_data_none(self, manager):
        """Test getting data when none stored."""
        session = await manager.get_or_create_session(123, "conv_1")

        data = await manager.get_data(session)

        assert data is None

    async def test_add_to_history(self, manager):
        """Test adding to conversation history."""
        session = await manager.get_or_create_session(123, "conv_1")

        await manager.add_to_history(session, "user", "Hello")
        await manager.add_to_history(session, "assistant", "Hi there")

        assert len(session.history) == 2
        assert session.history[0]["message"] == "Hello"
        assert session.history[1]["message"] == "Hi there"

    async def test_get_history(self, manager):
        """Test getting recent history."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Add 10 messages
        for i in range(10):
            await manager.add_to_history(session, "user", f"Message {i}")

        # Get last 5
        history = await manager.get_history(session, n_messages=5)

        assert len(history) == 5
        assert history[0]["message"] == "Message 5"
        assert history[-1]["message"] == "Message 9"

    async def test_get_history_empty(self, manager):
        """Test getting history when empty."""
        session = await manager.get_or_create_session(123, "conv_1")

        history = await manager.get_history(session)

        assert history == []

    async def test_cleanup_expired_sessions(self, manager):
        """Test cleaning up expired sessions."""
        # Create some sessions
        session1 = await manager.get_or_create_session(1, "conv_1")
        session2 = await manager.get_or_create_session(2, "conv_2")
        session3 = await manager.get_or_create_session(3, "conv_3")

        # Expire session1 and session2
        session1.last_activity = datetime.now() - timedelta(minutes=40)
        session2.last_activity = datetime.now() - timedelta(minutes=40)

        # Cleanup
        count = await manager.cleanup_expired_sessions()

        assert count == 2

        # Verify only session3 remains
        assert await manager.get_session(1, "conv_1") is None
        assert await manager.get_session(2, "conv_2") is None
        assert await manager.get_session(3, "conv_3") is not None

    async def test_get_active_session_count(self, manager):
        """Test getting active session count."""
        assert await manager.get_active_session_count() == 0

        await manager.get_or_create_session(1, "conv_1")
        await manager.get_or_create_session(2, "conv_2")

        assert await manager.get_active_session_count() == 2

    async def test_get_session_timeout_warning_no_warning(self, manager):
        """Test timeout warning when session is fresh."""
        session = await manager.get_or_create_session(123, "conv_1")

        warning = await manager.get_session_timeout_warning(session)

        assert warning is None

    async def test_get_session_timeout_warning_approaching(self, manager):
        """Test timeout warning when approaching timeout."""
        session = await manager.get_or_create_session(123, "conv_1")

        # Set activity to 28 minutes ago (2 minutes left, 80%+ elapsed)
        session.last_activity = datetime.now() - timedelta(minutes=28)

        warning = await manager.get_session_timeout_warning(session)

        assert warning is not None
        assert "expire" in warning.lower()

    async def test_concurrent_access(self, manager):
        """Test concurrent session access."""
        import asyncio

        async def access_session(user_id: int):
            session = await manager.get_or_create_session(user_id, f"conv_{user_id}")
            await manager.add_to_history(session, "user", f"Message from {user_id}")
            return session

        # Access sessions concurrently
        results = await asyncio.gather(
            access_session(1),
            access_session(2),
            access_session(3),
            access_session(1),  # Same as first one
        )

        # Verify all succeeded
        assert len(results) == 4
        assert results[0].user_id == 1
        assert results[3].user_id == 1

        # Verify session count is 3 (not 4, since user 1 accessed twice)
        count = await manager.get_active_session_count()
        assert count == 3

    async def test_reset_session_clears_all_state(self, manager):
        """Test that reset_session completely clears all session state."""
        # Create session and populate with various data
        session = await manager.get_or_create_session(123, "conv_1")

        # Add workflow state
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.current_state = MLTrainingState.SELECTING_TARGET.value
        session.selections["target"] = "price"
        session.selections["features"] = ["sqft", "bedrooms"]
        session.model_ids.append("model_123")

        # Add data
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        await manager.store_data(session, df)

        # Add conversation history
        await manager.add_to_history(session, "user", "Train a model")
        await manager.add_to_history(session, "assistant", "Sure!")

        # Add local path workflow data
        session.data_source = "local_path"
        session.file_path = "/tmp/data.csv"
        session.detected_schema = {"target": "price"}
        session.load_deferred = True
        session.manual_schema = {"features": ["col1", "col2"]}

        # Add password authentication state
        session.dynamic_allowed_directories.append("/tmp")
        session.pending_auth_path = "/tmp/secure"
        session.password_attempts = 2

        # Add back button state
        session.last_back_action = 12345.0

        # Add prediction workflow data
        session.compatible_models = [{"id": "model_1"}]

        # Save state snapshot for back navigation
        session.save_state_snapshot()

        await manager.update_session(session)

        # Perform reset
        await manager.reset_session(123, "conv_1")

        # Get session after reset
        reset_session = await manager.get_session(123, "conv_1")

        # Verify ALL state is cleared
        assert reset_session is not None  # Session still exists
        assert reset_session.workflow_type is None
        assert reset_session.current_state is None
        assert len(reset_session.selections) == 0
        assert len(reset_session.model_ids) == 0
        assert len(reset_session.history) == 0
        assert reset_session.uploaded_data is None

        # Verify local path workflow data cleared
        assert reset_session.data_source is None
        assert reset_session.file_path is None
        assert reset_session.detected_schema is None
        assert reset_session.load_deferred is False
        assert reset_session.manual_schema is None

        # Verify password authentication state cleared
        assert len(reset_session.dynamic_allowed_directories) == 0
        assert reset_session.pending_auth_path is None
        assert reset_session.password_attempts == 0

        # Verify back button state cleared
        assert reset_session.last_back_action is None
        assert not reset_session.can_go_back()  # State history should be empty

        # Verify prediction workflow data cleared
        assert reset_session.compatible_models is None

    async def test_reset_session_nonexistent(self, manager):
        """Test reset_session on non-existent session does not raise error."""
        # Should not raise error
        await manager.reset_session(999, "nonexistent")

        # Verify session was not created
        session = await manager.get_session(999, "nonexistent")
        assert session is None

    async def test_reset_session_preserves_user_info(self, manager):
        """Test that reset_session preserves user_id and conversation_id."""
        # Create session
        session = await manager.get_or_create_session(123, "conv_1")
        original_created_at = session.created_at

        # Add some data
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        await manager.update_session(session)

        # Reset
        await manager.reset_session(123, "conv_1")

        # Get session
        reset_session = await manager.get_session(123, "conv_1")

        # Verify user info preserved
        assert reset_session.user_id == 123
        assert reset_session.conversation_id == "conv_1"
        assert reset_session.created_at == original_created_at

        # Verify last_activity was updated
        assert reset_session.last_activity > original_created_at
