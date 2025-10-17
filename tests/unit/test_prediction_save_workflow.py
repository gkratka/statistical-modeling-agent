"""Unit tests for prediction save workflow state transitions."""

import pytest
from src.core.state_manager import (
    StateManager,
    UserSession,
    MLPredictionState,
    WorkflowType,
    StateMachine
)


@pytest.mark.asyncio
class TestPredictionSaveStateTransitions:
    """Test state machine transitions for local file save workflow."""

    @pytest.fixture
    def state_manager(self):
        """Create state manager instance."""
        return StateManager()

    async def test_complete_to_awaiting_save_path(self, state_manager):
        """Valid transition from COMPLETE to AWAITING_SAVE_PATH."""
        # Create session in COMPLETE state
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="chat_123"
        )
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        success, error_msg, missing = await state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )

        assert success is True
        assert error_msg is None
        assert missing == []
        assert session.current_state == MLPredictionState.AWAITING_SAVE_PATH.value

    async def test_awaiting_save_path_to_confirming_filename(self, state_manager):
        """Valid transition after directory path validation."""
        # Setup: create session and transition to AWAITING_SAVE_PATH
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="chat_124"
        )
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        await state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )

        # Test transition to CONFIRMING_SAVE_FILENAME
        success, error_msg, missing = await state_manager.transition_state(
            session,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value
        )

        assert success is True
        assert error_msg is None
        assert missing == []
        assert session.current_state == MLPredictionState.CONFIRMING_SAVE_FILENAME.value

    async def test_confirming_filename_to_complete(self, state_manager):
        """Valid transition after successful file save."""
        # Setup: transition to CONFIRMING_SAVE_FILENAME
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="chat_125"
        )
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        await state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )
        await state_manager.transition_state(
            session,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value
        )

        # Test transition back to COMPLETE
        success, error_msg, missing = await state_manager.transition_state(
            session,
            MLPredictionState.COMPLETE.value
        )

        assert success is True
        assert error_msg is None
        assert missing == []
        assert session.current_state == MLPredictionState.COMPLETE.value

    async def test_invalid_transition_skip_awaiting_save_path(self, state_manager):
        """Cannot skip directly from COMPLETE to CONFIRMING_SAVE_FILENAME."""
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="chat_126"
        )
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        success, error_msg, missing = await state_manager.transition_state(
            session,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value
        )

        assert success is False
        assert error_msg is not None
        assert "Invalid transition" in error_msg
        assert session.current_state == MLPredictionState.COMPLETE.value

    async def test_get_valid_next_states_from_complete(self):
        """Check valid next states from COMPLETE include AWAITING_SAVE_PATH."""
        valid_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_PREDICTION,
            MLPredictionState.COMPLETE.value
        )

        assert MLPredictionState.AWAITING_SAVE_PATH.value in valid_states

    async def test_get_valid_next_states_from_awaiting_save_path(self):
        """Check valid next states from AWAITING_SAVE_PATH."""
        valid_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_PREDICTION,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )

        assert MLPredictionState.CONFIRMING_SAVE_FILENAME.value in valid_states

    async def test_get_valid_next_states_from_confirming_filename(self):
        """Check valid next states from CONFIRMING_SAVE_FILENAME."""
        valid_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_PREDICTION,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value
        )

        assert MLPredictionState.COMPLETE.value in valid_states

    async def test_circular_transition_complete_to_save_to_complete(self, state_manager):
        """Test full circular workflow: COMPLETE -> AWAITING_SAVE_PATH -> CONFIRMING_SAVE_FILENAME -> COMPLETE."""
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="chat_127"
        )
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        # Start at COMPLETE
        assert session.current_state == MLPredictionState.COMPLETE.value

        # Transition 1: COMPLETE -> AWAITING_SAVE_PATH
        success, _, _ = await state_manager.transition_state(
            session,
            MLPredictionState.AWAITING_SAVE_PATH.value
        )
        assert success is True

        # Transition 2: AWAITING_SAVE_PATH -> CONFIRMING_SAVE_FILENAME
        success, _, _ = await state_manager.transition_state(
            session,
            MLPredictionState.CONFIRMING_SAVE_FILENAME.value
        )
        assert success is True

        # Transition 3: CONFIRMING_SAVE_FILENAME -> COMPLETE
        success, _, _ = await state_manager.transition_state(
            session,
            MLPredictionState.COMPLETE.value
        )
        assert success is True
        assert session.current_state == MLPredictionState.COMPLETE.value
