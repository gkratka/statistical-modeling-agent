"""
TDD Tests for Training State Transitions.

These tests verify that state transitions during ML training workflow follow
the correct sequence and handle errors appropriately.

Background:
After fixing the async event loop blocking issue, a new bug was discovered where
the state remained 'confirming_model' instead of transitioning to 'training_complete'.
This was because the code skipped the intermediate 'TRAINING' state, which violated
state machine transition rules.

Bug Fix: Add state transition to TRAINING before executing training, ensuring the
proper state flow: confirming_model → training → training_complete → naming_model
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, Chat, User, CallbackQuery
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType


class TestTrainingStateTransitions:
    """Test state transitions during ML training workflow."""

    @pytest.fixture
    def mock_query(self):
        """Create mock callback query."""
        query = MagicMock(spec=CallbackQuery)
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.message = MagicMock(spec=Message)
        query.message.reply_text = AsyncMock()
        return query

    @pytest.fixture
    def mock_update(self, mock_query):
        """Create mock Telegram update."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock(spec=User)
        update.effective_user.id = 12345
        update.effective_chat = MagicMock(spec=Chat)
        update.effective_chat.id = 67890
        update.callback_query = mock_query
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock bot context."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot_data = {'state_manager': StateManager()}
        return context

    @pytest.mark.asyncio
    async def test_state_transitions_from_confirming_to_training(self, mock_update, mock_context):
        """
        Test that state transitions from confirming_model to training before training starts.

        This validates the fix - before the fix, training started while still in
        confirming_model state, causing subsequent transitions to fail.
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in CONFIRMING_MODEL state with required prerequisites
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        session.selections['model_type'] = 'keras_binary_classification'  # Required prerequisite
        await state_manager.update_session(session)

        # Execute: Transition to TRAINING state
        success, error_msg, _ = await state_manager.transition_state(
            session,
            MLTrainingState.TRAINING.value
        )

        # Verify: Transition succeeds
        assert success is True
        assert error_msg is None or error_msg == ""
        assert session.current_state == MLTrainingState.TRAINING.value

    @pytest.mark.asyncio
    async def test_state_transitions_from_training_to_complete(self, mock_update, mock_context):
        """
        Test that state transitions from training to training_complete after training finishes.

        This is the second step in the correct state flow.
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in TRAINING state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.TRAINING.value
        await state_manager.update_session(session)

        # Execute: Transition to TRAINING_COMPLETE state
        success, error_msg, _ = await state_manager.transition_state(
            session,
            MLTrainingState.TRAINING_COMPLETE.value
        )

        # Verify: Transition succeeds
        assert success is True
        assert error_msg is None or error_msg == ""
        assert session.current_state == MLTrainingState.TRAINING_COMPLETE.value

    @pytest.mark.asyncio
    async def test_invalid_transition_from_confirming_to_complete_fails(self, mock_update, mock_context):
        """
        Test that invalid transition from confirming_model to training_complete fails.

        This test validates the bug we discovered - skipping the TRAINING state
        causes the transition to fail silently.
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in CONFIRMING_MODEL state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        await state_manager.update_session(session)

        # Execute: Try to transition directly to TRAINING_COMPLETE (invalid)
        success, error_msg, _ = await state_manager.transition_state(
            session,
            MLTrainingState.TRAINING_COMPLETE.value
        )

        # Verify: Transition fails
        assert success is False
        assert "Invalid transition" in error_msg
        assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value  # State unchanged

    @pytest.mark.asyncio
    async def test_complete_state_flow_confirming_to_naming(self, mock_update, mock_context):
        """
        Test the complete state flow for model naming workflow.

        Validates the entire sequence:
        confirming_model → training → training_complete → naming_model
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in CONFIRMING_MODEL state with required prerequisites
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        session.selections['model_type'] = 'keras_binary_classification'  # Required for TRAINING
        await state_manager.update_session(session)

        # Step 1: Transition to TRAINING
        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.TRAINING.value
        )
        assert success is True
        assert session.current_state == MLTrainingState.TRAINING.value

        # Step 2: Transition to TRAINING_COMPLETE
        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success is True
        assert session.current_state == MLTrainingState.TRAINING_COMPLETE.value

        # Step 3: Transition to NAMING_MODEL (requires pending_model_id)
        session.selections['pending_model_id'] = 'model_12345_test'
        await state_manager.update_session(session)

        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.NAMING_MODEL.value
        )
        assert success is True
        assert session.current_state == MLTrainingState.NAMING_MODEL.value

    @pytest.mark.asyncio
    async def test_state_transition_error_handling(self, mock_update, mock_context):
        """
        Test that failed transitions return proper error messages.

        This ensures we can log and handle transition failures appropriately.
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in CONFIRMING_MODEL state
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        await state_manager.update_session(session)

        # Execute: Try invalid transition
        success, error_msg, missing = await state_manager.transition_state(
            session,
            MLTrainingState.TRAINING_COMPLETE.value
        )

        # Verify: Error message is descriptive
        assert success is False
        assert error_msg != ""
        assert "confirming_model" in error_msg.lower()
        assert "training_complete" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_state_persists_after_transition(self, mock_update, mock_context):
        """
        Test that state changes persist after transition.

        Ensures the session is properly updated and saved.
        """
        state_manager = mock_context.bot_data['state_manager']

        # Setup: User in CONFIRMING_MODEL state with prerequisites
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CONFIRMING_MODEL.value
        session.selections['model_type'] = 'keras_binary_classification'  # Required prerequisite
        await state_manager.update_session(session)

        # Execute: Transition to TRAINING
        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.TRAINING.value
        )

        # Verify: State persists when reloading session
        reloaded_session = await state_manager.get_or_create_session(12345, "chat_67890")
        assert reloaded_session.current_state == MLTrainingState.TRAINING.value


class TestStateTransitionValidation:
    """Test state machine validation rules."""

    @pytest.mark.asyncio
    async def test_state_machine_rules_are_enforced(self):
        """
        Test that state machine rules are enforced.

        Validates that only allowed transitions succeed.
        """
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(99999, "test_chat")
        session.workflow_type = WorkflowType.ML_TRAINING

        # Valid transition: TRAINING → TRAINING_COMPLETE
        session.current_state = MLTrainingState.TRAINING.value
        await state_manager.update_session(session)

        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success is True

        # Invalid transition: SELECTING_TARGET → TRAINING_COMPLETE (skip multiple states)
        session.current_state = MLTrainingState.SELECTING_TARGET.value
        await state_manager.update_session(session)

        success, _, _ = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
