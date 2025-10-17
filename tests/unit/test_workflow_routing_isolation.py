"""
TDD Tests for Workflow Routing Isolation.

These tests verify that WorkflowRouter only processes ML_TRAINING workflows
and does not interfere with ML_PREDICTION workflows.

Background:
The bug occurs when WorkflowRouter (which handles training states) receives
messages from prediction workflows. It treats prediction states as "UNKNOWN"
and clears the workflow, breaking the user's prediction session.

Fix: Add workflow type filtering to WorkflowRouter.handle() to skip non-training workflows.
"""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from src.bot.workflow_handlers import WorkflowRouter
from src.core.state_manager import (
    StateManager,
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)
import pandas as pd


@pytest.fixture
def state_manager():
    """Create StateManager instance."""
    return StateManager()


@pytest.fixture
def workflow_router(state_manager):
    """Create WorkflowRouter instance."""
    return WorkflowRouter(state_manager)


@pytest.fixture
def mock_update():
    """Create mock Telegram update."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 67890
    update.message = AsyncMock(spec=Message)
    update.message.text = "Attribute1,Attribute2,Attribute3"
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


class TestWorkflowRoutingIsolation:
    """Test that WorkflowRouter only processes ML_TRAINING workflows."""

    @pytest.mark.asyncio
    async def test_workflow_router_skips_prediction_workflow(
        self,
        workflow_router,
        state_manager,
        mock_update,
        mock_context
    ):
        """
        Test that WorkflowRouter returns early for PREDICTION workflow.

        Given: User is in ML_PREDICTION workflow with AWAITING_FEATURE_SELECTION state
        When: WorkflowRouter.handle() is called
        Then: Should return early without processing (no error, no state change)
        """
        # Setup session in PREDICTION workflow
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
        session.uploaded_data = pd.DataFrame({
            'Attribute1': [1, 2, 3],
            'Attribute2': [4, 5, 6]
        })

        # Call handler - should return early
        await workflow_router.handle(mock_update, mock_context, session)

        # Should NOT have cleared workflow
        assert session.workflow_type == WorkflowType.ML_PREDICTION
        assert session.current_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value

        # Should NOT have shown error message
        mock_update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_router_processes_training_workflow(
        self,
        workflow_router,
        state_manager,
        mock_update,
        mock_context
    ):
        """
        Test that WorkflowRouter DOES process ML_TRAINING workflow.

        Given: User is in ML_TRAINING workflow with SELECTING_TARGET state
        When: WorkflowRouter.handle() is called
        Then: Should process the message (call handler method)
        """
        # Setup session in TRAINING workflow
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.SELECTING_TARGET.value

        # Mock uploaded data
        session.uploaded_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'target': [0, 1, 0]
        })

        mock_update.message.text = "3"  # Select column 3 as target
        mock_update.message.reply_text = AsyncMock()

        # Call handler - should process
        await workflow_router.handle(mock_update, mock_context, session)

        # Should have processed message (replied with feature selection prompt)
        assert mock_update.message.reply_text.call_count > 0

    @pytest.mark.asyncio
    async def test_workflow_router_skips_none_workflow(
        self,
        workflow_router,
        state_manager,
        mock_update,
        mock_context
    ):
        """
        Test that WorkflowRouter skips when no workflow is active.

        Given: Session has no active workflow (workflow_type = None)
        When: WorkflowRouter.handle() is called
        Then: Should return early
        """
        # Setup session with NO workflow
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = None
        session.current_state = None

        # Call handler - should return early
        await workflow_router.handle(mock_update, mock_context, session)

        # Should NOT have shown error
        mock_update.message.reply_text.assert_not_called()


class TestWorkflowRouterLogging:
    """Test that WorkflowRouter logs skip decisions for debugging."""

    @pytest.mark.asyncio
    async def test_workflow_router_logs_skip_reason(
        self,
        workflow_router,
        state_manager,
        mock_update,
        mock_context,
        caplog
    ):
        """
        Test that WorkflowRouter logs why it skipped processing.

        Given: User is in ML_PREDICTION workflow
        When: WorkflowRouter.handle() is called
        Then: Should log skip reason at debug level
        """
        caplog.set_level(logging.DEBUG)

        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value

        await workflow_router.handle(mock_update, mock_context, session)

        # Check log contains skip message
        assert any(
            "Skipping WorkflowRouter" in record.message or
            "not ML_TRAINING" in record.message
            for record in caplog.records
        ), f"Expected skip log message, got: {[r.message for r in caplog.records]}"


class TestBugRegression:
    """Regression test for the specific bug reported by user."""

    @pytest.mark.asyncio
    async def test_prediction_feature_selection_no_unknown_state_error(
        self,
        state_manager
    ):
        """
        REGRESSION TEST: "awaiting_feature_selection treated as UNKNOWN STATE"

        This test reproduces the exact scenario from the user's bug report:
        1. User runs /predict and completes local path data loading
        2. Bot transitions to AWAITING_FEATURE_SELECTION
        3. User types features: "Attribute1,Attribute2,...,Attribute20"
        4. BUG: WorkflowRouter treated state as UNKNOWN and cleared workflow

        Expected After Fix:
        - WorkflowRouter should skip prediction workflow
        - State should remain intact (not cleared)
        - No "Workflow state error" message
        """
        workflow_router = WorkflowRouter(state_manager)

        # Step 1-2: Setup session (after data loading completed)
        session = await state_manager.get_or_create_session(7715560927, "chat_12345")
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value

        # Mock loaded data (20 attributes as in user's screenshot)
        session.uploaded_data = pd.DataFrame({
            f'Attribute{i}': range(1, 201) for i in range(1, 21)
        })

        # Step 3: User types features (all 20 comma-separated)
        mock_update = MagicMock(spec=Update)
        mock_update.message.text = ",".join([f"Attribute{i}" for i in range(1, 21)])
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # BEFORE FIX: This would log "UNKNOWN STATE" and clear workflow
        # AFTER FIX: Should return early, leaving workflow intact
        await workflow_router.handle(mock_update, mock_context, session)

        # Verify 1: Workflow NOT cleared (state remains)
        assert session.workflow_type == WorkflowType.ML_PREDICTION, \
            "BUG: Workflow was cleared when it should have been preserved"

        assert session.current_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value, \
            f"BUG: State changed from AWAITING_FEATURE_SELECTION to {session.current_state}"

        # Verify 2: No error message sent to user
        error_messages = [
            call[0][0] for call in mock_update.message.reply_text.call_args_list
            if "Workflow state error" in call[0][0] or "UNKNOWN" in call[0][0].upper()
        ]
        assert len(error_messages) == 0, \
            f"BUG: Error message shown when it shouldn't: {error_messages}"

    @pytest.mark.asyncio
    async def test_training_workflow_still_works_after_fix(
        self,
        state_manager
    ):
        """
        Ensure fix doesn't break existing training workflow functionality.

        Given: User is in ML_TRAINING workflow
        When: WorkflowRouter processes training state
        Then: Should continue working as before
        """
        workflow_router = WorkflowRouter(state_manager)

        session = await state_manager.get_or_create_session(12345, "chat_67890")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.SELECTING_TARGET.value
        session.uploaded_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })

        mock_update = MagicMock(spec=Update)
        mock_update.message.text = "3"  # Select column 3
        mock_update.message.reply_text = AsyncMock()
        mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

        # Should process (not skip)
        await workflow_router.handle(mock_update, mock_context, session)

        # Should have sent response (feature selection prompt)
        assert mock_update.message.reply_text.call_count > 0, \
            "Training workflow should still be processed"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
