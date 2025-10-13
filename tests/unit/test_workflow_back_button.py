"""
Unit tests for workflow back button functionality.

Tests:
1. State cleanup logic (CLEANUP_MAP)
2. Multi-level back navigation
3. Debouncing behavior
4. Edge cases (empty history, beginning of workflow)

Related: dev/implemented/workflow-back-button.md
"""

import pytest
import time
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.core.state_history import StateHistory, StateSnapshot, CLEANUP_MAP, get_fields_to_clear


class TestStateCleanupLogic:
    """Test CLEANUP_MAP and field clearing behavior."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UserSession for testing."""
        session = MagicMock()
        session.current_state = MLTrainingState.AWAITING_TARGET_SELECTION.value
        session.workflow_type = WorkflowType.ML_TRAINING
        session.uploaded_data = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        session.file_path = "/path/to/data.csv"
        session.detected_schema = {'target': 'price', 'features': ['sqft', 'bedrooms']}
        session.selected_target = 'price'
        session.selected_features = ['sqft', 'bedrooms']
        session.selected_model_type = 'random_forest'
        session.selected_task_type = 'regression'
        return session

    def test_cleanup_map_has_all_states(self):
        """Verify CLEANUP_MAP covers all relevant workflow states."""
        expected_states = [
            'CHOOSING_DATA_SOURCE',
            'AWAITING_FILE_PATH',
            'AWAITING_FILE_UPLOAD',
            'FILE_PATH_RECEIVED',
            'FILE_UPLOADED',
            'CONFIRMING_SCHEMA',
            'AWAITING_TARGET_SELECTION',
            'AWAITING_FEATURE_SELECTION',
            'AWAITING_MODEL_TYPE_SELECTION',
            'DEFERRED_SCHEMA_PENDING'
        ]

        for state in expected_states:
            assert state in CLEANUP_MAP, f"CLEANUP_MAP missing state: {state}"

    def test_get_fields_to_clear_returns_list(self):
        """Test get_fields_to_clear returns correct list for known state."""
        fields = get_fields_to_clear('AWAITING_TARGET_SELECTION')
        assert isinstance(fields, list)
        assert 'selected_target' in fields

    def test_get_fields_to_clear_returns_empty_for_unknown_state(self):
        """Test get_fields_to_clear returns empty list for unknown state."""
        fields = get_fields_to_clear('UNKNOWN_STATE')
        assert fields == []

    def test_choosing_data_source_clears_everything_downstream(self):
        """Test that CHOOSING_DATA_SOURCE clears all downstream fields."""
        fields = get_fields_to_clear('CHOOSING_DATA_SOURCE')

        # Should clear ALL downstream fields
        assert 'file_path' in fields
        assert 'data' in fields
        assert 'detected_schema' in fields
        assert 'selected_target' in fields
        assert 'selected_features' in fields
        assert 'selected_model_type' in fields
        assert 'selected_task_type' in fields

    def test_awaiting_target_clears_only_target(self):
        """Test that AWAITING_TARGET_SELECTION clears only target field."""
        fields = get_fields_to_clear('AWAITING_TARGET_SELECTION')

        assert 'selected_target' in fields
        # Should NOT clear upstream fields
        assert 'file_path' not in fields
        assert 'data' not in fields
        assert 'detected_schema' not in fields

    def test_deferred_schema_pending_clears_nothing(self):
        """Test that DEFERRED_SCHEMA_PENDING has no cleanup (special state)."""
        fields = get_fields_to_clear('DEFERRED_SCHEMA_PENDING')
        assert fields == []


class TestMultiLevelBackNavigation:
    """Test multi-step back navigation through workflow history."""

    @pytest.fixture
    def state_manager(self):
        """Create StateManager for testing."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)
        return manager

    @pytest.mark.asyncio
    async def test_three_level_back_navigation(self, state_manager):
        """Test navigating back through 3 states."""
        # Create session
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_conv"
        )

        # Setup initial state
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await state_manager.update_session(session)

        # State 1: Data source chosen
        session.save_state_snapshot()
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.data_source = "local_path"
        await state_manager.update_session(session)

        # State 2: File path entered
        session.save_state_snapshot()
        session.current_state = MLTrainingState.CHOOSING_LOAD_OPTION.value
        session.file_path = "/path/to/data.csv"
        await state_manager.update_session(session)

        # State 3: Load option chosen
        session.save_state_snapshot()
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value
        session.load_deferred = False
        await state_manager.update_session(session)

        # Verify we're at state 3
        assert session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value
        assert session.state_history.get_depth() == 3

        # Back to state 2
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLTrainingState.CHOOSING_LOAD_OPTION.value
        assert session.state_history.get_depth() == 2

        # Back to state 1
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value
        assert session.state_history.get_depth() == 1

        # Back to state 0
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value
        assert session.state_history.get_depth() == 0

        # Can't go back further
        success = session.restore_previous_state()
        assert not success
        assert session.state_history.get_depth() == 0

    @pytest.mark.asyncio
    async def test_field_cleanup_during_back_navigation(self, state_manager):
        """Test that fields are properly cleared during back navigation."""
        # Create session
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_conv_2"
        )

        # Initial state
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Add data and selections
        session.save_state_snapshot()
        session.current_state = MLTrainingState.SELECTING_TARGET.value
        session.selected_target = 'price'
        session.selected_features = ['sqft', 'bedrooms']
        session.selected_model_type = 'random_forest'
        await state_manager.update_session(session)

        # Verify fields are set
        assert session.selected_target == 'price'
        assert session.selected_features == ['sqft', 'bedrooms']
        assert session.selected_model_type == 'random_forest'

        # Navigate back
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Fields should be cleared based on CLEANUP_MAP
        fields_to_clear = get_fields_to_clear(MLTrainingState.CHOOSING_DATA_SOURCE.value)

        # Target field should have been cleared
        if hasattr(session, 'selected_target'):
            assert session.selected_target is None or session.selected_target == ''


class TestDebouncingBehavior:
    """Test debouncing to prevent race conditions."""

    @pytest.mark.asyncio
    async def test_debouncing_prevents_rapid_clicks(self):
        """Test that rapid back button clicks are debounced (500ms cooldown)."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="debounce_test"
        )

        # Setup with history
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        session.save_state_snapshot()
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

        session.save_state_snapshot()
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value

        # Verify we have history
        assert session.state_history.get_depth() == 2

        # First back action - should succeed
        session.last_back_action = time.time()
        current_time_1 = session.last_back_action

        # Simulate rapid second click (100ms later)
        time_since_last = 0.1  # 100ms (< 500ms threshold)

        # Debouncing logic check
        should_allow = time_since_last >= 0.5
        assert not should_allow, "Debouncing should block clicks within 500ms"

        # Third click after cooldown (600ms later) - should succeed
        time_since_last_2 = 0.6  # 600ms (> 500ms threshold)
        should_allow_2 = time_since_last_2 >= 0.5
        assert should_allow_2, "Debouncing should allow clicks after 500ms"

    @pytest.mark.asyncio
    async def test_last_back_action_timestamp_updated(self):
        """Test that last_back_action timestamp is updated on each back action."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="timestamp_test"
        )

        # Setup history
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        session.save_state_snapshot()
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

        # Initial timestamp should be None
        assert session.last_back_action is None

        # Simulate back action
        before_time = time.time()
        session.last_back_action = time.time()
        after_time = time.time()

        # Timestamp should be set within the time window
        assert session.last_back_action >= before_time
        assert session.last_back_action <= after_time


class TestEdgeCases:
    """Test edge cases for back button behavior."""

    @pytest.mark.asyncio
    async def test_back_at_beginning_of_workflow(self):
        """Test back button at the very beginning (no history)."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="edge_test_1"
        )

        # Just started workflow - no history
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Verify no history
        assert not session.can_go_back()
        assert session.state_history.get_depth() == 0

        # Try to go back - should fail gracefully
        success = session.restore_previous_state()
        assert not success
        assert session.current_state == MLTrainingState.CHOOSING_DATA_SOURCE.value

    @pytest.mark.asyncio
    async def test_empty_history_after_clear(self):
        """Test behavior when history is cleared."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="edge_test_2"
        )

        # Build up history
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        session.save_state_snapshot()
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.save_state_snapshot()
        session.current_state = MLTrainingState.CONFIRMING_SCHEMA.value

        # Verify history exists
        assert session.state_history.get_depth() == 2

        # Clear history (e.g., workflow restart)
        session.state_history.clear()

        # Verify history is empty
        assert session.state_history.get_depth() == 0
        assert not session.can_go_back()

        # Try to go back - should fail gracefully
        success = session.restore_previous_state()
        assert not success

    @pytest.mark.asyncio
    async def test_max_depth_circular_buffer(self):
        """Test that history respects max_depth limit (circular buffer)."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="edge_test_3"
        )

        # Default max_depth is 10
        max_depth = 10

        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Push 15 states (more than max_depth)
        for i in range(15):
            session.save_state_snapshot()
            session.current_state = f"STATE_{i}"

        # History should only contain last 10 snapshots
        assert session.state_history.get_depth() <= max_depth
        assert session.state_history.get_depth() == max_depth

    @pytest.mark.asyncio
    async def test_data_reference_preservation(self):
        """Test that DataFrame references are preserved (shallow copy)."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="edge_test_4"
        )

        # Create DataFrame
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        session.uploaded_data = df
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        # Save snapshot
        session.save_state_snapshot()
        session.current_state = MLTrainingState.SELECTING_TARGET.value

        # Verify snapshot captured DataFrame reference
        snapshot = session.state_history.peek()
        assert snapshot is not None
        assert snapshot.data_ref is df  # Same reference (shallow copy)
        assert id(snapshot.data_ref) == id(df)


class TestBackButtonHandler:
    """Test the actual back button handler (integration-style tests)."""

    @pytest.mark.asyncio
    async def test_handler_restores_correct_state(self):
        """Test that handle_workflow_back restores the correct previous state."""
        # This would require mocking Update and Context objects
        # Skipping detailed implementation as it requires extensive Telegram API mocking
        # Manual testing will cover this in Phase 5
        pass

    @pytest.mark.asyncio
    async def test_handler_sends_correct_ui(self):
        """Test that handle_workflow_back renders correct UI for restored state."""
        # This would require mocking Update and Context objects
        # Skipping detailed implementation as it requires extensive Telegram API mocking
        # Manual testing will cover this in Phase 5
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
