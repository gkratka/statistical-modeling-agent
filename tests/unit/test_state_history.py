"""
Unit tests for state history management system.

Tests cover:
- StateSnapshot creation and serialization
- StateHistory push/pop operations
- Depth limiting (circular buffer)
- Session integration (save/restore)
- Field clearing (clean slate requirement)
- Memory optimization validation

Related: dev/implemented/workflow-back-button.md
"""

import pytest
import pandas as pd
import time
from src.core.state_history import (
    StateSnapshot,
    StateHistory,
    get_fields_to_clear,
    CLEANUP_MAP
)
from src.core.state_manager import UserSession, WorkflowType


class TestStateSnapshot:
    """Test StateSnapshot creation and serialization."""

    def test_snapshot_creation_basic(self):
        """Test basic snapshot creation from session."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "SELECTING_TARGET"
        session.workflow_type = WorkflowType.ML_TRAINING
        session.file_path = "/data/housing.csv"

        snapshot = StateSnapshot(session)

        assert snapshot.step == "SELECTING_TARGET"
        assert snapshot.workflow == "ml_training"
        assert snapshot.file_path == "/data/housing.csv"
        assert snapshot.timestamp > 0

    def test_snapshot_with_selections(self):
        """Test snapshot captures selections with deep copy."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.selected_target = "price"
        session.selected_features = ["sqft", "bedrooms"]
        session.selected_model_type = "random_forest"

        snapshot = StateSnapshot(session)

        assert snapshot.selections["selected_target"] == "price"
        assert snapshot.selections["selected_features"] == ["sqft", "bedrooms"]
        assert snapshot.selections["selected_model_type"] == "random_forest"

        # Verify deep copy: modifying session doesn't affect snapshot
        session.selected_target = "rent"
        assert snapshot.selections["selected_target"] == "price"

    def test_snapshot_with_dataframe(self):
        """Test snapshot with DataFrame (shallow copy)."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        session.uploaded_data = df

        snapshot = StateSnapshot(session)

        # Verify shallow copy: same DataFrame reference
        assert snapshot.data_ref is df
        assert snapshot.data_hash == hash(id(df))

    def test_snapshot_serialization(self):
        """Test snapshot to_dict and from_dict."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "AWAITING_FEATURES"
        session.workflow_type = WorkflowType.ML_TRAINING
        session.selected_target = "price"
        session.file_path = "/data/housing.csv"

        snapshot = StateSnapshot(session)
        snapshot_dict = snapshot.to_dict()

        # Verify serialization
        assert snapshot_dict["step"] == "AWAITING_FEATURES"
        assert snapshot_dict["workflow"] == "ml_training"
        assert snapshot_dict["selections"]["selected_target"] == "price"
        assert snapshot_dict["file_path"] == "/data/housing.csv"

        # Verify deserialization
        restored = StateSnapshot.from_dict(snapshot_dict)
        assert restored.step == snapshot.step
        assert restored.workflow == snapshot.workflow
        assert restored.selections == snapshot.selections
        assert restored.file_path == snapshot.file_path

    def test_snapshot_without_data(self):
        """Test snapshot handles None data correctly."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.uploaded_data = None

        snapshot = StateSnapshot(session)

        assert snapshot.data_ref is None
        assert snapshot.data_hash is None


class TestStateHistory:
    """Test StateHistory stack operations."""

    def test_history_initialization(self):
        """Test history initialization with default depth."""
        history = StateHistory()

        assert history.max_depth == 10
        assert history.get_depth() == 0
        assert not history.can_go_back()

    def test_history_push_pop(self):
        """Test basic push and pop operations."""
        history = StateHistory()
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "STATE_1"

        snapshot = StateSnapshot(session)
        history.push(snapshot)

        assert history.get_depth() == 1
        assert history.can_go_back()

        popped = history.pop()
        assert popped.step == "STATE_1"
        assert history.get_depth() == 0
        assert not history.can_go_back()

    def test_history_peek(self):
        """Test peek without removing snapshot."""
        history = StateHistory()
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "STATE_1"

        snapshot = StateSnapshot(session)
        history.push(snapshot)

        peeked = history.peek()
        assert peeked.step == "STATE_1"
        assert history.get_depth() == 1  # Not removed

    def test_history_depth_limit(self):
        """Test circular buffer with max depth limit."""
        history = StateHistory(max_depth=3)

        # Push 5 snapshots (exceeds max depth)
        for i in range(5):
            session = UserSession(user_id=12345, conversation_id="conv_1")
            session.current_state = f"STATE_{i}"
            history.push(StateSnapshot(session))

        # Should only retain last 3
        assert history.get_depth() == 3

        # Verify FIFO eviction: oldest snapshots removed
        popped_3 = history.pop()
        assert popped_3.step == "STATE_4"
        popped_2 = history.pop()
        assert popped_2.step == "STATE_3"
        popped_1 = history.pop()
        assert popped_1.step == "STATE_2"
        assert history.get_depth() == 0

    def test_history_clear(self):
        """Test clearing all history."""
        history = StateHistory()

        # Add 3 snapshots
        for i in range(3):
            session = UserSession(user_id=12345, conversation_id="conv_1")
            session.current_state = f"STATE_{i}"
            history.push(StateSnapshot(session))

        assert history.get_depth() == 3

        history.clear()

        assert history.get_depth() == 0
        assert not history.can_go_back()
        assert history.pop() is None

    def test_history_pop_empty(self):
        """Test pop on empty history returns None."""
        history = StateHistory()

        assert history.pop() is None
        assert history.peek() is None

    def test_history_serialization(self):
        """Test history to_dict and from_dict."""
        history = StateHistory(max_depth=5)

        # Add 2 snapshots
        for i in range(2):
            session = UserSession(user_id=12345, conversation_id="conv_1")
            session.current_state = f"STATE_{i}"
            session.selected_target = f"target_{i}"
            history.push(StateSnapshot(session))

        # Serialize
        history_dict = history.to_dict()
        assert history_dict["max_depth"] == 5
        assert len(history_dict["history"]) == 2

        # Deserialize
        restored = StateHistory.from_dict(history_dict)
        assert restored.max_depth == 5
        assert restored.get_depth() == 2

        # Verify snapshots preserved
        popped = restored.pop()
        assert popped.step == "STATE_1"
        assert popped.selections["selected_target"] == "target_1"


class TestSessionIntegration:
    """Test state history integration with UserSession."""

    def test_session_save_snapshot(self):
        """Test session saves snapshot correctly."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "SELECTING_TARGET"
        session.workflow_type = WorkflowType.ML_TRAINING
        session.selected_target = "price"

        session.save_state_snapshot()

        assert session.state_history.get_depth() == 1
        assert session.can_go_back()

    def test_session_restore_previous_state(self):
        """Test session restores previous state correctly."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "SELECTING_TARGET"
        session.workflow_type = WorkflowType.ML_TRAINING
        session.selected_target = "price"

        # Save snapshot
        session.save_state_snapshot()

        # Transition to new state
        session.current_state = "SELECTING_FEATURES"
        session.selected_features = ["sqft", "bedrooms"]

        # Restore previous state
        assert session.restore_previous_state()

        assert session.current_state == "SELECTING_TARGET"
        assert session.workflow_type == WorkflowType.ML_TRAINING
        assert session.selected_target == "price"
        assert session.state_history.get_depth() == 0

    def test_session_restore_clears_downstream_fields(self):
        """Test restore clears fields set after restored state."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "AWAITING_TARGET_SELECTION"
        session.workflow_type = WorkflowType.ML_TRAINING

        # Save snapshot at target selection
        session.save_state_snapshot()

        # Progress to model selection
        session.current_state = "AWAITING_MODEL_TYPE_SELECTION"
        session.selected_target = "price"
        session.selected_features = ["sqft", "bedrooms"]
        session.selected_model_type = "random_forest"

        # Go back to target selection
        session.restore_previous_state()

        # Verify state restored
        assert session.current_state == "AWAITING_TARGET_SELECTION"

        # Verify downstream fields cleared
        assert session.selected_target is None  # Should be cleared by cleanup map

    def test_session_clear_fields(self):
        """Test session clears specified fields."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.selected_target = "price"
        session.selected_features = ["sqft", "bedrooms"]
        session.selected_model_type = "random_forest"

        session.clear_fields(["selected_target", "selected_features"])

        assert session.selected_target is None
        assert session.selected_features is None
        assert session.selected_model_type == "random_forest"  # Not cleared

    def test_session_multi_level_back(self):
        """Test going back multiple levels."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.current_state = "CHOOSING_DATA_SOURCE"
        session.workflow_type = WorkflowType.ML_TRAINING

        # Save snapshots at each step
        session.save_state_snapshot()
        session.current_state = "AWAITING_FILE_PATH"

        session.save_state_snapshot()
        session.current_state = "CONFIRMING_SCHEMA"

        session.save_state_snapshot()
        session.current_state = "AWAITING_TARGET_SELECTION"

        # Go back 3 levels
        assert session.restore_previous_state()  # Back to CONFIRMING_SCHEMA
        assert session.current_state == "CONFIRMING_SCHEMA"

        assert session.restore_previous_state()  # Back to AWAITING_FILE_PATH
        assert session.current_state == "AWAITING_FILE_PATH"

        assert session.restore_previous_state()  # Back to CHOOSING_DATA_SOURCE
        assert session.current_state == "CHOOSING_DATA_SOURCE"

        # No more history
        assert not session.restore_previous_state()

    def test_session_restore_empty_history(self):
        """Test restore on empty history returns False."""
        session = UserSession(user_id=12345, conversation_id="conv_1")

        assert not session.restore_previous_state()
        assert session.state_history.get_depth() == 0

    def test_session_clear_history(self):
        """Test session clears all history."""
        session = UserSession(user_id=12345, conversation_id="conv_1")

        # Create 3 snapshots
        for i in range(3):
            session.current_state = f"STATE_{i}"
            session.save_state_snapshot()

        assert session.state_history.get_depth() == 3

        session.clear_history()

        assert session.state_history.get_depth() == 0
        assert not session.can_go_back()


class TestFieldClearing:
    """Test field clearing logic (clean slate requirement)."""

    def test_cleanup_map_all_states(self):
        """Test cleanup map defined for all eligible states."""
        expected_states = [
            "CHOOSING_DATA_SOURCE",
            "AWAITING_FILE_PATH",
            "AWAITING_FILE_UPLOAD",
            "FILE_PATH_RECEIVED",
            "FILE_UPLOADED",
            "CONFIRMING_SCHEMA",
            "AWAITING_TARGET_SELECTION",
            "AWAITING_FEATURE_SELECTION",
            "AWAITING_MODEL_TYPE_SELECTION",
            "DEFERRED_SCHEMA_PENDING",
        ]

        for state in expected_states:
            fields = get_fields_to_clear(state)
            assert isinstance(fields, list), f"Cleanup map missing for {state}"

    def test_get_fields_to_clear_target_state(self):
        """Test fields to clear for target selection state."""
        fields = get_fields_to_clear("AWAITING_TARGET_SELECTION")

        assert "selected_target" in fields

    def test_get_fields_to_clear_data_source_state(self):
        """Test fields to clear for data source state (clears everything)."""
        fields = get_fields_to_clear("CHOOSING_DATA_SOURCE")

        assert "file_path" in fields
        assert "data" in fields
        assert "detected_schema" in fields
        assert "selected_target" in fields
        assert "selected_features" in fields
        assert "selected_model_type" in fields

    def test_get_fields_to_clear_unknown_state(self):
        """Test get_fields_to_clear returns empty list for unknown state."""
        fields = get_fields_to_clear("UNKNOWN_STATE")

        assert fields == []


class TestMemoryOptimization:
    """Test memory optimization (shallow vs deep copy)."""

    def test_dataframe_shallow_copy(self):
        """Test DataFrame uses shallow copy (reference only)."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        df = pd.DataFrame({"col1": [1, 2, 3] * 100, "col2": [4, 5, 6] * 100})
        session.uploaded_data = df

        # Create 10 snapshots
        snapshots = []
        for i in range(10):
            snapshot = StateSnapshot(session)
            snapshots.append(snapshot)

        # Verify all snapshots reference same DataFrame
        for snapshot in snapshots:
            assert snapshot.data_ref is df
            assert id(snapshot.data_ref) == id(df)

    def test_selections_deep_copy(self):
        """Test selections use deep copy (independent)."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.selected_features = ["sqft", "bedrooms"]

        snapshot = StateSnapshot(session)

        # Modify session's selections
        session.selected_features.append("bathrooms")

        # Snapshot should not be affected (deep copy)
        assert snapshot.selections["selected_features"] == ["sqft", "bedrooms"]
        assert "bathrooms" not in snapshot.selections["selected_features"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_snapshot_with_none_selections(self):
        """Test snapshot handles None selections gracefully."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        session.selected_target = None
        session.selected_features = None
        session.selected_model_type = None

        snapshot = StateSnapshot(session)

        assert snapshot.selections["selected_target"] is None
        assert snapshot.selections["selected_features"] is None
        assert snapshot.selections["selected_model_type"] is None

    def test_restore_with_dataframe_mutation(self):
        """Test restore handles DataFrame mutations correctly."""
        session = UserSession(user_id=12345, conversation_id="conv_1")
        df = pd.DataFrame({"col1": [1, 2, 3]})
        session.uploaded_data = df

        snapshot = StateSnapshot(session)
        original_hash = snapshot.data_hash

        # Mutate DataFrame
        session.uploaded_data.loc[0, "col1"] = 999

        # Hash should detect mutation
        new_hash = hash(id(session.uploaded_data))
        assert new_hash == original_hash  # Same object ID

    def test_history_with_max_depth_1(self):
        """Test history with minimal depth limit."""
        history = StateHistory(max_depth=1)

        session1 = UserSession(user_id=1, conversation_id="1")
        session1.current_state = "STATE_1"
        history.push(StateSnapshot(session1))

        session2 = UserSession(user_id=2, conversation_id="2")
        session2.current_state = "STATE_2"
        history.push(StateSnapshot(session2))

        # Should only retain last snapshot
        assert history.get_depth() == 1
        popped = history.pop()
        assert popped.step == "STATE_2"
