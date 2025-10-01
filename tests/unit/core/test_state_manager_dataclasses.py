"""
Unit tests for state manager dataclasses.

Tests UserSession, StateManagerConfig, and enum types.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.core.state_manager import (
    UserSession,
    StateManagerConfig,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)


class TestWorkflowType:
    """Test WorkflowType enum."""

    def test_workflow_types_exist(self):
        """Test all workflow types are defined."""
        assert WorkflowType.ML_TRAINING.value == "ml_training"
        assert WorkflowType.ML_PREDICTION.value == "ml_prediction"
        assert WorkflowType.STATS_ANALYSIS.value == "stats_analysis"
        assert WorkflowType.DATA_EXPLORATION.value == "data_exploration"

    def test_workflow_type_count(self):
        """Test expected number of workflow types."""
        assert len(WorkflowType) == 4


class TestMLTrainingState:
    """Test MLTrainingState enum."""

    def test_ml_training_states_exist(self):
        """Test all ML training states are defined."""
        assert MLTrainingState.AWAITING_DATA.value == "awaiting_data"
        assert MLTrainingState.SELECTING_TARGET.value == "selecting_target"
        assert MLTrainingState.SELECTING_FEATURES.value == "selecting_features"
        assert MLTrainingState.CONFIRMING_MODEL.value == "confirming_model"
        assert MLTrainingState.TRAINING.value == "training"
        assert MLTrainingState.COMPLETE.value == "complete"

    def test_ml_training_state_count(self):
        """Test expected number of ML training states."""
        assert len(MLTrainingState) == 6


class TestMLPredictionState:
    """Test MLPredictionState enum."""

    def test_ml_prediction_states_exist(self):
        """Test all ML prediction states are defined."""
        assert MLPredictionState.AWAITING_MODEL.value == "awaiting_model"
        assert MLPredictionState.AWAITING_DATA.value == "awaiting_data"
        assert MLPredictionState.PREDICTING.value == "predicting"
        assert MLPredictionState.COMPLETE.value == "complete"

    def test_ml_prediction_state_count(self):
        """Test expected number of ML prediction states."""
        assert len(MLPredictionState) == 4


class TestUserSession:
    """Test UserSession dataclass."""

    def test_session_creation_minimal(self):
        """Test creating session with only required fields."""
        session = UserSession(user_id=12345, conversation_id="conv_123")

        assert session.user_id == 12345
        assert session.conversation_id == "conv_123"
        assert session.workflow_type is None
        assert session.current_state is None
        assert session.uploaded_data is None
        assert session.selections == {}
        assert session.model_ids == []
        assert session.history == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)

    def test_session_creation_full(self):
        """Test creating session with all fields."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        session = UserSession(
            user_id=12345,
            conversation_id="conv_123",
            workflow_type=WorkflowType.ML_TRAINING,
            current_state="selecting_target",
            uploaded_data=df,
            selections={"target": "y"},
            model_ids=["model_1"],
            history=[{"role": "user", "message": "test"}]
        )

        assert session.user_id == 12345
        assert session.workflow_type == WorkflowType.ML_TRAINING
        assert session.current_state == "selecting_target"
        assert session.uploaded_data is df
        assert session.selections == {"target": "y"}
        assert session.model_ids == ["model_1"]
        assert len(session.history) == 1

    def test_session_key_generation(self):
        """Test session key is correctly formatted."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        assert session.session_key == "12345_conv_123"

    def test_validation_invalid_user_id(self):
        """Test validation fails for invalid user_id."""
        with pytest.raises(ValueError, match="user_id must be positive"):
            UserSession(user_id=0, conversation_id="conv_123")

        with pytest.raises(ValueError, match="user_id must be positive"):
            UserSession(user_id=-1, conversation_id="conv_123")

    def test_validation_empty_conversation_id(self):
        """Test validation fails for empty conversation_id."""
        with pytest.raises(ValueError, match="conversation_id cannot be empty"):
            UserSession(user_id=12345, conversation_id="")

    def test_is_expired_not_expired(self):
        """Test session is not expired when within timeout."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        assert not session.is_expired(timeout_minutes=30)

    def test_is_expired_expired(self):
        """Test session is expired when past timeout."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        # Set last activity to 31 minutes ago
        session.last_activity = datetime.now() - timedelta(minutes=31)
        assert session.is_expired(timeout_minutes=30)

    def test_is_expired_boundary(self):
        """Test session expiration at exact timeout boundary."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        # Set last activity to exactly 30 minutes ago
        session.last_activity = datetime.now() - timedelta(minutes=30)
        # Should be expired (delta > timeout_minutes)
        assert session.is_expired(timeout_minutes=30)

    def test_time_until_timeout(self):
        """Test time until timeout calculation."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        # Set last activity to 10 minutes ago
        session.last_activity = datetime.now() - timedelta(minutes=10)

        time_left = session.time_until_timeout(timeout_minutes=30)
        # Should be approximately 20 minutes (allow small margin)
        assert 19.9 < time_left < 20.1

    def test_time_until_timeout_expired(self):
        """Test time until timeout returns 0 when expired."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        session.last_activity = datetime.now() - timedelta(minutes=35)

        time_left = session.time_until_timeout(timeout_minutes=30)
        assert time_left == 0

    def test_update_activity(self):
        """Test update_activity updates timestamp."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        old_activity = session.last_activity

        # Wait a tiny bit and update
        import time
        time.sleep(0.01)
        session.update_activity()

        assert session.last_activity > old_activity

    def test_add_to_history_single_message(self):
        """Test adding single message to history."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        session.add_to_history(role="user", message="Hello")

        assert len(session.history) == 1
        assert session.history[0]["role"] == "user"
        assert session.history[0]["message"] == "Hello"
        assert "timestamp" in session.history[0]

    def test_add_to_history_multiple_messages(self):
        """Test adding multiple messages to history."""
        session = UserSession(user_id=12345, conversation_id="conv_123")

        session.add_to_history(role="user", message="Message 1")
        session.add_to_history(role="assistant", message="Message 2")
        session.add_to_history(role="user", message="Message 3")

        assert len(session.history) == 3
        assert session.history[0]["message"] == "Message 1"
        assert session.history[1]["message"] == "Message 2"
        assert session.history[2]["message"] == "Message 3"

    def test_add_to_history_max_limit(self):
        """Test history respects max limit."""
        session = UserSession(user_id=12345, conversation_id="conv_123")

        # Add 55 messages with max_history=50
        for i in range(55):
            session.add_to_history(role="user", message=f"Message {i}", max_history=50)

        # Should keep only last 50
        assert len(session.history) == 50
        # First message should be "Message 5" (messages 0-4 dropped)
        assert session.history[0]["message"] == "Message 5"
        # Last message should be "Message 54"
        assert session.history[-1]["message"] == "Message 54"

    def test_get_data_size_mb_no_data(self):
        """Test get_data_size_mb returns 0 when no data."""
        session = UserSession(user_id=12345, conversation_id="conv_123")
        assert session.get_data_size_mb() == 0.0

    def test_get_data_size_mb_with_data(self):
        """Test get_data_size_mb calculates size correctly."""
        # Create DataFrame with known size
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000),
            'col3': ['text'] * 1000
        })

        session = UserSession(
            user_id=12345,
            conversation_id="conv_123",
            uploaded_data=df
        )

        size = session.get_data_size_mb()
        # Should be positive and reasonable (depends on pandas memory)
        assert size > 0
        assert size < 10  # Should be less than 10MB for this small dataset


class TestStateManagerConfig:
    """Test StateManagerConfig dataclass."""

    def test_config_defaults(self):
        """Test configuration default values."""
        config = StateManagerConfig()

        assert config.session_timeout_minutes == 30
        assert config.grace_period_minutes == 5
        assert config.max_data_size_mb == 100
        assert config.max_history_messages == 50
        assert config.cleanup_interval_seconds == 300
        assert config.max_concurrent_sessions == 1000

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = StateManagerConfig(
            session_timeout_minutes=60,
            grace_period_minutes=10,
            max_data_size_mb=200,
            max_history_messages=100,
            cleanup_interval_seconds=600,
            max_concurrent_sessions=2000
        )

        assert config.session_timeout_minutes == 60
        assert config.grace_period_minutes == 10
        assert config.max_data_size_mb == 200
        assert config.max_history_messages == 100
        assert config.cleanup_interval_seconds == 600
        assert config.max_concurrent_sessions == 2000

    def test_validation_invalid_timeout(self):
        """Test validation fails for invalid timeout."""
        with pytest.raises(ValueError, match="session_timeout_minutes must be positive"):
            StateManagerConfig(session_timeout_minutes=0)

        with pytest.raises(ValueError, match="session_timeout_minutes must be positive"):
            StateManagerConfig(session_timeout_minutes=-1)

    def test_validation_invalid_grace_period(self):
        """Test validation fails for negative grace period."""
        with pytest.raises(ValueError, match="grace_period_minutes cannot be negative"):
            StateManagerConfig(grace_period_minutes=-1)

    def test_validation_invalid_data_size(self):
        """Test validation fails for invalid max data size."""
        with pytest.raises(ValueError, match="max_data_size_mb must be positive"):
            StateManagerConfig(max_data_size_mb=0)

    def test_validation_invalid_history_size(self):
        """Test validation fails for invalid max history."""
        with pytest.raises(ValueError, match="max_history_messages must be positive"):
            StateManagerConfig(max_history_messages=0)

    def test_validation_invalid_cleanup_interval(self):
        """Test validation fails for invalid cleanup interval."""
        with pytest.raises(ValueError, match="cleanup_interval_seconds must be positive"):
            StateManagerConfig(cleanup_interval_seconds=0)

    def test_validation_invalid_session_limit(self):
        """Test validation fails for invalid session limit."""
        with pytest.raises(ValueError, match="max_concurrent_sessions must be positive"):
            StateManagerConfig(max_concurrent_sessions=0)

    def test_validation_grace_period_zero_allowed(self):
        """Test grace period of 0 is allowed."""
        config = StateManagerConfig(grace_period_minutes=0)
        assert config.grace_period_minutes == 0
