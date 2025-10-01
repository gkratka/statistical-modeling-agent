"""
Unit tests for state machine logic.

Tests state transitions, prerequisite checking, and workflow validation.
"""

import pytest
import pandas as pd

from src.core.state_manager import (
    StateMachine,
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)


class TestStateMachine:
    """Test StateMachine class."""

    def test_check_prerequisites_ml_training_met(self):
        """Test ML training prerequisites when met."""
        # Session with data uploaded
        session = UserSession(
            user_id=123,
            conversation_id="conv",
            uploaded_data=pd.DataFrame({"x": [1, 2, 3]})
        )

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.SELECTING_TARGET.value,
            session
        )

        assert prereqs_met
        assert missing == []

    def test_check_prerequisites_ml_training_not_met(self):
        """Test ML training prerequisites when not met."""
        # Session without data
        session = UserSession(user_id=123, conversation_id="conv")

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.SELECTING_TARGET.value,
            session
        )

        assert not prereqs_met
        assert "uploaded_data" in missing

    def test_check_prerequisites_target_selection(self):
        """Test prerequisite for feature selection (target must be selected)."""
        # Session without target selected
        session = UserSession(
            user_id=123,
            conversation_id="conv",
            uploaded_data=pd.DataFrame({"x": [1]})
        )

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.SELECTING_FEATURES.value,
            session
        )

        assert not prereqs_met
        assert "target_selection" in missing

        # Now add target selection
        session.selections["target"] = "y"

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.SELECTING_FEATURES.value,
            session
        )

        assert prereqs_met
        assert missing == []

    def test_check_prerequisites_feature_selection(self):
        """Test prerequisite for model confirmation (features must be selected)."""
        session = UserSession(
            user_id=123,
            conversation_id="conv",
            selections={"target": "y"}
        )

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.CONFIRMING_MODEL.value,
            session
        )

        assert not prereqs_met
        assert "feature_selection" in missing

        # Add features
        session.selections["features"] = ["x1", "x2"]

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.CONFIRMING_MODEL.value,
            session
        )

        assert prereqs_met

    def test_check_prerequisites_model_type_selection(self):
        """Test prerequisite for training (model type must be selected)."""
        session = UserSession(
            user_id=123,
            conversation_id="conv",
            selections={"target": "y", "features": ["x1"]}
        )

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.TRAINING.value,
            session
        )

        assert not prereqs_met
        assert "model_type_selection" in missing

        # Add model type
        session.selections["model_type"] = "neural_network"

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_TRAINING,
            MLTrainingState.TRAINING.value,
            session
        )

        assert prereqs_met

    def test_check_prerequisites_ml_prediction(self):
        """Test ML prediction prerequisites."""
        # Need model for data upload
        session = UserSession(user_id=123, conversation_id="conv")

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_PREDICTION,
            MLPredictionState.AWAITING_DATA.value,
            session
        )

        assert not prereqs_met
        assert "trained_model" in missing

        # Add model
        session.model_ids.append("model_123")

        prereqs_met, missing = StateMachine.check_prerequisites(
            WorkflowType.ML_PREDICTION,
            MLPredictionState.AWAITING_DATA.value,
            session
        )

        assert prereqs_met

    def test_validate_transition_success(self):
        """Test successful state transition validation."""
        session = UserSession(
            user_id=123,
            conversation_id="conv",
            uploaded_data=pd.DataFrame({"x": [1]})
        )

        success, error_msg, missing = StateMachine.validate_transition(
            WorkflowType.ML_TRAINING,
            MLTrainingState.AWAITING_DATA.value,
            MLTrainingState.SELECTING_TARGET.value,
            session
        )

        assert success
        assert error_msg is None
        assert missing == []

    def test_validate_transition_invalid_transition(self):
        """Test validation fails for invalid transition."""
        session = UserSession(user_id=123, conversation_id="conv")

        success, error_msg, missing = StateMachine.validate_transition(
            WorkflowType.ML_TRAINING,
            MLTrainingState.AWAITING_DATA.value,
            MLTrainingState.TRAINING.value,  # Can't skip states
            session
        )

        assert not success
        assert "Invalid transition" in error_msg

    def test_validate_transition_prerequisites_not_met(self):
        """Test validation fails when prerequisites not met."""
        session = UserSession(user_id=123, conversation_id="conv")  # No data

        success, error_msg, missing = StateMachine.validate_transition(
            WorkflowType.ML_TRAINING,
            MLTrainingState.AWAITING_DATA.value,
            MLTrainingState.SELECTING_TARGET.value,
            session
        )

        assert not success
        assert "Prerequisites not met" in error_msg
        assert "uploaded_data" in missing

    def test_get_valid_next_states(self):
        """Test getting valid next states."""
        # From start
        next_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_TRAINING,
            None
        )
        assert MLTrainingState.AWAITING_DATA.value in next_states

        # From middle state
        next_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_TRAINING,
            MLTrainingState.SELECTING_TARGET.value
        )
        assert MLTrainingState.SELECTING_FEATURES.value in next_states
        assert len(next_states) == 1

        # From terminal state
        next_states = StateMachine.get_valid_next_states(
            WorkflowType.ML_TRAINING,
            MLTrainingState.COMPLETE.value
        )
        assert len(next_states) == 0
