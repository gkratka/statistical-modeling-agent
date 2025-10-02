"""
Unit tests for ResponseBuilder class.

Tests workflow prompts, error formatting, column selection, and progress indicators.
"""

import pytest
from typing import List

from src.bot.response_builder import ResponseBuilder
from src.core.state_manager import (
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)
from src.utils.exceptions import (
    PrerequisiteNotMetError,
    InvalidStateTransitionError,
    DataSizeLimitError
)


class TestResponseBuilder:
    """Test ResponseBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create response builder for testing."""
        return ResponseBuilder()

    @pytest.fixture
    def sample_session(self):
        """Create sample session for testing."""
        return UserSession(
            user_id=123,
            conversation_id="conv_1",
            workflow_type=WorkflowType.ML_TRAINING,
            current_state=MLTrainingState.AWAITING_DATA.value
        )

    def test_build_workflow_prompt_awaiting_data(self, builder, sample_session):
        """Test prompt for awaiting data state."""
        prompt = builder.build_workflow_prompt(
            MLTrainingState.AWAITING_DATA.value,
            sample_session
        )

        assert "upload" in prompt.lower()
        assert "data" in prompt.lower()

    def test_build_workflow_prompt_selecting_target(self, builder, sample_session):
        """Test prompt for target selection state."""
        sample_session.current_state = MLTrainingState.SELECTING_TARGET.value

        prompt = builder.build_workflow_prompt(
            MLTrainingState.SELECTING_TARGET.value,
            sample_session
        )

        assert "target" in prompt.lower()
        assert "variable" in prompt.lower() or "column" in prompt.lower()

    def test_build_workflow_prompt_selecting_features(self, builder, sample_session):
        """Test prompt for feature selection state."""
        sample_session.current_state = MLTrainingState.SELECTING_FEATURES.value

        prompt = builder.build_workflow_prompt(
            MLTrainingState.SELECTING_FEATURES.value,
            sample_session
        )

        assert "feature" in prompt.lower()
        assert "select" in prompt.lower()

    def test_build_workflow_prompt_confirming_model(self, builder, sample_session):
        """Test prompt for model confirmation state."""
        sample_session.current_state = MLTrainingState.CONFIRMING_MODEL.value

        prompt = builder.build_workflow_prompt(
            MLTrainingState.CONFIRMING_MODEL.value,
            sample_session
        )

        assert "model" in prompt.lower()
        assert "neural" in prompt.lower() or "random" in prompt.lower()

    def test_build_workflow_prompt_training(self, builder, sample_session):
        """Test prompt for training state."""
        sample_session.current_state = MLTrainingState.TRAINING.value

        prompt = builder.build_workflow_prompt(
            MLTrainingState.TRAINING.value,
            sample_session
        )

        assert "training" in prompt.lower()

    def test_format_column_selection_target(self, builder):
        """Test formatting target column selection prompt."""
        columns = ["age", "income", "score", "category"]

        prompt = builder.format_column_selection(columns, "target")

        assert "target" in prompt.lower()
        for col in columns:
            assert col in prompt

    def test_format_column_selection_features(self, builder):
        """Test formatting feature column selection prompt."""
        columns = ["feature1", "feature2", "feature3"]

        prompt = builder.format_column_selection(columns, "features")

        assert "feature" in prompt.lower()
        for col in columns:
            assert col in prompt

    def test_format_column_selection_numbered(self, builder):
        """Test column selection with numbered list."""
        columns = ["col1", "col2", "col3"]

        prompt = builder.format_column_selection(
            columns,
            "target",
            numbered=True
        )

        assert "1." in prompt or "1)" in prompt
        assert "2." in prompt or "2)" in prompt
        assert "3." in prompt or "3)" in prompt

    def test_build_progress_indicator_ml_training(self, builder, sample_session):
        """Test progress indicator for ML training workflow."""
        sample_session.current_state = MLTrainingState.SELECTING_TARGET.value

        progress = builder.build_progress_indicator(
            MLTrainingState.SELECTING_TARGET.value,
            WorkflowType.ML_TRAINING
        )

        assert progress is not None
        assert "2" in progress or "step" in progress.lower()

    def test_build_progress_indicator_complete(self, builder, sample_session):
        """Test progress indicator at completion."""
        sample_session.current_state = MLTrainingState.COMPLETE.value

        progress = builder.build_progress_indicator(
            MLTrainingState.COMPLETE.value,
            WorkflowType.ML_TRAINING
        )

        assert "complete" in progress.lower() or "done" in progress.lower()

    def test_format_prerequisite_error(self, builder, sample_session):
        """Test formatting prerequisite error."""
        missing = ["uploaded_data", "target_selection"]

        error_msg = builder.format_prerequisite_error(missing, sample_session)

        assert "prerequisite" in error_msg.lower() or "required" in error_msg.lower()
        assert "upload" in error_msg.lower() and "data" in error_msg.lower()
        assert "target" in error_msg.lower()

    def test_format_state_error_invalid_transition(self, builder, sample_session):
        """Test formatting invalid transition error."""
        error = InvalidStateTransitionError(
            "Invalid transition",
            current_state=MLTrainingState.AWAITING_DATA.value,
            requested_state=MLTrainingState.TRAINING.value
        )

        formatted = builder.format_state_error(error, sample_session)

        assert "invalid" in formatted.lower()
        assert MLTrainingState.AWAITING_DATA.value in formatted
        assert MLTrainingState.TRAINING.value in formatted

    def test_format_state_error_data_limit(self, builder, sample_session):
        """Test formatting data size limit error."""
        error = DataSizeLimitError(
            "Data too large",
            actual_size_mb=150.0,
            limit_mb=100
        )

        formatted = builder.format_state_error(error, sample_session)

        assert "150" in formatted
        assert "100" in formatted

    def test_format_success_message_training_complete(self, builder):
        """Test formatting training completion success message."""
        results = {
            "accuracy": 0.95,
            "model_id": "model_123"
        }

        message = builder.format_success_message(
            "training_complete",
            results
        )

        assert "success" in message.lower() or "complete" in message.lower()
        assert "95" in message or "0.95" in message
        assert "model_123" in message

    def test_format_success_message_prediction_complete(self, builder):
        """Test formatting prediction completion success message."""
        results = {
            "predictions": [0, 1, 1, 0],
            "count": 4
        }

        message = builder.format_success_message(
            "prediction_complete",
            results
        )

        assert "prediction" in message.lower()
        assert "4" in message

    def test_format_model_options(self, builder):
        """Test formatting model type options."""
        options = builder.format_model_options()

        assert "neural" in options.lower()
        assert "random forest" in options.lower()
        assert "gradient boosting" in options.lower()
        assert "1" in options or "1." in options

    def test_format_workflow_summary_ml_training(self, builder, sample_session):
        """Test formatting workflow summary."""
        sample_session.selections = {
            "target": "price",
            "features": ["age", "income"],
            "model_type": "neural_network"
        }
        sample_session.current_state = MLTrainingState.CONFIRMING_MODEL.value

        summary = builder.format_workflow_summary(sample_session)

        assert "target" in summary.lower()
        assert "price" in summary
        assert "age" in summary
        assert "income" in summary
        assert "neural" in summary.lower()

    def test_format_help_message(self, builder):
        """Test formatting help message."""
        help_msg = builder.format_help_message()

        assert "/train" in help_msg
        assert "/predict" in help_msg
        assert "/stats" in help_msg
        assert "/cancel" in help_msg

    def test_format_welcome_message(self, builder):
        """Test formatting welcome message."""
        welcome = builder.format_welcome_message()

        assert "welcome" in welcome.lower() or "hello" in welcome.lower()
        assert "statistical" in welcome.lower() or "analysis" in welcome.lower()

    def test_format_cancel_confirmation(self, builder, sample_session):
        """Test formatting workflow cancellation confirmation."""
        confirmation = builder.format_cancel_confirmation(sample_session)

        assert "cancel" in confirmation.lower()
        assert "ml_training" in confirmation.lower() or "training" in confirmation.lower()

    def test_format_timeout_warning(self, builder, sample_session):
        """Test formatting session timeout warning."""
        minutes_left = 5

        warning = builder.format_timeout_warning(sample_session, minutes_left)

        assert "5" in warning
        assert "minute" in warning.lower()
        assert "expire" in warning.lower() or "timeout" in warning.lower()

    def test_format_data_upload_confirmation(self, builder):
        """Test formatting data upload confirmation."""
        metadata = {
            "filename": "data.csv",
            "rows": 1000,
            "columns": 15
        }

        confirmation = builder.format_data_upload_confirmation(metadata)

        assert "data.csv" in confirmation
        assert "1000" in confirmation
        assert "15" in confirmation
