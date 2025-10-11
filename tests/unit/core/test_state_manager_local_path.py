"""
Unit tests for StateManager local path workflow states (Phase 4).

Tests the new workflow states for local file path training:
- CHOOSING_DATA_SOURCE
- AWAITING_FILE_PATH
- CONFIRMING_SCHEMA

And the new state transitions.
"""

import pytest
import pandas as pd

from src.core.state_manager import (
    StateManager,
    StateManagerConfig,
    UserSession,
    WorkflowType,
    MLTrainingState,
    ML_TRAINING_TRANSITIONS
)
from src.utils.exceptions import InvalidStateTransitionError


@pytest.mark.asyncio
class TestLocalPathWorkflowStates:

    @pytest.fixture
    def manager(self):
        return StateManager()

    async def test_new_states_exist(self):
        assert hasattr(MLTrainingState, 'CHOOSING_DATA_SOURCE')
        assert hasattr(MLTrainingState, 'AWAITING_FILE_PATH')
        assert hasattr(MLTrainingState, 'CONFIRMING_SCHEMA')

    async def test_transition_from_none_to_choosing_data_source(self, manager):
        session = await manager.get_or_create_session(
            user_id=123,
            conversation_id="test_conv"
        )

        # Start ML training workflow
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Should start at one of the allowed initial states
        session = await manager.get_session(123, "test_conv")
        assert session.current_state in [
            MLTrainingState.CHOOSING_DATA_SOURCE.value,
            MLTrainingState.AWAITING_DATA.value
        ]

    async def test_transition_choosing_to_file_path(self, manager):
        session = await manager.get_or_create_session(123, "test")
        session = await manager.get_or_create_session(123, "test")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await manager.update_session(session)

        # User chose local path
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_FILE_PATH.value)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

    async def test_transition_choosing_to_telegram_upload(self, manager):
        session = await manager.get_or_create_session(123, "test")
        session = await manager.get_or_create_session(123, "test")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value
        await manager.update_session(session)

        # User chose Telegram upload
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_DATA.value)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.AWAITING_DATA.value

    async def test_transition_file_path_to_schema_confirmation(self, manager):
        session = await manager.get_or_create_session(123, "test")
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CHOOSING_DATA_SOURCE.value)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_FILE_PATH.value)

        # File loaded successfully, show schema
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CONFIRMING_SCHEMA.value)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.CONFIRMING_SCHEMA.value

    async def test_transition_schema_accepted_to_target_selection(self, manager):
        session = await manager.get_or_create_session(123, "test")
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CHOOSING_DATA_SOURCE.value)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_FILE_PATH.value)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CONFIRMING_SCHEMA.value)

        # User accepts schema
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.SELECTING_TARGET.value

    async def test_transition_schema_rejected_back_to_file_path(self, manager):
        session = await manager.get_or_create_session(123, "test")
        await manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CHOOSING_DATA_SOURCE.value)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_FILE_PATH.value)
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.CONFIRMING_SCHEMA.value)

        # User rejects schema, wants different file
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.AWAITING_FILE_PATH.value)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.AWAITING_FILE_PATH.value

    async def test_legacy_workflow_still_works(self, manager):
        session = await manager.get_or_create_session(123, "test")

        # Legacy: Start directly at AWAITING_DATA
        session = await manager.get_or_create_session(123, "test")
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_DATA.value
        await manager.update_session(session)

        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.AWAITING_DATA.value

        # Should be able to continue normal workflow
        session = await manager.get_session(123, "test")
        await manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)
        session = await manager.get_session(123, "test")
        assert session.current_state == MLTrainingState.SELECTING_TARGET.value

    async def test_complete_local_path_workflow(self, manager):
        user_id, conv_id = 123, "test"

        # Start with choosing data source
        session = await manager.get_or_create_session(user_id, conv_id)
        await manager.start_workflow(
            user_id, conv_id,
            WorkflowType.ML_TRAINING,
            MLTrainingState.CHOOSING_DATA_SOURCE.value
        )

        # Choose local path
        await manager.transition_state(user_id, conv_id, MLTrainingState.AWAITING_FILE_PATH.value)

        # Provide file path and load
        await manager.transition_state(user_id, conv_id, MLTrainingState.CONFIRMING_SCHEMA.value)

        # Accept schema
        await manager.transition_state(user_id, conv_id, MLTrainingState.SELECTING_TARGET.value)

        # Select features
        await manager.transition_state(user_id, conv_id, MLTrainingState.SELECTING_FEATURES.value)

        # Confirm model
        await manager.transition_state(user_id, conv_id, MLTrainingState.CONFIRMING_MODEL.value)

        # Start training
        await manager.transition_state(user_id, conv_id, MLTrainingState.TRAINING.value)

        # Complete
        await manager.transition_state(user_id, conv_id, MLTrainingState.COMPLETE.value)

        session = await manager.get_session(user_id, conv_id)
        assert session.current_state == MLTrainingState.COMPLETE.value


class TestUserSessionLocalPathFields:

    def test_user_session_has_new_fields(self):
        session = UserSession(
            user_id=123,
            conversation_id="test"
        )

        assert hasattr(session, 'data_source')
        assert hasattr(session, 'file_path')
        assert hasattr(session, 'detected_schema')

    def test_user_session_new_fields_default_to_none(self):
        session = UserSession(user_id=123, conversation_id="test")

        assert session.data_source is None
        assert session.file_path is None
        assert session.detected_schema is None

    def test_user_session_can_set_data_source(self):
        session = UserSession(
            user_id=123,
            conversation_id="test",
            data_source="local_path"
        )

        assert session.data_source == "local_path"

    def test_user_session_can_set_file_path(self):
        session = UserSession(
            user_id=123,
            conversation_id="test",
            file_path="/path/to/data.csv"
        )

        assert session.file_path == "/path/to/data.csv"

    def test_user_session_can_set_detected_schema(self):
        schema_info = {
            'task_type': 'regression',
            'target': 'price',
            'features': ['sqft', 'bedrooms']
        }

        session = UserSession(
            user_id=123,
            conversation_id="test",
            detected_schema=schema_info
        )

        assert session.detected_schema == schema_info
        assert session.detected_schema['task_type'] == 'regression'

    def test_user_session_backward_compatibility(self):
        # Old-style session creation (no new fields)
        session = UserSession(
            user_id=123,
            conversation_id="test",
            workflow_type=WorkflowType.ML_TRAINING,
            current_state=MLTrainingState.AWAITING_DATA.value
        )

        # Should still have new fields (default None)
        assert session.data_source is None
        assert session.file_path is None
        assert session.detected_schema is None

        # Old fields should work
        assert session.workflow_type == WorkflowType.ML_TRAINING
        assert session.current_state == MLTrainingState.AWAITING_DATA.value


class TestMLTrainingTransitions:

    def test_none_can_transition_to_choosing_or_awaiting(self):
        allowed = ML_TRAINING_TRANSITIONS[None]

        assert MLTrainingState.CHOOSING_DATA_SOURCE.value in allowed
        assert MLTrainingState.AWAITING_DATA.value in allowed

    def test_choosing_data_source_transitions(self):
        allowed = ML_TRAINING_TRANSITIONS[MLTrainingState.CHOOSING_DATA_SOURCE.value]

        assert MLTrainingState.AWAITING_FILE_PATH.value in allowed
        assert MLTrainingState.AWAITING_DATA.value in allowed

    def test_awaiting_file_path_transitions(self):
        allowed = ML_TRAINING_TRANSITIONS[MLTrainingState.AWAITING_FILE_PATH.value]

        assert MLTrainingState.CONFIRMING_SCHEMA.value in allowed
        assert len(allowed) == 1

    def test_confirming_schema_transitions(self):
        allowed = ML_TRAINING_TRANSITIONS[MLTrainingState.CONFIRMING_SCHEMA.value]

        assert MLTrainingState.SELECTING_TARGET.value in allowed
        assert MLTrainingState.AWAITING_FILE_PATH.value in allowed

    def test_existing_transitions_unchanged(self):
        # AWAITING_DATA -> SELECTING_TARGET
        assert MLTrainingState.SELECTING_TARGET.value in \
               ML_TRAINING_TRANSITIONS[MLTrainingState.AWAITING_DATA.value]

        # SELECTING_TARGET -> SELECTING_FEATURES
        assert MLTrainingState.SELECTING_FEATURES.value in \
               ML_TRAINING_TRANSITIONS[MLTrainingState.SELECTING_TARGET.value]

        # SELECTING_FEATURES -> CONFIRMING_MODEL
        assert MLTrainingState.CONFIRMING_MODEL.value in \
               ML_TRAINING_TRANSITIONS[MLTrainingState.SELECTING_FEATURES.value]

        # TRAINING -> COMPLETE
        assert MLTrainingState.COMPLETE.value in \
               ML_TRAINING_TRANSITIONS[MLTrainingState.TRAINING.value]


@pytest.mark.asyncio
class TestWorkflowIntegration:

    @pytest.fixture
    def manager(self):
        return StateManager()

    async def test_local_path_workflow_with_session_data(self, manager):
        user_id, conv_id = 456, "integration_test"

        # Create session and start workflow
        session = await manager.get_or_create_session(user_id, conv_id)
        await manager.start_workflow(
            user_id, conv_id,
            WorkflowType.ML_TRAINING,
            MLTrainingState.CHOOSING_DATA_SOURCE.value
        )

        # Store data source choice
        session = await manager.get_session(user_id, conv_id)
        session.data_source = "local_path"

        # Transition to file path input
        await manager.transition_state(user_id, conv_id, MLTrainingState.AWAITING_FILE_PATH.value)

        # Store file path
        session = await manager.get_session(user_id, conv_id)
        session.file_path = "/data/housing.csv"

        # Transition to schema confirmation
        await manager.transition_state(user_id, conv_id, MLTrainingState.CONFIRMING_SCHEMA.value)

        # Store detected schema
        session = await manager.get_session(user_id, conv_id)
        session.detected_schema = {
            'task_type': 'regression',
            'target': 'price',
            'features': ['sqft', 'bedrooms', 'bathrooms']
        }

        # Accept schema and continue
        await manager.transition_state(user_id, conv_id, MLTrainingState.SELECTING_TARGET.value)

        # Verify all data was preserved
        final_session = await manager.get_session(user_id, conv_id)
        assert final_session.data_source == "local_path"
        assert final_session.file_path == "/data/housing.csv"
        assert final_session.detected_schema is not None
        assert final_session.detected_schema['task_type'] == 'regression'
        assert final_session.current_state == MLTrainingState.SELECTING_TARGET.value

    async def test_telegram_workflow_unchanged(self, manager):
        user_id, conv_id = 789, "telegram_test"

        session = await manager.get_or_create_session(user_id, conv_id)
        await manager.start_workflow(
            user_id, conv_id,
            WorkflowType.ML_TRAINING,
            MLTrainingState.AWAITING_DATA.value
        )

        # Store data (Telegram upload)
        session = await manager.get_session(user_id, conv_id)
        session.uploaded_data = pd.DataFrame({'a': [1, 2, 3]})
        session.data_source = "telegram"  # Optional: track source

        # Continue normal workflow
        await manager.transition_state(user_id, conv_id, MLTrainingState.SELECTING_TARGET.value)
        await manager.transition_state(user_id, conv_id, MLTrainingState.SELECTING_FEATURES.value)

        final_session = await manager.get_session(user_id, conv_id)
        assert final_session.data_source == "telegram"
        assert final_session.uploaded_data is not None
        assert final_session.current_state == MLTrainingState.SELECTING_FEATURES.value
