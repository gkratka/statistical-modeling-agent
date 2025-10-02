"""Unit tests for IntegrationLayer class."""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta

from src.bot.integration_layer import IntegrationLayer
from src.core.state_manager import (
    StateManager,
    UserSession,
    WorkflowType,
    MLTrainingState
)
from src.utils.exceptions import DataSizeLimitError


@pytest.mark.asyncio
class TestIntegrationLayer:
    """Test IntegrationLayer functionality."""

    @pytest.fixture
    def state_manager(self):
        return StateManager()

    @pytest.fixture
    def integration_layer(self, state_manager):
        return IntegrationLayer(state_manager=state_manager)

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 1]
        })

    async def test_state_manager_integration(self, integration_layer):
        """Test integration layer has state manager."""
        assert hasattr(integration_layer, 'state_manager')
        assert integration_layer.state_manager is not None

    async def test_store_uploaded_data_success(self, integration_layer, sample_dataframe):
        """Test storing uploaded data successfully."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")
        metadata = {"filename": "test.csv", "upload_time": datetime.now().isoformat()}

        await integration_layer.store_uploaded_data(session, sample_dataframe, metadata)

        assert session.uploaded_data is not None
        assert len(session.uploaded_data) == 5
        assert list(session.uploaded_data.columns) == ['feature1', 'feature2', 'target']

    async def test_store_uploaded_data_size_limit(self, integration_layer):
        """Test data size limit enforcement."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")
        large_df = pd.DataFrame({f'col_{i}': range(1000000) for i in range(20)})

        with pytest.raises(DataSizeLimitError):
            await integration_layer.store_uploaded_data(session, large_df, {"filename": "large.csv"})

    @pytest.mark.parametrize("message_text,workflow_active,expected_handler", [
        ("calculate mean", False, "general"),
        ("/start", False, "command"),
        ("column 3", True, "workflow"),
    ])
    async def test_route_message(self, integration_layer, message_text, workflow_active, expected_handler):
        """Test message routing to correct handler."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")

        if workflow_active:
            await integration_layer.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)

        handler_type = await integration_layer.route_message(session, message_text)
        assert handler_type == expected_handler

    async def test_handle_workflow_message_ml_training(self, integration_layer, sample_dataframe):
        """Test handling workflow message for ML training."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")
        await integration_layer.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.uploaded_data = sample_dataframe

        await integration_layer.state_manager.transition_state(
            session, MLTrainingState.SELECTING_TARGET.value
        )

        response = await integration_layer.handle_workflow_message(session, "target")
        assert response is not None
        assert "feature" in response.lower() or "next" in response.lower()

    async def test_get_session_info(self, integration_layer):
        """Test getting session information."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")
        await integration_layer.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        await integration_layer.state_manager.add_to_history(session, "user", "start training")

        info = await integration_layer.get_session_info(session)

        assert info["user_id"] == 123
        assert info["workflow_type"] == "ml_training"
        assert info["current_state"] == "awaiting_data"
        assert "created_at" in info
        assert "message_count" in info

    async def test_cancel_workflow(self, integration_layer, sample_dataframe):
        """Test canceling active workflow."""
        session = await integration_layer.state_manager.get_or_create_session(123, "conv_1")
        await integration_layer.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.uploaded_data = sample_dataframe
        session.selections["key"] = "value"

        await integration_layer.cancel_workflow(session)

        assert session.workflow_type is None
        assert session.current_state is None
        assert len(session.selections) == 0
        assert session.uploaded_data is None

    async def test_concurrent_session_access(self, integration_layer):
        """Test concurrent access to different sessions."""
        async def create_and_use_session(user_id: int):
            session = await integration_layer.state_manager.get_or_create_session(
                user_id, f"conv_{user_id}"
            )
            await integration_layer.state_manager.add_to_history(
                session, "user", f"message from {user_id}"
            )
            return session

        results = await asyncio.gather(
            create_and_use_session(1),
            create_and_use_session(2),
            create_and_use_session(3),
        )

        assert len(results) == 3
        assert all(results[i].user_id == i+1 for i in range(3))
