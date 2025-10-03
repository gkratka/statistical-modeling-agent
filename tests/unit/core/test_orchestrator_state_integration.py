"""Unit tests for Orchestrator integration with new StateManager."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch

from src.core.orchestrator import TaskOrchestrator
from src.core.state_manager import (
    StateManager,
    StateManagerConfig,
    UserSession,
    WorkflowType,
    MLTrainingState
)
from src.core.parser import TaskDefinition


@pytest.mark.asyncio
class TestOrchestratorStateIntegration:
    """Test orchestrator integration with new State Manager."""

    @pytest.fixture
    def orchestrator(self):
        config = StateManagerConfig(session_timeout_minutes=30, max_data_size_mb=100)
        return TaskOrchestrator(state_config=config)

    @pytest.fixture
    def sample_task(self):
        return TaskDefinition(
            task_type="stats",
            operation="mean",
            parameters={"columns": ["x", "y"]},
            data_source=None,
            user_id=123,
            conversation_id="conv_1"
        )

    async def test_orchestrator_uses_new_state_manager(self, orchestrator):
        """Test orchestrator uses new StateManager class."""
        assert isinstance(orchestrator.state_manager, StateManager)
        for method in ['get_or_create_session', 'start_workflow', 'transition_state']:
            assert hasattr(orchestrator.state_manager, method)

    async def test_orchestrator_creates_session_for_task(self, orchestrator, sample_task):
        """Test orchestrator creates session when executing task."""
        with patch.object(orchestrator.stats_engine, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"mean_x": 5.0, "mean_y": 10.0}
            result = await orchestrator.execute_task(sample_task)
            assert result is not None

        session = await orchestrator.state_manager.get_session(sample_task.user_id, sample_task.conversation_id)
        assert session.user_id == 123
        assert session.conversation_id == "conv_1"

    async def test_orchestrator_workflow_state_checking(self, orchestrator):
        """Test orchestrator checks workflow state correctly."""
        session = await orchestrator.state_manager.get_or_create_session(123, "conv_1")
        await orchestrator.state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        assert session.workflow_type == WorkflowType.ML_TRAINING
        assert session.current_state == MLTrainingState.AWAITING_DATA.value

    async def test_orchestrator_data_storage_integration(self, orchestrator):
        """Test orchestrator integrates with state manager data storage."""
        session = await orchestrator.state_manager.get_or_create_session(123, "conv_1")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        await orchestrator.state_manager.store_data(session, df)
        assert session.uploaded_data is not None
        assert len(session.uploaded_data) == 3
        assert list(session.uploaded_data.columns) == ["x", "y"]

    async def test_orchestrator_no_old_classes(self, orchestrator):
        """Test orchestrator doesn't use old conflicting classes."""
        import src.core.orchestrator as orch_module
        assert not hasattr(orch_module, 'WorkflowState')
        assert not hasattr(orch_module, 'ConversationState')
        if hasattr(orch_module, 'StateManager'):
            assert getattr(orch_module, 'StateManager').__module__ == 'src.core.state_manager'

    async def test_ml_training_validation_error_message(self, orchestrator):
        """Test ML training task returns proper error message for missing target_column."""
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={},  # Missing required target_column
            data_source=None,
            user_id=123,
            conversation_id="conv_1"
        )
        df = pd.DataFrame({"age": [25, 30, 35], "income": [50000, 60000, 70000], "price": [200000, 250000, 300000]})

        result = await orchestrator.execute_task(task, df)

        # Should return properly formatted error result with specific validation message
        assert result.get("success") is False
        assert "error" in result
        assert "target_column is required" in result["error"]
        assert result.get("error_code") == "VALIDATION"
        assert "suggestions" in result
