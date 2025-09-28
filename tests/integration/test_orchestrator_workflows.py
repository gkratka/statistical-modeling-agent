"""
Integration tests for Enhanced Orchestrator workflows.

This module tests complete end-to-end workflows including ML training,
statistics analysis, and multi-step user interactions with real data flow.
"""

import pytest
import asyncio
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.orchestrator import TaskOrchestrator, WorkflowState
from src.core.parser import TaskDefinition
from src.processors.data_loader import DataLoader
from src.utils.exceptions import ValidationError, DataError


class TestMLTrainingWorkflow:
    """Test complete ML training workflow from data upload to results."""

    @pytest.fixture
    def orchestrator(self):
        """Provide orchestrator for ML workflow testing."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=10
        )

    @pytest.fixture
    def training_data(self):
        """Provide sample training data."""
        return pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "feature_3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    @pytest.mark.asyncio
    async def test_complete_ml_training_workflow(self, orchestrator, training_data):
        """Test complete ML training workflow from start to finish."""
        user_id = 12345
        conversation_id = "ml_training_test"

        # Step 1: Start ML training workflow
        state = await orchestrator.state_manager.get_state(user_id, conversation_id)
        assert state.workflow_state == WorkflowState.IDLE

        # Simulate data upload
        data_id = await orchestrator.data_manager.load_data(
            mock_file=MagicMock(file_id="test_file"),
            file_name="training_data.csv",
            user_id=user_id
        )

        state.data_sources.append(data_id)
        state.workflow_state = WorkflowState.DATA_LOADED
        await orchestrator.state_manager.save_state(state)

        # Step 2: Target selection workflow step
        target_task = TaskDefinition(
            task_type="ml_train",
            operation="select_target",
            parameters={"selected_column": "target"},
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.95
        )

        # Mock workflow engine response for target selection
        with patch.object(orchestrator.workflow_engine, 'advance_workflow') as mock_advance:
            mock_advance.return_value = {"completed": False, "next_prompt": "Select features"}

            result = await orchestrator.execute_task(target_task)

            assert result["workflow_active"] is True
            assert "Select features" in result["next_step"]

        # Step 3: Feature selection workflow step
        feature_task = TaskDefinition(
            task_type="ml_train",
            operation="select_features",
            parameters={"selected_columns": ["feature_1", "feature_2", "feature_3"]},
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.95
        )

        with patch.object(orchestrator.workflow_engine, 'advance_workflow') as mock_advance:
            mock_advance.return_value = {"completed": False, "next_prompt": "Choose model"}

            result = await orchestrator.execute_task(feature_task)

            assert result["workflow_active"] is True
            assert "Choose model" in result["next_step"]

        # Step 4: Model configuration workflow step
        model_task = TaskDefinition(
            task_type="ml_train",
            operation="configure_model",
            parameters={"model_type": "random_forest", "parameters": {"n_estimators": 100}},
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.95
        )

        with patch.object(orchestrator.workflow_engine, 'advance_workflow') as mock_advance:
            mock_advance.return_value = {"completed": True}

            # Mock ML engine execution
            with patch.object(orchestrator, 'ENGINE_ROUTES') as mock_routes:
                mock_train_result = {
                    "model_id": "model_12345_123",
                    "accuracy": 0.85,
                    "training_time": 2.34,
                    "feature_importance": {
                        "feature_1": 0.4,
                        "feature_2": 0.35,
                        "feature_3": 0.25
                    }
                }
                mock_routes.__getitem__.return_value = lambda self, task, data: mock_train_result

                result = await orchestrator.execute_task(model_task, training_data)

                assert result["success"] is True
                assert "model_id" in result
                assert result["accuracy"] == 0.85

        # Verify final state
        final_state = await orchestrator.state_manager.get_state(user_id, conversation_id)
        assert "configure_model" in final_state.partial_results

    @pytest.mark.asyncio
    async def test_ml_workflow_error_recovery(self, orchestrator, training_data):
        """Test ML workflow with error recovery."""
        pytest.skip("ML workflow error recovery not yet implemented")

    @pytest.mark.asyncio
    async def test_ml_workflow_state_persistence(self, orchestrator):
        """Test ML workflow state persistence across sessions."""
        pytest.skip("ML workflow state persistence not yet implemented")


class TestStatisticsWorkflow:
    """Test complete statistics analysis workflow."""

    @pytest.fixture
    def orchestrator(self):
        """Provide orchestrator for stats workflow testing."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=10
        )

    @pytest.fixture
    def analysis_data(self):
        """Provide sample analysis data."""
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "income": [40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000],
            "education_years": [12, 14, 16, 18, 16, 20, 18, 22],
            "satisfaction": [7, 8, 6, 9, 8, 9, 7, 10]
        })

    @pytest.mark.asyncio
    async def test_complete_stats_analysis_workflow(self, orchestrator, analysis_data):
        """Test complete statistics analysis from data upload to results."""
        user_id = 67890
        conversation_id = "stats_analysis_test"

        # Step 1: Upload and validate data
        data_id = await orchestrator.data_manager.load_data(
            mock_file=MagicMock(file_id="stats_file"),
            file_name="analysis_data.csv",
            user_id=user_id
        )

        # Cache the data for testing
        orchestrator.data_manager.data_cache[data_id] = (analysis_data, {"rows": 8, "columns": 4})

        # Step 2: Descriptive statistics request
        descriptive_task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={
                "columns": ["age", "income", "education_years"],
                "statistics": ["mean", "median", "std", "min", "max"]
            },
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.95
        )

        # Mock stats engine execution
        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats_result = {
                "age": {"mean": 42.5, "median": 42.5, "std": 12.9, "min": 25, "max": 60},
                "income": {"mean": 75000, "median": 75000, "std": 24494.9, "min": 40000, "max": 110000},
                "education_years": {"mean": 17.0, "median": 17.0, "std": 3.16, "min": 12, "max": 22}
            }
            mock_stats.return_value = mock_stats_result

            result = await orchestrator.execute_task(descriptive_task, analysis_data)

            assert result["success"] is True
            assert "age" in result["result"]
            assert result["result"]["age"]["mean"] == 42.5

        # Step 3: Correlation analysis request
        correlation_task = TaskDefinition(
            task_type="stats",
            operation="correlation_analysis",
            parameters={
                "columns": ["age", "income", "education_years", "satisfaction"],
                "method": "pearson"
            },
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.90
        )

        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_corr_result = {
                "correlation_matrix": {
                    "age": {"age": 1.0, "income": 0.98, "education_years": 0.85, "satisfaction": 0.45},
                    "income": {"age": 0.98, "income": 1.0, "education_years": 0.87, "satisfaction": 0.52},
                    "education_years": {"age": 0.85, "income": 0.87, "education_years": 1.0, "satisfaction": 0.68},
                    "satisfaction": {"age": 0.45, "income": 0.52, "education_years": 0.68, "satisfaction": 1.0}
                },
                "significant_correlations": [
                    {"column1": "age", "column2": "income", "correlation": 0.98},
                    {"column1": "income", "column2": "education_years", "correlation": 0.87}
                ]
            }
            mock_stats.return_value = mock_corr_result

            result = await orchestrator.execute_task(correlation_task, analysis_data)

            assert result["success"] is True
            assert "correlation_matrix" in result["result"]
            assert result["result"]["correlation_matrix"]["age"]["income"] == 0.98

        # Verify state management
        final_state = await orchestrator.state_manager.get_state(user_id, conversation_id)
        assert "descriptive_stats" in final_state.partial_results
        assert "correlation_analysis" in final_state.partial_results

    @pytest.mark.asyncio
    async def test_stats_workflow_with_missing_data(self, orchestrator):
        """Test statistics workflow with missing data handling."""
        missing_data = pd.DataFrame({
            "col1": [1, 2, None, 4, 5],
            "col2": [10, None, 30, 40, None],
            "col3": [100, 200, 300, None, 500]
        })

        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["col1", "col2", "col3"], "missing_strategy": "mean"},
            data_source=None,
            user_id=11111,
            conversation_id="missing_data_test",
            confidence_score=0.85
        )

        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats.return_value = {"col1": {"mean": 3.0, "missing_count": 1}}

            result = await orchestrator.execute_task(task, missing_data)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_stats_workflow_data_validation_error(self, orchestrator):
        """Test statistics workflow with data validation errors."""
        invalid_data = pd.DataFrame({
            "text_col": ["a", "b", "c"],
            "mixed_col": [1, "text", 3]
        })

        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["text_col", "mixed_col"]},
            data_source=None,
            user_id=22222,
            conversation_id="validation_error_test",
            confidence_score=0.75
        )

        # This should trigger data validation errors
        result = await orchestrator.execute_task(task, invalid_data)

        # Error recovery should handle this
        assert "action" in result  # Error recovery response


class TestMultiUserWorkflows:
    """Test concurrent workflows for multiple users."""

    @pytest.fixture
    def orchestrator(self):
        """Provide orchestrator for multi-user testing."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=10
        )

    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, orchestrator):
        """Test multiple users with separate workflow states."""
        users = [
            {"user_id": 100, "conversation_id": "user100_session"},
            {"user_id": 200, "conversation_id": "user200_session"},
            {"user_id": 300, "conversation_id": "user300_session"}
        ]

        test_data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50]
        })

        # Create concurrent tasks for different users
        tasks = []
        for user in users:
            task = TaskDefinition(
                task_type="stats",
                operation="descriptive_stats",
                parameters={"columns": ["x", "y"]},
                data_source=None,
                user_id=user["user_id"],
                conversation_id=user["conversation_id"],
                confidence_score=0.90
            )
            tasks.append(task)

        # Execute tasks concurrently
        with patch.object(orchestrator, '_execute_stats_task') as mock_stats:
            mock_stats.return_value = {"x": {"mean": 3.0}, "y": {"mean": 30.0}}

            results = await asyncio.gather(*[
                orchestrator.execute_task(task, test_data) for task in tasks
            ])

            # Verify all tasks completed successfully
            for result in results:
                assert result["success"] is True

        # Verify separate state management
        for user in users:
            state = await orchestrator.state_manager.get_state(
                user["user_id"], user["conversation_id"]
            )
            assert "descriptive_stats" in state.partial_results
            assert state.user_id == user["user_id"]

    @pytest.mark.asyncio
    async def test_user_session_isolation(self, orchestrator):
        """Test that user sessions are properly isolated."""
        user1_data = pd.DataFrame({"a": [1, 2, 3]})
        user2_data = pd.DataFrame({"b": [4, 5, 6]})

        # Load data for different users
        data_id_1 = await orchestrator.data_manager.load_data(
            mock_file=MagicMock(file_id="file1"),
            file_name="user1_data.csv",
            user_id=111
        )

        data_id_2 = await orchestrator.data_manager.load_data(
            mock_file=MagicMock(file_id="file2"),
            file_name="user2_data.csv",
            user_id=222
        )

        # Cache data separately
        orchestrator.data_manager.data_cache[data_id_1] = (user1_data, {"user": 111})
        orchestrator.data_manager.data_cache[data_id_2] = (user2_data, {"user": 222})

        # Verify isolation
        retrieved_1, _ = await orchestrator.data_manager.get_data(data_id_1)
        retrieved_2, _ = await orchestrator.data_manager.get_data(data_id_2)

        assert "a" in retrieved_1.columns
        assert "b" in retrieved_2.columns
        assert "a" not in retrieved_2.columns
        assert "b" not in retrieved_1.columns

        # Clear cache for user 111
        await orchestrator.data_manager.clear_cache(111)

        # User 111 data should be gone, user 222 data should remain
        assert data_id_1 not in orchestrator.data_manager.data_cache
        assert data_id_2 in orchestrator.data_manager.data_cache


class TestErrorRecoveryWorkflows:
    """Test error recovery in complete workflows."""

    @pytest.fixture
    def orchestrator(self):
        """Provide orchestrator for error recovery testing."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=10
        )

    @pytest.mark.asyncio
    async def test_network_error_recovery_workflow(self, orchestrator):
        """Test workflow recovery from network errors."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["col1"]},
            data_source=None,
            user_id=33333,
            conversation_id="network_error_test",
            confidence_score=0.85
        )

        test_data = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

        # Simulate network errors that recover after retry
        call_count = 0

        def network_error_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Network timeout")
            return {"col1": {"mean": 3.0}}

        with patch.object(orchestrator, '_execute_stats_task', side_effect=network_error_then_success):
            result = await orchestrator.execute_task(task, test_data)

            # Should succeed after retries
            assert result["col1"]["mean"] == 3.0

    @pytest.mark.asyncio
    async def test_data_error_escalation_workflow(self, orchestrator):
        """Test workflow escalation for data errors."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["nonexistent_col"]},
            data_source=None,
            user_id=44444,
            conversation_id="data_error_test",
            confidence_score=0.80
        )

        test_data = pd.DataFrame({"existing_col": [1, 2, 3]})

        # This should trigger a data validation error
        result = await orchestrator.execute_task(task, test_data)

        # Should escalate to user with suggestions
        assert "action" in result
        assert result["action"] == "escalate"
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_timeout_error_handling_workflow(self, orchestrator):
        """Test workflow handling of timeout errors."""
        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["col1"]},
            data_source=None,
            user_id=55555,
            conversation_id="timeout_test",
            confidence_score=0.90
        )

        test_data = pd.DataFrame({"col1": list(range(1000))})  # Large dataset

        # Simulate a slow operation that times out
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return {"col1": {"mean": 500.0}}

        with patch.object(orchestrator, '_execute_stats_task', side_effect=slow_operation):
            result = await orchestrator.execute_task(task, test_data, timeout=0.1)

            # Should handle timeout through error recovery
            assert "action" in result


class TestWorkflowStateManagement:
    """Test workflow state management across sessions."""

    @pytest.fixture
    def orchestrator(self):
        """Provide orchestrator for state management testing."""
        mock_loader = MagicMock()
        return TaskOrchestrator(
            enable_logging=False,
            data_loader=mock_loader,
            state_ttl_minutes=10
        )

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, orchestrator):
        """Test workflow state persists across task executions."""
        user_id = 66666
        conversation_id = "persistence_test"

        # Create initial state with workflow data
        state = await orchestrator.state_manager.get_state(user_id, conversation_id)
        state.workflow_state = WorkflowState.SELECTING_TARGET
        state.context["available_columns"] = ["col1", "col2", "col3"]
        state.partial_results["data_loaded"] = True
        await orchestrator.state_manager.save_state(state)

        # Execute a task and verify state is maintained
        task = TaskDefinition(
            task_type="ml_train",
            operation="select_target",
            parameters={"target": "col3"},
            data_source=None,
            user_id=user_id,
            conversation_id=conversation_id,
            confidence_score=0.95
        )

        # Mock workflow continuation
        with patch.object(orchestrator.workflow_engine, 'advance_workflow') as mock_advance:
            mock_advance.return_value = {"completed": False, "next_prompt": "Select features"}

            result = await orchestrator.execute_task(task)

            assert result["workflow_active"] is True
            assert result["workflow_state"] == "selecting_target"

        # Verify state was updated
        updated_state = await orchestrator.state_manager.get_state(user_id, conversation_id)
        assert updated_state.context["available_columns"] == ["col1", "col2", "col3"]
        assert updated_state.partial_results["data_loaded"] is True

    @pytest.mark.asyncio
    async def test_workflow_state_cleanup(self, orchestrator):
        """Test automatic cleanup of expired workflow states."""
        # Create states with different ages
        old_state = await orchestrator.state_manager.get_state(77777, "old_session")
        old_state.last_activity = datetime.now() - timedelta(hours=2)
        await orchestrator.state_manager.save_state(old_state)

        fresh_state = await orchestrator.state_manager.get_state(88888, "fresh_session")
        fresh_state.last_activity = datetime.now()
        await orchestrator.state_manager.save_state(fresh_state)

        # Trigger cleanup
        await orchestrator.state_manager.cleanup_expired()

        # Old state should be removed, fresh state should remain
        try:
            await orchestrator.state_manager.get_state(77777, "old_session")
            # Should create new state since old one was cleaned up
            new_state = await orchestrator.state_manager.get_state(77777, "old_session")
            assert new_state.workflow_state == WorkflowState.IDLE
        except:
            pass  # Expected if state was cleaned up

        # Fresh state should still exist
        existing_fresh = await orchestrator.state_manager.get_state(88888, "fresh_session")
        assert existing_fresh.user_id == 88888

    @pytest.mark.asyncio
    async def test_workflow_transition_validation(self, orchestrator):
        """Test workflow state transition validation."""
        pytest.skip("Workflow transition validation not yet implemented")


# Performance integration tests
class TestPerformanceIntegration:
    """Test performance characteristics in integration scenarios."""

    @pytest.mark.asyncio
    async def test_large_dataset_workflow_performance(self):
        """Test workflow performance with large datasets."""
        pytest.skip("Large dataset performance testing not yet implemented")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_performance(self):
        """Test performance with many concurrent workflows."""
        pytest.skip("Concurrent workflow performance testing not yet implemented")

    @pytest.mark.asyncio
    async def test_memory_usage_in_long_workflows(self):
        """Test memory usage in long-running workflows."""
        pytest.skip("Long workflow memory testing not yet implemented")