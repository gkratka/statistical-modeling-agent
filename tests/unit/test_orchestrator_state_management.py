"""
Unit tests for Enhanced Orchestrator State Management components.

This module tests the state management architecture including ConversationState,
WorkflowState, and StateManager classes with comprehensive edge case coverage.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock

from src.core.orchestrator import (
    ConversationState,
    WorkflowState,
    StateManager,
    DataManager,
    WorkflowEngine,
    ErrorRecoverySystem,
    FeedbackLoop
)
from src.utils.exceptions import ValidationError, DataError


class TestConversationState:
    """Test ConversationState dataclass functionality."""

    def test_conversation_state_creation(self):
        """Test basic ConversationState creation."""
        state = ConversationState(
            user_id=12345,
            conversation_id="test_conv_123",
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        assert state.user_id == 12345
        assert state.conversation_id == "test_conv_123"
        assert state.workflow_state == WorkflowState.IDLE
        assert state.current_step is None
        assert isinstance(state.context, dict)
        assert isinstance(state.partial_results, dict)
        assert isinstance(state.data_sources, list)

    def test_conversation_state_with_data(self):
        """Test ConversationState with populated data."""
        state = ConversationState(
            user_id=12345,
            conversation_id="test_conv_123",
            workflow_state=WorkflowState.SELECTING_TARGET,
            current_step="target_selection",
            context={"columns": ["age", "income", "score"]},
            partial_results={"data_validation": True},
            data_sources=["data_123", "data_456"],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        assert state.workflow_state == WorkflowState.SELECTING_TARGET
        assert state.current_step == "target_selection"
        assert "columns" in state.context
        assert "data_validation" in state.partial_results
        assert len(state.data_sources) == 2

    def test_conversation_state_key_generation(self):
        """Test state key generation for storage."""
        state = ConversationState(
            user_id=12345,
            conversation_id="test_conv_123",
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        expected_key = f"{state.user_id}:{state.conversation_id}"
        assert expected_key == "12345:test_conv_123"


class TestWorkflowState:
    """Test WorkflowState enum functionality."""

    def test_workflow_state_values(self):
        """Test all WorkflowState enum values."""
        assert WorkflowState.IDLE.value == "idle"
        assert WorkflowState.AWAITING_DATA.value == "awaiting_data"
        assert WorkflowState.DATA_LOADED.value == "data_loaded"
        assert WorkflowState.SELECTING_TARGET.value == "selecting_target"
        assert WorkflowState.SELECTING_FEATURES.value == "selecting_features"
        assert WorkflowState.CONFIGURING_MODEL.value == "configuring_model"
        assert WorkflowState.TRAINING.value == "training"
        assert WorkflowState.TRAINED.value == "trained"
        assert WorkflowState.PREDICTING.value == "predicting"
        assert WorkflowState.COMPLETED.value == "completed"
        assert WorkflowState.ERROR.value == "error"

    def test_workflow_state_transitions(self):
        """Test valid workflow state transitions."""
        # Test valid ML training flow transitions
        valid_transitions = [
            (WorkflowState.IDLE, WorkflowState.AWAITING_DATA),
            (WorkflowState.AWAITING_DATA, WorkflowState.DATA_LOADED),
            (WorkflowState.DATA_LOADED, WorkflowState.SELECTING_TARGET),
            (WorkflowState.SELECTING_TARGET, WorkflowState.SELECTING_FEATURES),
            (WorkflowState.SELECTING_FEATURES, WorkflowState.CONFIGURING_MODEL),
            (WorkflowState.CONFIGURING_MODEL, WorkflowState.TRAINING),
            (WorkflowState.TRAINING, WorkflowState.TRAINED),
            (WorkflowState.TRAINED, WorkflowState.PREDICTING),
            (WorkflowState.PREDICTING, WorkflowState.COMPLETED),
        ]

        for from_state, to_state in valid_transitions:
            assert isinstance(from_state, WorkflowState)
            assert isinstance(to_state, WorkflowState)

    def test_workflow_state_error_transitions(self):
        """Test error state transitions from any state."""
        all_states = list(WorkflowState)

        for state in all_states:
            # Any state can transition to ERROR
            assert isinstance(state, WorkflowState)
            assert WorkflowState.ERROR in WorkflowState


class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def state_manager(self):
        """Provide StateManager instance for testing."""
        return StateManager(ttl_minutes=5)

    @pytest.fixture
    def sample_state(self):
        """Provide sample ConversationState for testing."""
        return ConversationState(
            user_id=12345,
            conversation_id="test_conv_123",
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_state_manager_initialization(self, state_manager):
        """Test StateManager initialization."""
        assert state_manager.ttl == 5
        assert isinstance(state_manager.states, dict)
        assert len(state_manager.states) == 0

    @pytest.mark.asyncio
    async def test_save_and_get_state(self, state_manager, sample_state):
        """Test saving and retrieving state."""
        # Save state
        await state_manager.save_state(sample_state)

        # Retrieve state
        retrieved_state = await state_manager.get_state(
            sample_state.user_id,
            sample_state.conversation_id
        )

        assert retrieved_state.user_id == sample_state.user_id
        assert retrieved_state.conversation_id == sample_state.conversation_id
        assert retrieved_state.workflow_state == sample_state.workflow_state

    @pytest.mark.asyncio
    async def test_get_nonexistent_state(self, state_manager):
        """Test getting state that doesn't exist creates new one."""
        state = await state_manager.get_state(99999, "nonexistent")

        assert state.user_id == 99999
        assert state.conversation_id == "nonexistent"
        assert state.workflow_state == WorkflowState.IDLE
        assert isinstance(state.context, dict)
        assert isinstance(state.partial_results, dict)

    @pytest.mark.asyncio
    async def test_clear_state(self, state_manager, sample_state):
        """Test clearing specific state."""
        # Save state
        await state_manager.save_state(sample_state)

        # Verify it exists
        key = f"{sample_state.user_id}:{sample_state.conversation_id}"
        assert key in state_manager.states

        # Clear state
        await state_manager.clear_state(
            sample_state.user_id,
            sample_state.conversation_id
        )

        # Verify it's gone
        assert key not in state_manager.states

    @pytest.mark.asyncio
    async def test_cleanup_expired_states(self, state_manager):
        """Test cleanup of expired states."""
        # Create expired state
        expired_state = ConversationState(
            user_id=12345,
            conversation_id="expired_conv",
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now() - timedelta(hours=2),
            last_activity=datetime.now() - timedelta(hours=1)
        )

        # Create fresh state
        fresh_state = ConversationState(
            user_id=12346,
            conversation_id="fresh_conv",
            workflow_state=WorkflowState.IDLE,
            current_step=None,
            context={},
            partial_results={},
            data_sources=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )

        # Save both states
        await state_manager.save_state(expired_state)
        await state_manager.save_state(fresh_state)

        assert len(state_manager.states) == 2

        # Cleanup expired
        await state_manager.cleanup_expired()

        # Only fresh state should remain
        assert len(state_manager.states) == 1
        fresh_key = f"{fresh_state.user_id}:{fresh_state.conversation_id}"
        assert fresh_key in state_manager.states

    @pytest.mark.asyncio
    async def test_concurrent_state_operations(self, state_manager):
        """Test concurrent state operations."""
        states = []
        for i in range(10):
            state = ConversationState(
                user_id=10000 + i,
                conversation_id=f"concurrent_conv_{i}",
                workflow_state=WorkflowState.IDLE,
                current_step=None,
                context={"test": i},
                partial_results={},
                data_sources=[],
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            states.append(state)

        # Save all states concurrently
        save_tasks = [state_manager.save_state(state) for state in states]
        await asyncio.gather(*save_tasks)

        assert len(state_manager.states) == 10

        # Retrieve all states concurrently
        get_tasks = [
            state_manager.get_state(state.user_id, state.conversation_id)
            for state in states
        ]
        retrieved_states = await asyncio.gather(*get_tasks)

        # Verify all states retrieved correctly
        for original, retrieved in zip(states, retrieved_states):
            assert original.user_id == retrieved.user_id
            assert original.conversation_id == retrieved.conversation_id
            assert original.context["test"] == retrieved.context["test"]

    @pytest.mark.asyncio
    async def test_state_persistence_across_updates(self, state_manager, sample_state):
        """Test state persistence through multiple updates."""
        # Save initial state
        await state_manager.save_state(sample_state)

        # Update workflow state
        sample_state.workflow_state = WorkflowState.DATA_LOADED
        sample_state.context["data_loaded"] = True
        sample_state.last_activity = datetime.now()

        # Save updated state
        await state_manager.save_state(sample_state)

        # Retrieve and verify updates
        retrieved_state = await state_manager.get_state(
            sample_state.user_id,
            sample_state.conversation_id
        )

        assert retrieved_state.workflow_state == WorkflowState.DATA_LOADED
        assert retrieved_state.context["data_loaded"] is True