"""Integration tests for ML workflow state handlers."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.workflow_handlers import (
    WorkflowRouter,
    parse_column_selection,
    parse_feature_selection
)
from src.core.state_manager import StateManager, MLTrainingState, WorkflowType


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35],
        'income': [30000, 45000, 55000],
        'sqft': [1000, 1500, 2000],
        'price': [200000, 300000, 400000]
    })


@pytest.fixture
def mock_update():
    """Create mock Telegram update object."""
    update = MagicMock()
    update.message.text = ""
    update.message.reply_text = AsyncMock()
    return update


@pytest.mark.asyncio
class TestWorkflowStateHandlers:
    """Test workflow state handlers."""

    async def test_target_selection_by_number(self, sample_dataframe, mock_update):
        """Test target selection using number input."""
        # Create fresh session
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12345, "chat_67890")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)

        mock_update.message.text = "4"  # Select "price" (4th column)

        router = WorkflowRouter(state_manager)
        await router.handle_target_selection(mock_update, None, session)

        assert session.selections.get("target_column") == "price"
        assert session.current_state == MLTrainingState.SELECTING_FEATURES.value

    async def test_target_selection_by_name(self, sample_dataframe, mock_update):
        """Test target selection using column name."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12346, "chat_67891")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)

        mock_update.message.text = "income"

        router = WorkflowRouter(state_manager)
        await router.handle_target_selection(mock_update, None, session)

        assert session.selections.get("target_column") == "income"
        assert session.current_state == MLTrainingState.SELECTING_FEATURES.value

    async def test_invalid_target_selection(self, sample_dataframe, mock_update):
        """Test invalid target selection shows error."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12347, "chat_67892")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)

        mock_update.message.text = "99"  # Invalid number

        router = WorkflowRouter(state_manager)
        await router.handle_target_selection(mock_update, None, session)

        # Should show error message
        mock_update.message.reply_text.assert_called_once()
        error_msg = mock_update.message.reply_text.call_args[0][0]
        assert "Invalid selection" in error_msg or "out of range" in error_msg

        # State should remain unchanged
        assert session.current_state == MLTrainingState.SELECTING_TARGET.value

    async def test_feature_selection_multiple_columns(self, sample_dataframe, mock_update):
        """Test feature selection with multiple columns."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12348, "chat_67893")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        # First transition to SELECTING_TARGET, then set target, then transition to SELECTING_FEATURES
        await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)
        session.selections["target_column"] = "price"
        await state_manager.transition_state(session, MLTrainingState.SELECTING_FEATURES.value)

        mock_update.message.text = "1,2,3"  # age, income, sqft

        router = WorkflowRouter(state_manager)
        await router.handle_feature_selection(mock_update, None, session)

        assert set(session.selections.get("feature_columns", [])) == {"age", "income", "sqft"}
        assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value

    async def test_feature_selection_range(self, sample_dataframe, mock_update):
        """Test feature selection with range notation."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12349, "chat_67894")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.selections["target_column"] = "price"
        await state_manager.transition_state(session, MLTrainingState.SELECTING_FEATURES.value)

        mock_update.message.text = "1-3"  # age, income, sqft

        router = WorkflowRouter(state_manager)
        await router.handle_feature_selection(mock_update, None, session)

        assert len(session.selections.get("feature_columns", [])) == 3

    async def test_feature_selection_all(self, sample_dataframe, mock_update):
        """Test feature selection with 'all' keyword."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12350, "chat_67895")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.selections["target_column"] = "price"
        await state_manager.transition_state(session, MLTrainingState.SELECTING_FEATURES.value)

        mock_update.message.text = "all"

        router = WorkflowRouter(state_manager)
        await router.handle_feature_selection(mock_update, None, session)

        # Should select all columns except target
        features = session.selections.get("feature_columns", [])
        assert len(features) == 3  # age, income, sqft (excluding price)
        assert "price" not in features

    async def test_model_type_selection_by_number(self, sample_dataframe, mock_update):
        """Test model type selection using number."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12351, "chat_67896")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        # Transition through states properly
        await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)
        session.selections["target_column"] = "price"
        await state_manager.transition_state(session, MLTrainingState.SELECTING_FEATURES.value)
        session.selections["feature_columns"] = ["age", "income"]
        await state_manager.transition_state(session, MLTrainingState.CONFIRMING_MODEL.value)

        mock_update.message.text = "2"  # Random Forest

        router = WorkflowRouter(state_manager)

        # Mock the execute_training method to avoid actual training
        with patch.object(router, 'execute_training', new_callable=AsyncMock):
            await router.handle_model_confirmation(mock_update, None, session)

        assert session.selections.get("model_type") == "random_forest"
        assert session.current_state == MLTrainingState.TRAINING.value

    async def test_model_type_selection_by_name(self, sample_dataframe, mock_update):
        """Test model type selection using name."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12352, "chat_67897")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
        session.selections["target_column"] = "price"
        session.selections["feature_columns"] = ["age", "income"]
        await state_manager.transition_state(session, MLTrainingState.CONFIRMING_MODEL.value)

        mock_update.message.text = "linear regression"

        router = WorkflowRouter(state_manager)

        with patch.object(router, 'execute_training', new_callable=AsyncMock):
            await router.handle_model_confirmation(mock_update, None, session)

        assert session.selections.get("model_type") == "linear_regression"

    async def test_workflow_cancellation(self, sample_dataframe, mock_update):
        """Test workflow cancellation."""
        state_manager = StateManager()
        session = await state_manager.get_or_create_session(12353, "chat_67898")
        await state_manager.store_data(session, sample_dataframe)
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)

        router = WorkflowRouter(state_manager)
        await router.cancel_workflow(mock_update, session)

        # Workflow should be cleared
        assert session.workflow_type is None
        assert session.current_state is None
        mock_update.message.reply_text.assert_called_once()


class TestColumnParsing:
    """Test column parsing utility functions."""

    def test_parse_column_by_number(self):
        """Test parsing column selection by number."""
        columns = ['age', 'income', 'price']
        assert parse_column_selection("2", columns) == "income"

    def test_parse_column_by_name(self):
        """Test parsing column selection by name."""
        columns = ['age', 'income', 'price']
        assert parse_column_selection("price", columns) == "price"

    def test_parse_column_case_insensitive(self):
        """Test case-insensitive column name matching."""
        columns = ['Age', 'Income', 'Price']
        assert parse_column_selection("age", columns) == "Age"

    def test_parse_column_invalid_number(self):
        """Test invalid number raises error."""
        columns = ['age', 'income', 'price']
        with pytest.raises(ValueError, match="out of range"):
            parse_column_selection("99", columns)

    def test_parse_column_invalid_name(self):
        """Test invalid column name raises error."""
        columns = ['age', 'income', 'price']
        with pytest.raises(ValueError, match="not found"):
            parse_column_selection("xyz", columns)

    def test_parse_features_multiple(self):
        """Test parsing multiple feature selections."""
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("1,3", features, "price")
        assert result == ['age', 'sqft']

    def test_parse_features_range(self):
        """Test parsing range of features."""
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("1-2", features, "price")
        assert result == ['age', 'income']

    def test_parse_features_all(self):
        """Test parsing 'all' keyword."""
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("all", features, "price")
        assert result == features

    def test_parse_features_by_name(self):
        """Test parsing features by name."""
        features = ['age', 'income', 'sqft']
        result = parse_feature_selection("age,income", features, "price")
        assert result == ['age', 'income']

    def test_parse_features_invalid_number(self):
        """Test invalid feature number raises error."""
        features = ['age', 'income', 'sqft']
        with pytest.raises(ValueError):
            parse_feature_selection("99", features, "price")


@pytest.mark.asyncio
async def test_complete_ml_workflow(sample_dataframe, mock_update):
    """Test complete ML training workflow from start to finish."""
    state_manager = StateManager()
    session = await state_manager.get_or_create_session(12354, "chat_67899")
    await state_manager.store_data(session, sample_dataframe)

    # Step 1: Initiate workflow
    await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)
    # Start workflow begins at AWAITING_DATA state, transition to SELECTING_TARGET
    await state_manager.transition_state(session, MLTrainingState.SELECTING_TARGET.value)
    assert session.current_state == MLTrainingState.SELECTING_TARGET.value

    # Step 2: Select target
    router = WorkflowRouter(state_manager)
    mock_update.message.text = "price"
    await router.handle_target_selection(mock_update, None, session)

    assert session.current_state == MLTrainingState.SELECTING_FEATURES.value

    # Step 3: Select features
    mock_update.message.text = "1,2"  # age, income
    await router.handle_feature_selection(mock_update, None, session)

    assert session.current_state == MLTrainingState.CONFIRMING_MODEL.value

    # Step 4: Select model
    mock_update.message.text = "auto"

    # Mock the execute_training to avoid actual execution
    with patch.object(router, 'execute_training', new_callable=AsyncMock):
        await router.handle_model_confirmation(mock_update, None, session)

    assert session.current_state == MLTrainingState.TRAINING.value
