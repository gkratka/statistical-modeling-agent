"""
Unit tests for ML Prediction state machine and transitions.

Tests:
1. State transition validation
2. Prerequisites checking
3. Feature validation logic
4. Column name validation
5. Model filtering by features

Related: dev/planning/predict-workflow.md
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.core.state_manager import (
    StateManager,
    MLPredictionState,
    WorkflowType,
    ML_PREDICTION_TRANSITIONS,
    ML_PREDICTION_PREREQUISITES
)


class TestPredictionStateTransitions:
    """Test state transition rules for prediction workflow."""

    def test_transitions_map_completeness(self):
        """Verify all MLPredictionState values are covered in transitions."""
        all_states = {state.value for state in MLPredictionState}

        # States that appear as sources in transition map
        source_states = set(ML_PREDICTION_TRANSITIONS.keys())
        source_states.discard(None)  # Remove None (initial state)

        # States that appear as targets in transition map
        target_states = set()
        for targets in ML_PREDICTION_TRANSITIONS.values():
            target_states.update(targets)

        # All states should appear somewhere in transitions
        all_covered = source_states.union(target_states)
        assert all_states == all_covered, f"Missing states: {all_states - all_covered}"

    def test_start_transition(self):
        """Test initial transition from None to STARTED."""
        valid_starts = ML_PREDICTION_TRANSITIONS[None]
        assert MLPredictionState.STARTED.value in valid_starts
        assert len(valid_starts) == 1  # Only one entry point

    def test_data_source_selection_transitions(self):
        """Test transitions from CHOOSING_DATA_SOURCE."""
        valid_next = ML_PREDICTION_TRANSITIONS[MLPredictionState.CHOOSING_DATA_SOURCE.value]

        # Can go to either upload or local path
        assert MLPredictionState.AWAITING_FILE_UPLOAD.value in valid_next
        assert MLPredictionState.AWAITING_FILE_PATH.value in valid_next
        assert len(valid_next) == 2

    def test_schema_confirmation_transitions(self):
        """Test transitions from CONFIRMING_SCHEMA."""
        valid_next = ML_PREDICTION_TRANSITIONS[MLPredictionState.CONFIRMING_SCHEMA.value]

        # Can proceed to features or go back to data source
        assert MLPredictionState.AWAITING_FEATURE_SELECTION.value in valid_next
        assert MLPredictionState.CHOOSING_DATA_SOURCE.value in valid_next
        assert len(valid_next) == 2

    def test_ready_to_run_transitions(self):
        """Test transitions from READY_TO_RUN."""
        valid_next = ML_PREDICTION_TRANSITIONS[MLPredictionState.READY_TO_RUN.value]

        # Can run model or go back to model selection
        assert MLPredictionState.RUNNING_PREDICTION.value in valid_next
        assert MLPredictionState.SELECTING_MODEL.value in valid_next
        assert len(valid_next) == 2

    def test_complete_state_no_transitions(self):
        """Test that COMPLETE state has no outgoing transitions."""
        valid_next = ML_PREDICTION_TRANSITIONS[MLPredictionState.COMPLETE.value]
        assert len(valid_next) == 0  # Terminal state


class TestPredictionPrerequisites:
    """Test prerequisite checking for prediction workflow."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock UserSession for testing."""
        session = MagicMock()
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.STARTED.value
        session.uploaded_data = None
        session.file_path = None
        session.selections = {}
        return session

    def test_confirming_schema_requires_data(self, mock_session):
        """Test that CONFIRMING_SCHEMA requires data to be loaded."""
        prerequisite = ML_PREDICTION_PREREQUISITES[MLPredictionState.CONFIRMING_SCHEMA.value]

        # No data - should fail
        assert not prerequisite(mock_session)

        # Data uploaded - should pass
        mock_session.uploaded_data = pd.DataFrame({'a': [1, 2, 3]})
        assert prerequisite(mock_session)

        # File path set - should pass
        mock_session.uploaded_data = None
        mock_session.file_path = "/path/to/data.csv"
        assert prerequisite(mock_session)

    def test_selecting_model_requires_features(self, mock_session):
        """Test that SELECTING_MODEL requires features to be selected."""
        prerequisite = ML_PREDICTION_PREREQUISITES[MLPredictionState.SELECTING_MODEL.value]

        # No features - should fail
        assert not prerequisite(mock_session)

        # Empty features - should fail
        mock_session.selections['selected_features'] = []
        assert not prerequisite(mock_session)

        # Features selected - should pass
        mock_session.selections['selected_features'] = ['feature1', 'feature2']
        assert prerequisite(mock_session)

    def test_confirming_column_requires_model(self, mock_session):
        """Test that CONFIRMING_PREDICTION_COLUMN requires model selection."""
        prerequisite = ML_PREDICTION_PREREQUISITES[MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value]

        # No model - should fail
        assert not prerequisite(mock_session)

        # Model selected - should pass
        mock_session.selections['selected_model_id'] = 'model_12345_random_forest'
        assert prerequisite(mock_session)

    def test_ready_to_run_requires_column_name(self, mock_session):
        """Test that READY_TO_RUN requires prediction column name."""
        prerequisite = ML_PREDICTION_PREREQUISITES[MLPredictionState.READY_TO_RUN.value]

        # No column name - should fail
        assert not prerequisite(mock_session)

        # Column name set - should pass
        mock_session.selections['prediction_column_name'] = 'price_predicted'
        assert prerequisite(mock_session)

    def test_running_prediction_requires_all(self, mock_session):
        """Test that RUNNING_PREDICTION requires all prerequisites."""
        prerequisite = ML_PREDICTION_PREREQUISITES[MLPredictionState.RUNNING_PREDICTION.value]

        # No data - should fail
        assert not prerequisite(mock_session)

        # Add data
        mock_session.uploaded_data = pd.DataFrame({'a': [1, 2, 3]})
        assert not prerequisite(mock_session)  # Still missing other requirements

        # Add model
        mock_session.selections['selected_model_id'] = 'model_12345_random_forest'
        assert not prerequisite(mock_session)  # Still missing features and column

        # Add features
        mock_session.selections['selected_features'] = ['feature1', 'feature2']
        assert not prerequisite(mock_session)  # Still missing column name

        # Add column name - should now pass
        mock_session.selections['prediction_column_name'] = 'price_predicted'
        assert prerequisite(mock_session)


class TestFeatureValidation:
    """Test feature validation logic for model compatibility."""

    def test_exact_feature_match(self):
        """Test that features must match exactly (set equality)."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Exact match
        assert set(model_features) == set(selected_features)

    def test_feature_order_independence(self):
        """Test that feature order doesn't matter."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['bathrooms', 'sqft', 'bedrooms']

        # Different order, same features
        assert set(model_features) == set(selected_features)

    def test_missing_features_detected(self):
        """Test detection of missing features."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['sqft', 'bedrooms']  # Missing bathrooms

        # Not a match
        assert set(model_features) != set(selected_features)

        # Calculate missing
        missing = set(model_features) - set(selected_features)
        assert missing == {'bathrooms'}

    def test_extra_features_detected(self):
        """Test detection of extra features."""
        model_features = ['sqft', 'bedrooms']
        selected_features = ['sqft', 'bedrooms', 'bathrooms']  # Extra bathrooms

        # Not a match
        assert set(model_features) != set(selected_features)

        # Calculate extra
        extra = set(selected_features) - set(model_features)
        assert extra == {'bathrooms'}

    def test_completely_different_features(self):
        """Test detection of completely different feature sets."""
        model_features = ['sqft', 'bedrooms', 'bathrooms']
        selected_features = ['age', 'income', 'credit_score']

        # No match
        assert set(model_features) != set(selected_features)

        # Calculate differences
        missing = set(model_features) - set(selected_features)
        extra = set(selected_features) - set(model_features)

        assert missing == {'sqft', 'bedrooms', 'bathrooms'}
        assert extra == {'age', 'income', 'credit_score'}


class TestColumnNameValidation:
    """Test prediction column name validation."""

    def test_column_name_not_in_dataframe(self):
        """Test that prediction column name doesn't conflict with existing columns."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2]
        })

        prediction_column = 'price_predicted'

        # Should not conflict
        assert prediction_column not in df.columns

    def test_column_name_conflict_detection(self):
        """Test detection of column name conflicts."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4],
            'price_predicted': [100000, 200000, 300000]  # Already exists
        })

        prediction_column = 'price_predicted'

        # Should conflict
        assert prediction_column in df.columns

    def test_default_column_name_generation(self):
        """Test default column name generation from target."""
        target_column = 'price'
        default_prediction_column = f'{target_column}_predicted'

        assert default_prediction_column == 'price_predicted'

    def test_custom_column_name_allowed(self):
        """Test that custom column names are allowed."""
        df = pd.DataFrame({
            'sqft': [1000, 2000, 3000],
            'bedrooms': [2, 3, 4]
        })

        custom_names = [
            'price_forecast',
            'predicted_value',
            'model_output',
            'my_prediction'
        ]

        # All custom names should be valid
        for name in custom_names:
            assert name not in df.columns


class TestModelFiltering:
    """Test model filtering logic based on features."""

    def test_filter_models_by_features(self):
        """Test filtering models that match selected features."""
        # Mock models from ML engine
        all_models = [
            {
                'model_id': 'model_1',
                'model_type': 'random_forest',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms']
            },
            {
                'model_id': 'model_2',
                'model_type': 'linear',
                'feature_columns': ['sqft', 'bedrooms']
            },
            {
                'model_id': 'model_3',
                'model_type': 'gradient_boosting',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms', 'age']
            }
        ]

        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Filter compatible models
        compatible_models = [
            m for m in all_models
            if set(m['feature_columns']) == set(selected_features)
        ]

        # Only model_1 should match
        assert len(compatible_models) == 1
        assert compatible_models[0]['model_id'] == 'model_1'

    def test_no_compatible_models(self):
        """Test case where no models match selected features."""
        all_models = [
            {
                'model_id': 'model_1',
                'model_type': 'random_forest',
                'feature_columns': ['sqft', 'bedrooms']
            },
            {
                'model_id': 'model_2',
                'model_type': 'linear',
                'feature_columns': ['age', 'income']
            }
        ]

        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Filter compatible models
        compatible_models = [
            m for m in all_models
            if set(m['feature_columns']) == set(selected_features)
        ]

        # No models should match
        assert len(compatible_models) == 0

    def test_multiple_compatible_models(self):
        """Test case where multiple models match selected features."""
        all_models = [
            {
                'model_id': 'model_1',
                'model_type': 'random_forest',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms']
            },
            {
                'model_id': 'model_2',
                'model_type': 'gradient_boosting',
                'feature_columns': ['sqft', 'bathrooms', 'bedrooms']  # Different order
            },
            {
                'model_id': 'model_3',
                'model_type': 'linear',
                'feature_columns': ['sqft', 'bedrooms']  # Different features
            }
        ]

        selected_features = ['bathrooms', 'sqft', 'bedrooms']  # Different order

        # Filter compatible models
        compatible_models = [
            m for m in all_models
            if set(m['feature_columns']) == set(selected_features)
        ]

        # model_1 and model_2 should match
        assert len(compatible_models) == 2
        assert compatible_models[0]['model_id'] == 'model_1'
        assert compatible_models[1]['model_id'] == 'model_2'


class TestPredictionWorkflowIntegration:
    """Integration tests for state machine with StateManager."""

    @pytest.mark.asyncio
    async def test_complete_workflow_state_progression(self):
        """Test complete workflow from start to finish."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        # Create session
        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_prediction"
        )

        # Initialize prediction workflow
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.STARTED.value
        await manager.update_session(session)

        # Verify initial state
        assert session.current_state == MLPredictionState.STARTED.value

        # Progress through workflow
        states_sequence = [
            MLPredictionState.CHOOSING_DATA_SOURCE.value,
            MLPredictionState.AWAITING_FILE_PATH.value,
            MLPredictionState.CONFIRMING_SCHEMA.value,
            MLPredictionState.AWAITING_FEATURE_SELECTION.value,
            MLPredictionState.SELECTING_MODEL.value,
            MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value,
            MLPredictionState.READY_TO_RUN.value,
            MLPredictionState.RUNNING_PREDICTION.value,
            MLPredictionState.COMPLETE.value
        ]

        for next_state in states_sequence:
            # Verify transition is valid
            valid_transitions = ML_PREDICTION_TRANSITIONS[session.current_state]
            assert next_state in valid_transitions, \
                f"Invalid transition from {session.current_state} to {next_state}"

            # Update state
            session.current_state = next_state
            await manager.update_session(session)

            # Verify state updated
            assert session.current_state == next_state

    @pytest.mark.asyncio
    async def test_back_navigation_with_state_history(self):
        """Test back button navigation clears appropriate state."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        manager = StateManager(config)

        session = await manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_back_nav"
        )

        # Setup workflow with data
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.STARTED.value
        session.uploaded_data = pd.DataFrame({'a': [1, 2, 3]})
        session.selections['selected_features'] = ['feature1', 'feature2']
        session.selections['selected_model_id'] = 'model_12345'

        # Save snapshot
        session.save_state_snapshot()

        # Move to next state
        session.current_state = MLPredictionState.SELECTING_MODEL.value
        await manager.update_session(session)

        # Verify we have history
        assert session.can_go_back()

        # Go back
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLPredictionState.STARTED.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
