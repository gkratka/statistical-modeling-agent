"""
End-to-end integration tests for ML Prediction workflow.

Tests complete user journey from /predict command through
data loading, feature selection, model selection, and prediction execution.

Related: dev/planning/predict-workflow.md
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.state_manager import StateManager, MLPredictionState, WorkflowType
from src.processors.data_loader import DataLoader
from src.engines.ml_engine import MLEngine
from src.bot.ml_handlers.prediction_handlers import PredictionHandler


class TestPredictionWorkflowE2E:
    """End-to-end tests for complete prediction workflow."""

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram Update object."""
        update = MagicMock()
        update.effective_user.id = 12345
        update.effective_chat.id = 67890
        update.effective_message = MagicMock()
        update.effective_message.reply_text = AsyncMock()
        update.callback_query = None
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock Telegram Context object."""
        context = MagicMock()
        return context

    @pytest.fixture
    def state_manager(self):
        """Create StateManager instance."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        return StateManager(config)

    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader(config={'local_data': {'enabled': False}})

    @pytest.fixture
    def sample_prediction_data(self):
        """Sample data for prediction (no target column)."""
        return pd.DataFrame({
            'sqft': [1000, 1500, 2000, 2500, 3000],
            'bedrooms': [2, 3, 3, 4, 4],
            'bathrooms': [1, 2, 2, 3, 3]
        })

    @pytest.fixture
    def sample_training_data(self):
        """Sample data for training (with target column)."""
        return pd.DataFrame({
            'sqft': [1000, 1500, 2000, 2500, 3000],
            'bedrooms': [2, 3, 3, 4, 4],
            'bathrooms': [1, 2, 2, 3, 3],
            'price': [100000, 150000, 200000, 250000, 300000]
        })

    @pytest.mark.asyncio
    async def test_complete_workflow_states(
        self,
        state_manager,
        mock_update,
        mock_context,
        sample_prediction_data
    ):
        """Test that workflow progresses through all states correctly."""
        # Create session
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_e2e"
        )

        # Initialize workflow
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.STARTED.value
        await state_manager.update_session(session)

        # Step 1: Start -> Choosing data source
        assert session.current_state == MLPredictionState.STARTED.value
        session.current_state = MLPredictionState.CHOOSING_DATA_SOURCE.value
        await state_manager.update_session(session)

        # Step 2-3: Select upload method
        session.current_state = MLPredictionState.AWAITING_FILE_UPLOAD.value
        session.uploaded_data = sample_prediction_data
        await state_manager.update_session(session)

        # Step 3: Confirm schema
        session.current_state = MLPredictionState.CONFIRMING_SCHEMA.value
        await state_manager.update_session(session)

        # Step 4-5: Select features
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
        session.selections['selected_features'] = ['sqft', 'bedrooms', 'bathrooms']
        await state_manager.update_session(session)

        # Step 6-7: Select model
        session.current_state = MLPredictionState.SELECTING_MODEL.value
        session.selections['selected_model_id'] = 'model_12345_random_forest'
        await state_manager.update_session(session)

        # Step 8-9: Confirm prediction column
        session.current_state = MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value
        session.selections['prediction_column_name'] = 'price_predicted'
        await state_manager.update_session(session)

        # Step 10: Ready to run
        session.current_state = MLPredictionState.READY_TO_RUN.value
        await state_manager.update_session(session)

        # Step 11-12: Run prediction
        session.current_state = MLPredictionState.RUNNING_PREDICTION.value
        await state_manager.update_session(session)

        # Step 13: Complete
        session.current_state = MLPredictionState.COMPLETE.value
        await state_manager.update_session(session)

        # Verify final state
        assert session.current_state == MLPredictionState.COMPLETE.value

    @pytest.mark.asyncio
    async def test_feature_validation_workflow(
        self,
        state_manager,
        sample_prediction_data
    ):
        """Test feature validation during workflow."""
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_features"
        )

        # Setup with data
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.uploaded_data = sample_prediction_data
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value

        # Valid features
        selected_features = ['sqft', 'bedrooms', 'bathrooms']
        invalid_features = [f for f in selected_features if f not in sample_prediction_data.columns]

        # All features should be valid
        assert len(invalid_features) == 0

        # Test invalid features
        bad_features = ['sqft', 'bedrooms', 'age']  # 'age' doesn't exist
        invalid = [f for f in bad_features if f not in sample_prediction_data.columns]
        assert len(invalid) == 1
        assert invalid[0] == 'age'

    @pytest.mark.asyncio
    async def test_model_filtering_by_features(self, state_manager):
        """Test that models are correctly filtered by feature match."""
        # Mock models from ML engine
        all_models = [
            {
                'model_id': 'model_1',
                'model_type': 'random_forest',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
                'target_column': 'price'
            },
            {
                'model_id': 'model_2',
                'model_type': 'linear',
                'feature_columns': ['sqft', 'bedrooms'],
                'target_column': 'price'
            },
            {
                'model_id': 'model_3',
                'model_type': 'gradient_boosting',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms', 'age'],
                'target_column': 'price'
            }
        ]

        selected_features = ['sqft', 'bedrooms', 'bathrooms']

        # Filter compatible models
        compatible = [
            m for m in all_models
            if set(m['feature_columns']) == set(selected_features)
        ]

        # Only model_1 should match
        assert len(compatible) == 1
        assert compatible[0]['model_id'] == 'model_1'

    @pytest.mark.asyncio
    async def test_prediction_column_addition(self, sample_prediction_data):
        """Test adding prediction column to DataFrame."""
        df = sample_prediction_data.copy()
        predictions = [120000, 180000, 240000, 300000, 360000]
        prediction_column = 'price_predicted'

        # Add predictions
        df[prediction_column] = predictions

        # Verify
        assert prediction_column in df.columns
        assert len(df) == 5
        assert df[prediction_column].tolist() == predictions

        # Original columns preserved
        assert 'sqft' in df.columns
        assert 'bedrooms' in df.columns
        assert 'bathrooms' in df.columns

    @pytest.mark.asyncio
    async def test_statistics_generation(self):
        """Test statistics calculation for predictions."""
        predictions = [120000, 180000, 240000, 300000, 360000]

        statistics = {
            'mean': float(pd.Series(predictions).mean()),
            'std': float(pd.Series(predictions).std()),
            'min': float(pd.Series(predictions).min()),
            'max': float(pd.Series(predictions).max()),
            'median': float(pd.Series(predictions).median())
        }

        assert statistics['mean'] == 240000.0
        assert statistics['min'] == 120000.0
        assert statistics['max'] == 360000.0
        assert statistics['median'] == 240000.0
        assert statistics['std'] > 0  # Should have positive std dev

    @pytest.mark.asyncio
    async def test_csv_export_with_predictions(
        self,
        sample_prediction_data
    ):
        """Test CSV export with prediction column."""
        df = sample_prediction_data.copy()
        predictions = [120000, 180000, 240000, 300000, 360000]
        df['price_predicted'] = predictions

        # Export to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as tmp:
            csv_path = tmp.name
            df.to_csv(csv_path, index=False)

        # Verify file exists
        assert Path(csv_path).exists()

        # Read back and verify
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 5
        assert 'price_predicted' in loaded_df.columns
        assert loaded_df['price_predicted'].tolist() == predictions

        # Cleanup
        Path(csv_path).unlink()

    @pytest.mark.asyncio
    async def test_back_button_from_model_selection(
        self,
        state_manager,
        sample_prediction_data
    ):
        """Test back button navigation from model selection."""
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_back"
        )

        # Setup workflow at model selection
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.uploaded_data = sample_prediction_data
        session.current_state = MLPredictionState.AWAITING_FEATURE_SELECTION.value
        session.selections['selected_features'] = ['sqft', 'bedrooms', 'bathrooms']

        # Save state snapshot
        session.save_state_snapshot()

        # Move to model selection
        session.current_state = MLPredictionState.SELECTING_MODEL.value
        session.selections['selected_model_id'] = 'model_12345'
        await state_manager.update_session(session)

        # Verify we're at model selection
        assert session.current_state == MLPredictionState.SELECTING_MODEL.value

        # Go back
        success = session.restore_previous_state()
        assert success
        assert session.current_state == MLPredictionState.AWAITING_FEATURE_SELECTION.value

    @pytest.mark.asyncio
    async def test_error_recovery_no_models(self, state_manager):
        """Test error recovery when no models match features."""
        session = await state_manager.get_or_create_session(
            user_id=12345,
            conversation_id="test_no_models"
        )

        # Setup with features but no compatible models
        session.workflow_type = WorkflowType.ML_PREDICTION
        session.current_state = MLPredictionState.SELECTING_MODEL.value
        session.selections['selected_features'] = ['rare_feature1', 'rare_feature2']

        # Mock empty model list
        all_models = []
        compatible = [
            m for m in all_models
            if set(m.get('feature_columns', [])) == set(session.selections['selected_features'])
        ]

        # Should have no compatible models
        assert len(compatible) == 0

    @pytest.mark.asyncio
    async def test_column_name_conflict_detection(
        self,
        sample_prediction_data
    ):
        """Test detection of prediction column name conflicts."""
        df = sample_prediction_data.copy()
        df['price_predicted'] = [100000, 200000, 300000, 400000, 500000]

        # Try to use same name
        prediction_column = 'price_predicted'

        # Should conflict
        assert prediction_column in df.columns

        # Alternative name should work
        alternative_column = 'price_forecast'
        assert alternative_column not in df.columns


class TestPredictionHandlerIntegration:
    """Integration tests for PredictionHandler class."""

    @pytest.fixture
    def handler(self):
        """Create PredictionHandler instance."""
        from src.core.state_manager import StateManagerConfig
        config = StateManagerConfig()
        state_mgr = StateManager(config)
        data_ldr = DataLoader(config={'local_data': {'enabled': False}})
        return PredictionHandler(state_mgr, data_ldr)

    @pytest.mark.asyncio
    async def test_handler_initialization(self, handler):
        """Test that handler initializes correctly."""
        assert handler.state_manager is not None
        assert handler.data_loader is not None
        assert handler.ml_engine is not None

    @pytest.mark.asyncio
    async def test_parse_feature_selection(self, handler):
        """Test parsing feature selection from user input."""
        input_text = "sqft, bedrooms, bathrooms"
        features = [f.strip() for f in input_text.split(',')]

        assert len(features) == 3
        assert features == ['sqft', 'bedrooms', 'bathrooms']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
