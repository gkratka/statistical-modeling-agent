"""Integration tests for deferred loading workflow end-to-end."""

import pytest
import pandas as pd
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.state_manager import StateManager, MLTrainingState
from src.processors.data_loader import DataLoader
from src.bot.ml_handlers.ml_training_local_path import LocalPathMLTrainingHandler
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


@pytest.fixture
def test_csv_path(tmp_path):
    """Create a test CSV file with sufficient samples."""
    csv_file = tmp_path / "housing.csv"
    # Need at least 10 samples for ML training
    import numpy as np
    n_rows = 50
    df = pd.DataFrame({
        'price': np.random.randint(100000, 500000, n_rows),
        'sqft': np.random.randint(800, 2500, n_rows),
        'bedrooms': np.random.randint(1, 5, n_rows),
        'bathrooms': np.random.randint(1, 4, n_rows)
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def large_csv_path(tmp_path):
    """Create a large test CSV file (simulating 10M+ rows)."""
    csv_file = tmp_path / "housing_large.csv"
    # Create a larger dataset (10k rows to simulate large file)
    import numpy as np
    n_rows = 10000
    df = pd.DataFrame({
        'price': np.random.randint(100000, 500000, n_rows),
        'sqft': np.random.randint(800, 3000, n_rows),
        'bedrooms': np.random.randint(1, 6, n_rows),
        'bathrooms': np.random.randint(1, 4, n_rows),
        'lot_size': np.random.randint(5000, 20000, n_rows),
        'year_built': np.random.randint(1950, 2024, n_rows)
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def state_manager():
    """Create state manager instance."""
    return StateManager()


@pytest.fixture
def data_loader(tmp_path):
    """Create data loader with test configuration."""
    loader = DataLoader()
    # Allow tmp_path for testing
    loader.local_enabled = True
    loader.allowed_directories = [str(tmp_path)]
    return loader


@pytest.fixture
def ml_engine():
    """Create ML engine instance."""
    return MLEngine(MLEngineConfig.get_default())


@pytest.mark.asyncio
class TestDeferredLoadingWorkflow:
    """Test the complete deferred loading workflow end-to-end."""

    async def test_deferred_path_full_workflow(
        self,
        state_manager,
        data_loader,
        ml_engine,
        large_csv_path
    ):
        """
        Test full deferred loading workflow:
        1. User provides file path
        2. Path validated
        3. User chooses "Defer Loading"
        4. User provides manual schema
        5. Training starts with lazy loading from file path
        """
        # Setup
        user_id = 12345
        conversation_id = "test_conv"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        from src.core.state_manager import WorkflowType
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value

        # Step 1: Store file path
        session.file_path = large_csv_path
        await state_manager.update_session(session)

        # Step 2: Transition to choosing load option
        success, _, _ = await state_manager.transition_state(
            session,
            MLTrainingState.CHOOSING_LOAD_OPTION.value
        )
        assert success, "Should successfully transition to CHOOSING_LOAD_OPTION"

        # Step 3: User chooses defer loading
        session.load_deferred = True
        success, _, _ = await state_manager.transition_state(
            session,
            MLTrainingState.AWAITING_SCHEMA_INPUT.value
        )
        assert success, "Should successfully transition to AWAITING_SCHEMA_INPUT"

        # Step 4: User provides manual schema
        from src.utils.schema_parser import SchemaParser

        schema_input = "price, sqft, bedrooms, bathrooms, lot_size, year_built"
        parsed_schema = SchemaParser.parse(schema_input)

        session.manual_schema = {
            'target': parsed_schema.target,
            'features': parsed_schema.features,
            'format_detected': parsed_schema.format_detected
        }
        await state_manager.update_session(session)

        # Step 5: Verify schema parsed correctly
        assert session.manual_schema['target'] == 'price'
        assert len(session.manual_schema['features']) == 5
        assert 'sqft' in session.manual_schema['features']

        # Step 6: Skip target selection transition (not needed for direct ML engine test)

        # Step 7: Simulate training with lazy loading (file_path instead of data)
        result = ml_engine.train_model(
            file_path=session.file_path,  # Lazy loading!
            task_type="regression",
            model_type="linear",
            target_column=session.manual_schema['target'],
            feature_columns=session.manual_schema['features'],
            user_id=user_id,
            test_size=0.2
        )

        # Verify training succeeded with lazy loaded data
        assert 'model_id' in result
        assert 'metrics' in result
        assert result['metrics']['test']['mse'] is not None

    async def test_deferred_path_schema_formats(
        self,
        state_manager,
        data_loader,
        large_csv_path
    ):
        """Test all 3 schema input formats work correctly."""
        from src.utils.schema_parser import SchemaParser

        # Test Format 1: Key-Value
        schema_kv = """target: price
features: sqft, bedrooms, bathrooms"""
        parsed_kv = SchemaParser.parse(schema_kv)
        assert parsed_kv.target == "price"
        assert len(parsed_kv.features) == 3
        assert parsed_kv.format_detected == "key_value"

        # Test Format 2: JSON
        schema_json = '{"target": "price", "features": ["sqft", "bedrooms", "bathrooms"]}'
        parsed_json = SchemaParser.parse(schema_json)
        assert parsed_json.target == "price"
        assert len(parsed_json.features) == 3
        assert parsed_json.format_detected == "json"

        # Test Format 3: Simple List
        schema_list = "price, sqft, bedrooms, bathrooms"
        parsed_list = SchemaParser.parse(schema_list)
        assert parsed_list.target == "price"
        assert len(parsed_list.features) == 3
        assert parsed_list.format_detected == "simple_list"

    async def test_deferred_vs_immediate_comparison(
        self,
        state_manager,
        data_loader,
        ml_engine,
        test_csv_path
    ):
        """Compare deferred and immediate loading results - should be equivalent."""
        user_id = 12345

        # Immediate loading workflow
        df_immediate, _, schema_immediate = await data_loader.load_from_local_path(
            file_path=test_csv_path,
            detect_schema_flag=True
        )

        result_immediate = ml_engine.train_model(
            data=df_immediate,
            task_type="regression",
            model_type="linear",
            target_column="price",
            feature_columns=["sqft", "bedrooms", "bathrooms"],
            user_id=user_id
        )

        # Deferred loading workflow
        result_deferred = ml_engine.train_model(
            file_path=test_csv_path,  # Lazy loading
            task_type="regression",
            model_type="linear",
            target_column="price",
            feature_columns=["sqft", "bedrooms", "bathrooms"],
            user_id=user_id + 1  # Different user to avoid model ID conflict
        )

        # Both should succeed (check for model_id existence)
        assert 'model_id' in result_immediate
        assert 'model_id' in result_deferred

        # Metrics should be similar (not exactly equal due to randomness)
        assert abs(result_immediate['metrics']['test']['r2'] - result_deferred['metrics']['test']['r2']) < 0.3


@pytest.mark.asyncio
class TestImmediateLoadingBackwardCompatibility:
    """Test that immediate loading (existing workflow) still works."""

    async def test_immediate_path_workflow(
        self,
        state_manager,
        data_loader,
        ml_engine,
        test_csv_path
    ):
        """
        Test immediate loading workflow (backward compatibility):
        1. User provides file path
        2. User chooses "Load Now"
        3. Data loads immediately with schema detection
        4. Schema confirmation shown
        5. Training proceeds with pre-loaded data
        """
        # Setup
        user_id = 12345
        conversation_id = "test_conv"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        from src.core.state_manager import WorkflowType
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.AWAITING_FILE_PATH.value
        session.file_path = test_csv_path

        # User chooses immediate loading
        df, metadata, schema = await data_loader.load_from_local_path(
            file_path=test_csv_path,
            detect_schema_flag=True
        )

        session.uploaded_data = df
        session.load_deferred = False
        session.detected_schema = {
            'task_type': schema.suggested_task_type,
            'target': schema.suggested_target,
            'features': schema.suggested_features
        }

        # Transition to schema confirmation
        success, _, _ = await state_manager.transition_state(
            session,
            MLTrainingState.CONFIRMING_SCHEMA.value
        )
        assert success, "Should transition to CONFIRMING_SCHEMA"

        # User accepts schema - store target/features for transition prerequisite
        session.selections['target_column'] = schema.suggested_target
        session.selections['feature_columns'] = schema.suggested_features
        await state_manager.update_session(session)

        success, _, _ = await state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_TARGET.value
        )
        assert success, "Should transition to SELECTING_TARGET"

        # Training with pre-loaded data (backward compatible)
        result = ml_engine.train_model(
            data=session.uploaded_data,  # Pre-loaded data
            task_type="regression",
            model_type="linear",
            target_column=schema.suggested_target,
            feature_columns=schema.suggested_features,
            user_id=user_id
        )

        assert 'model_id' in result

    async def test_telegram_upload_workflow_unchanged(
        self,
        state_manager,
        ml_engine
    ):
        """Test that Telegram upload workflow is unchanged."""
        # Setup
        user_id = 12345
        conversation_id = "test_conv"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        from src.core.state_manager import WorkflowType
        session.workflow_type = WorkflowType.ML_TRAINING

        # User chooses Telegram upload (existing workflow)
        session.current_state = MLTrainingState.AWAITING_DATA.value
        session.data_source = "telegram"

        # Simulate uploaded data (need enough samples)
        import numpy as np
        df = pd.DataFrame({
            'price': np.random.randint(100000, 500000, 50),
            'sqft': np.random.randint(800, 2500, 50),
            'bedrooms': np.random.randint(1, 5, 50)
        })
        session.uploaded_data = df

        # Transition to target selection (legacy path)
        success, _, _ = await state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_TARGET.value
        )
        assert success, "Telegram upload workflow should work unchanged"

        # Training with uploaded data
        result = ml_engine.train_model(
            data=df,
            task_type="regression",
            model_type="linear",
            target_column="price",
            feature_columns=["sqft", "bedrooms"],
            user_id=user_id
        )

        assert 'model_id' in result


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in deferred loading workflow."""

    async def test_invalid_schema_format(self):
        """Test handling of invalid schema input."""
        from src.utils.schema_parser import SchemaParser
        from src.utils.exceptions import ValidationError

        invalid_inputs = [
            "",  # Empty
            "price",  # Only one column
            "target price features sqft",  # Ambiguous format
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValidationError):
                SchemaParser.parse(invalid_input)

    async def test_file_not_found_lazy_loading(self, ml_engine):
        """Test error when lazy loading file doesn't exist."""
        from src.utils.exceptions import DataValidationError

        with pytest.raises(DataValidationError, match="Failed to load data"):
            ml_engine.train_model(
                file_path="/nonexistent/path/data.csv",
                task_type="regression",
                model_type="linear",
                target_column="price",
                feature_columns=["sqft"],
                user_id=12345
            )

    async def test_missing_data_and_path(self, ml_engine):
        """Test error when neither data nor file_path provided."""
        from src.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Either 'data' or 'file_path' must be provided"):
            ml_engine.train_model(
                task_type="regression",
                model_type="linear",
                target_column="price",
                feature_columns=["sqft"],
                user_id=12345
            )
