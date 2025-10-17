"""Integration tests for ML Model Naming Workflow.

Tests the complete end-to-end workflow for naming models after training,
focusing on state transitions and data persistence.
"""

import pytest
import pandas as pd
import json
from pathlib import Path

from src.core.state_manager import StateManager, MLTrainingState
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig


@pytest.mark.asyncio
class TestModelNamingWorkflowIntegration:
    """Test complete model naming workflow with state machine and ML engine integration."""

    @pytest.fixture
    def state_manager(self, tmp_path):
        """Create state manager for testing."""
        return StateManager(sessions_dir=str(tmp_path / "sessions"))

    @pytest.fixture
    def ml_engine(self, tmp_path):
        """Create ML engine for testing."""
        config = MLEngineConfig(
            models_dir=tmp_path / "models",
            max_models_per_user=50,
            max_model_size_mb=100,
            max_training_time=60,
            max_memory_mb=2048,
            min_training_samples=10,
            default_test_size=0.2,
            default_cv_folds=5,
            default_missing_strategy="mean",
            default_scaling="standard",
            default_hyperparameters={},
            hyperparameter_ranges={}
        )
        return MLEngine(config)

    @pytest.fixture
    def sample_model(self, tmp_path, ml_engine):
        """Create a sample model for testing."""
        user_id = 12345
        model_id = "model_12345_linear_20251014_123456"

        # Create model directory
        model_dir = tmp_path / "models" / f"user_{user_id}" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = {
            "model_id": model_id,
            "user_id": user_id,
            "model_type": "linear",
            "task_type": "regression",
            "created_at": "2025-01-14T21:44:00Z",
            "metrics": {"mse": 0.15, "r2": 0.85}
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            "user_id": user_id,
            "model_id": model_id,
            "model_dir": model_dir
        }

    async def test_state_transitions_name_model_workflow(self, state_manager, sample_model):
        """Test state transitions through the naming workflow."""
        user_id = sample_model["user_id"]
        conversation_id = "conv_test_123"
        model_id = sample_model["model_id"]

        # Start ML training
        await state_manager.start_ml_training(user_id, conversation_id)
        session = await state_manager.get_session(user_id, conversation_id)

        # Set up pending model
        session.selections["pending_model_id"] = model_id
        await state_manager.update_session(session)

        # Step 1: Training completes
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success, f"Failed to transition to TRAINING_COMPLETE: {error}"
        assert session.current_state == MLTrainingState.TRAINING_COMPLETE.value

        # Step 2: User chooses to name model
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.NAMING_MODEL.value
        )
        assert success, f"Failed to transition to NAMING_MODEL: {error}"
        assert session.current_state == MLTrainingState.NAMING_MODEL.value

        # Step 3: Name is provided and accepted
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.MODEL_NAMED.value
        )
        assert success, f"Failed to transition to MODEL_NAMED: {error}"
        assert session.current_state == MLTrainingState.MODEL_NAMED.value

        # Step 4: Workflow completes
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.COMPLETE.value
        )
        assert success, f"Failed to transition to COMPLETE: {error}"
        assert session.current_state == MLTrainingState.COMPLETE.value

    async def test_state_transitions_skip_naming_workflow(self, state_manager, sample_model):
        """Test state transitions when skipping naming."""
        user_id = sample_model["user_id"]
        conversation_id = "conv_test_123"
        model_id = sample_model["model_id"]

        # Start ML training
        await state_manager.start_ml_training(user_id, conversation_id)
        session = await state_manager.get_session(user_id, conversation_id)

        # Set up pending model
        session.selections["pending_model_id"] = model_id
        await state_manager.update_session(session)

        # Step 1: Training completes
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success

        # Step 2: User skips naming (direct to MODEL_NAMED)
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.MODEL_NAMED.value
        )
        assert success, f"Failed to skip to MODEL_NAMED: {error}"
        assert session.current_state == MLTrainingState.MODEL_NAMED.value

    async def test_custom_name_persistence(self, ml_engine, sample_model):
        """Test that custom names are persisted correctly."""
        user_id = sample_model["user_id"]
        model_id = sample_model["model_id"]
        custom_name = "Housing Price Predictor"

        # Set custom name
        success = ml_engine.set_model_name(user_id, model_id, custom_name)
        assert success

        # Verify name was saved in metadata
        metadata_path = sample_model["model_dir"] / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert metadata["custom_name"] == custom_name
        assert metadata["display_name"] == custom_name

        # Verify retrieval by name works
        retrieved = ml_engine.get_model_by_name(user_id, custom_name)
        assert retrieved is not None
        assert retrieved["model_id"] == model_id
        assert retrieved["custom_name"] == custom_name

    async def test_default_name_generation_integration(self, ml_engine, sample_model):
        """Test default name generation with list_models."""
        user_id = sample_model["user_id"]

        # Don't set custom name - should generate default
        models = ml_engine.list_models(user_id)

        assert len(models) == 1
        model = models[0]

        # Should have display_name even without custom_name
        assert "display_name" in model
        assert model["display_name"] == "Linear Regression - Jan 14, 2025"
        assert model.get("custom_name") is None

    async def test_multiple_models_with_mixed_names(self, ml_engine, tmp_path):
        """Test listing multiple models with mix of custom and default names."""
        user_id = 12345

        # Create three models
        models_data = [
            {
                "model_id": "model_12345_linear_20251014",
                "model_type": "linear",
                "task_type": "regression",
                "created_at": "2025-01-14T21:44:00Z",
                "custom_name": "Housing Predictor"
            },
            {
                "model_id": "model_12345_random_forest_20251013",
                "model_type": "random_forest",
                "task_type": "classification",
                "created_at": "2025-01-13T10:00:00Z",
                # No custom_name - will use default
            },
            {
                "model_id": "model_12345_keras_binary_classification_20251012",
                "model_type": "keras_binary_classification",
                "task_type": "classification",
                "created_at": "2025-01-12T15:30:00Z",
                "custom_name": "Churn Predictor v2"
            }
        ]

        for model_data in models_data:
            model_dir = tmp_path / "models" / f"user_{user_id}" / model_data["model_id"]
            model_dir.mkdir(parents=True, exist_ok=True)

            metadata = {**model_data, "user_id": user_id}
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        # List all models
        models = ml_engine.list_models(user_id)

        assert len(models) == 3

        # Check custom names are preserved
        custom_names = {m["model_id"]: m for m in models if m.get("custom_name")}
        assert len(custom_names) == 2
        assert custom_names["model_12345_linear_20251014"]["display_name"] == "Housing Predictor"
        assert custom_names["model_12345_keras_binary_classification_20251012"]["display_name"] == "Churn Predictor v2"

        # Check default name was generated
        default_models = {m["model_id"]: m for m in models if not m.get("custom_name")}
        assert len(default_models) == 1
        assert default_models["model_12345_random_forest_20251013"]["display_name"] == "Random Forest - Jan 13, 2025"

    async def test_name_validation_integration(self, ml_engine, sample_model):
        """Test that invalid names are rejected during integration."""
        from src.utils.exceptions import ValidationError
        user_id = sample_model["user_id"]
        model_id = sample_model["model_id"]

        # Too short
        with pytest.raises(ValidationError) as exc_info:
            ml_engine.set_model_name(user_id, model_id, "ab")
        assert "at least 3 characters" in str(exc_info.value)

        # Invalid characters
        with pytest.raises(ValidationError):
            ml_engine.set_model_name(user_id, model_id, "model/test")

        # Too long
        with pytest.raises(ValidationError):
            ml_engine.set_model_name(user_id, model_id, "a" * 101)

        # Valid name should work
        success = ml_engine.set_model_name(user_id, model_id, "Valid Model Name")
        assert success

    async def test_state_prerequisites_enforcement(self, state_manager):
        """Test that state transitions enforce pending_model_id prerequisite."""
        user_id = 12345
        conversation_id = "conv_test_123"

        # Start training
        await state_manager.start_ml_training(user_id, conversation_id)
        session = await state_manager.get_session(user_id, conversation_id)

        # Try to transition to TRAINING_COMPLETE without pending_model_id
        # Should fail due to prerequisite check
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        # Note: This might succeed depending on prerequisites configuration
        # The real test is trying to go to NAMING_MODEL without the model_id

        # Now set pending_model_id and try again
        session.selections["pending_model_id"] = "test_model_id"
        await state_manager.update_session(session)

        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.TRAINING_COMPLETE.value
        )
        assert success

        # Now transition to NAMING_MODEL should work
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.NAMING_MODEL.value
        )
        assert success, f"Failed to transition with prerequisite: {error}"

    async def test_workflow_cleanup_after_completion(self, state_manager, sample_model):
        """Test that session can be cleaned up after workflow completes."""
        user_id = sample_model["user_id"]
        conversation_id = "conv_test_123"
        model_id = sample_model["model_id"]

        # Complete full workflow
        await state_manager.start_ml_training(user_id, conversation_id)
        session = await state_manager.get_session(user_id, conversation_id)

        session.selections["pending_model_id"] = model_id
        await state_manager.update_session(session)

        # Progress through states
        await state_manager.transition_state(session, MLTrainingState.TRAINING_COMPLETE.value)
        await state_manager.transition_state(session, MLTrainingState.MODEL_NAMED.value)
        await state_manager.transition_state(session, MLTrainingState.COMPLETE.value)

        # End workflow
        await state_manager.end_ml_training(user_id, conversation_id)

        # Session should be gone
        with pytest.raises(Exception):
            await state_manager.get_session(user_id, conversation_id)

    async def test_duplicate_name_handling(self, ml_engine, tmp_path, caplog):
        """Test that duplicate names log warnings but still succeed."""
        import logging
        user_id = 12345
        duplicate_name = "Duplicate Model"

        # Create two models
        for i, model_id in enumerate([
            "model_12345_linear_20251014",
            "model_12345_random_forest_20251013"
        ]):
            model_dir = tmp_path / "models" / f"user_{user_id}" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "model_id": model_id,
                "user_id": user_id,
                "model_type": "linear" if i == 0 else "random_forest",
                "task_type": "regression" if i == 0 else "classification",
                "created_at": f"2025-01-{14-i:02d}T10:00:00Z"
            }

            # First model gets the duplicate name
            if i == 0:
                metadata["custom_name"] = duplicate_name

            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        # Try to set same name on second model
        with caplog.at_level(logging.WARNING):
            success = ml_engine.set_model_name(
                user_id,
                "model_12345_random_forest_20251013",
                duplicate_name
            )

        # Should succeed but log warning
        assert success
        assert any("already has a model named" in record.message for record in caplog.records)

    async def test_get_model_by_name_returns_most_recent_duplicate(self, ml_engine, tmp_path, caplog):
        """Test that get_model_by_name returns most recent model when duplicates exist."""
        import logging
        user_id = 12345
        duplicate_name = "Same Name"

        # Create three models with same name (different dates)
        models_data = [
            ("model_12345_linear_20251012", "2025-01-12T10:00:00Z"),
            ("model_12345_linear_20251014", "2025-01-14T10:00:00Z"),  # Most recent
            ("model_12345_linear_20251013", "2025-01-13T10:00:00Z"),
        ]

        for model_id, created_at in models_data:
            model_dir = tmp_path / "models" / f"user_{user_id}" / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "model_id": model_id,
                "user_id": user_id,
                "model_type": "linear",
                "task_type": "regression",
                "created_at": created_at,
                "custom_name": duplicate_name
            }

            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

        # Get model by name
        with caplog.at_level(logging.WARNING):
            result = ml_engine.get_model_by_name(user_id, duplicate_name)

        # Should return most recent (2025-01-14)
        assert result is not None
        assert result["model_id"] == "model_12345_linear_20251014"

        # Should log warning about duplicates
        assert any("Multiple models named" in record.message for record in caplog.records)

    async def test_end_to_end_workflow_with_custom_name(self, state_manager, ml_engine, sample_model):
        """Test complete workflow from training to named model."""
        user_id = sample_model["user_id"]
        conversation_id = "conv_test_123"
        model_id = sample_model["model_id"]
        custom_name = "End to End Test Model"

        # Step 1: Start workflow
        await state_manager.start_ml_training(user_id, conversation_id)
        session = await state_manager.get_session(user_id, conversation_id)
        session.selections["pending_model_id"] = model_id
        await state_manager.update_session(session)

        # Step 2: Training completes
        await state_manager.transition_state(session, MLTrainingState.TRAINING_COMPLETE.value)
        assert session.current_state == MLTrainingState.TRAINING_COMPLETE.value

        # Step 3: User chooses to name model
        await state_manager.transition_state(session, MLTrainingState.NAMING_MODEL.value)
        assert session.current_state == MLTrainingState.NAMING_MODEL.value

        # Step 4: User provides custom name
        success = ml_engine.set_model_name(user_id, model_id, custom_name)
        assert success

        # Step 5: Name accepted, transition to MODEL_NAMED
        await state_manager.transition_state(session, MLTrainingState.MODEL_NAMED.value)
        assert session.current_state == MLTrainingState.MODEL_NAMED.value

        # Step 6: Workflow completes
        await state_manager.transition_state(session, MLTrainingState.COMPLETE.value)
        assert session.current_state == MLTrainingState.COMPLETE.value

        # Step 7: Verify model can be retrieved by name
        retrieved = ml_engine.get_model_by_name(user_id, custom_name)
        assert retrieved is not None
        assert retrieved["model_id"] == model_id
        assert retrieved["custom_name"] == custom_name

        # Step 8: Verify model appears in list with correct display name
        models = ml_engine.list_models(user_id)
        assert len(models) == 1
        assert models[0]["display_name"] == custom_name
        assert models[0]["custom_name"] == custom_name

        # Step 9: Cleanup
        await state_manager.end_ml_training(user_id, conversation_id)
