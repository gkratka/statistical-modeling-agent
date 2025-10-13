"""Integration tests for ML training templates workflow end-to-end."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.core.state_manager import StateManager, MLTrainingState, WorkflowType
from src.core.template_manager import TemplateManager
from src.core.training_template import TemplateConfig
from src.processors.data_loader import DataLoader
from src.utils.path_validator import PathValidator
from src.bot.ml_handlers.template_handlers import TemplateHandlers


@pytest.fixture
def test_csv_path(tmp_path):
    """Create a test CSV file."""
    csv_file = tmp_path / "housing.csv"
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
def templates_dir(tmp_path):
    """Create temporary templates directory."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return str(templates_dir)


@pytest.fixture
def state_manager():
    """Create state manager instance."""
    return StateManager()


@pytest.fixture
def template_manager(templates_dir):
    """Create template manager instance."""
    config = TemplateConfig(
        enabled=True,
        templates_dir=templates_dir,
        max_templates_per_user=50,
        allowed_name_pattern=r"^[a-zA-Z0-9_]{1,32}$",
        name_max_length=32
    )
    return TemplateManager(config)


@pytest.fixture
def data_loader(tmp_path):
    """Create data loader with test configuration."""
    loader = DataLoader()
    loader.local_enabled = True
    loader.allowed_directories = [str(tmp_path)]
    return loader


@pytest.fixture
def path_validator(tmp_path):
    """Create path validator."""
    return PathValidator(
        allowed_directories=[str(tmp_path)],
        max_size_mb=1000,
        allowed_extensions=['.csv', '.xlsx', '.parquet']
    )


@pytest.fixture
def template_handlers(state_manager, template_manager, data_loader, path_validator):
    """Create template handlers instance."""
    return TemplateHandlers(
        state_manager=state_manager,
        template_manager=template_manager,
        data_loader=data_loader,
        path_validator=path_validator
    )


@pytest.mark.asyncio
class TestTemplateSaveWorkflow:
    """Test template save workflow end-to-end."""

    async def test_save_template_full_workflow(
        self,
        state_manager,
        template_handlers,
        test_csv_path
    ):
        """
        Test full template save workflow:
        1. User completes ML training configuration
        2. User clicks "Save as Template"
        3. User enters template name
        4. Template is saved with all configuration
        5. User can choose to continue training or finish
        """
        # Setup - user has completed configuration
        user_id = 12345
        conversation_id = "test_conv"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.workflow_type = WorkflowType.ML_TRAINING
        session.file_path = test_csv_path
        session.selections = {
            'target_column': 'price',
            'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
            'model_category': 'regression',
            'model_type': 'random_forest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10
            }
        }

        # Mock update object for template_save_request
        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        # Step 1: User clicks "Save as Template"
        session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS.value
        await state_manager.update_session(session)

        await template_handlers.handle_template_save_request(update_mock, context_mock)

        # Verify state transition
        session = await state_manager.get_session(user_id, conversation_id)
        assert session.current_state == MLTrainingState.SAVING_TEMPLATE.value

        # Step 2: User enters template name
        update_text_mock = AsyncMock()
        update_text_mock.message.text = "housing_rf_model"
        update_text_mock.message.chat_id = conversation_id
        update_text_mock.message.reply_text = AsyncMock()
        update_text_mock.effective_user.id = user_id

        await template_handlers.handle_template_name_input(update_text_mock, context_mock)

        # Step 3: Verify template was saved
        templates = template_handlers.template_manager.list_templates(user_id)
        assert len(templates) == 1
        assert templates[0].template_name == "housing_rf_model"
        assert templates[0].file_path == test_csv_path
        assert templates[0].target_column == "price"
        assert len(templates[0].feature_columns) == 3
        assert templates[0].model_type == "random_forest"

    async def test_save_template_invalid_name(
        self,
        state_manager,
        template_handlers
    ):
        """Test template save with invalid name."""
        user_id = 12345
        conversation_id = "test_conv"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.current_state = MLTrainingState.SAVING_TEMPLATE.value
        session.file_path = "/path/to/data.csv"
        session.selections = {
            'target_column': 'price',
            'feature_columns': ['sqft'],
            'model_type': 'linear'
        }

        update_mock = AsyncMock()
        update_mock.message.text = "invalid template name!"  # Has space and !
        update_mock.message.chat_id = conversation_id
        update_mock.message.reply_text = AsyncMock()
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_template_name_input(update_mock, context_mock)

        # Verify error message was shown
        update_mock.message.reply_text.assert_called()
        call_args = update_mock.message.reply_text.call_args[0][0]
        assert "Invalid template name" in call_args

        # Verify template was not saved
        templates = template_handlers.template_manager.list_templates(user_id)
        assert len(templates) == 0

    async def test_save_template_duplicate_name(
        self,
        state_manager,
        template_handlers,
        test_csv_path
    ):
        """Test saving template with duplicate name."""
        user_id = 12345
        conversation_id = "test_conv"

        # Save first template
        template_handlers.template_manager.save_template(
            user_id=user_id,
            template_name="existing_template",
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft'],
                'model_type': 'linear'
            }
        )

        # Try to save another template with same name
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.current_state = MLTrainingState.SAVING_TEMPLATE.value
        session.file_path = test_csv_path
        session.selections = {
            'target_column': 'price',
            'feature_columns': ['sqft', 'bedrooms'],
            'model_type': 'ridge'
        }

        update_mock = AsyncMock()
        update_mock.message.text = "existing_template"
        update_mock.message.chat_id = conversation_id
        update_mock.message.reply_text = AsyncMock()
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_template_name_input(update_mock, context_mock)

        # Verify error message about duplicate
        update_mock.message.reply_text.assert_called()
        call_args = update_mock.message.reply_text.call_args[0][0]
        assert "already exists" in call_args


@pytest.mark.asyncio
class TestTemplateLoadWorkflow:
    """Test template load workflow end-to-end."""

    async def test_load_template_full_workflow(
        self,
        state_manager,
        template_handlers,
        test_csv_path
    ):
        """
        Test full template load workflow:
        1. User selects "Use Template" data source
        2. Template list is displayed
        3. User selects a template
        4. Template configuration is shown
        5. User chooses to load data now
        6. Training proceeds with template configuration
        """
        user_id = 12345
        conversation_id = "test_conv"

        # Setup - create a saved template
        template_handlers.template_manager.save_template(
            user_id=user_id,
            template_name="saved_rf_model",
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms', 'bathrooms'],
                'model_category': 'regression',
                'model_type': 'random_forest',
                'hyperparameters': {'n_estimators': 100}
            }
        )

        # Step 1: User chooses "Use Template"
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_template_source_selection(update_mock, context_mock)

        # Verify state transition and template list shown
        session = await state_manager.get_session(user_id, conversation_id)
        assert session.current_state == MLTrainingState.LOADING_TEMPLATE.value

        # Step 2: User selects template
        update_mock.callback_query.data = "load_template:saved_rf_model"
        await template_handlers.handle_template_selection(update_mock, context_mock)

        # Verify session populated with template data
        session = await state_manager.get_session(user_id, conversation_id)
        assert session.file_path == test_csv_path
        assert session.selections['target_column'] == 'price'
        assert len(session.selections['feature_columns']) == 3
        assert session.selections['model_type'] == 'random_forest'
        assert session.current_state == MLTrainingState.CONFIRMING_TEMPLATE.value

    async def test_load_template_no_templates_available(
        self,
        state_manager,
        template_handlers
    ):
        """Test loading template when user has no templates."""
        user_id = 99999  # User with no templates
        conversation_id = "test_conv"

        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.current_state = MLTrainingState.CHOOSING_DATA_SOURCE.value

        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_template_source_selection(update_mock, context_mock)

        # Verify "no templates" message shown
        update_mock.callback_query.edit_message_text.assert_called()
        call_args = update_mock.callback_query.edit_message_text.call_args[0][0]
        assert "No templates found" in call_args

    async def test_load_template_with_deferred_loading(
        self,
        state_manager,
        template_handlers,
        test_csv_path
    ):
        """Test template load with deferred data loading."""
        user_id = 12345
        conversation_id = "test_conv"

        # Create template
        template_handlers.template_manager.save_template(
            user_id=user_id,
            template_name="deferred_template",
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms'],
                'model_type': 'linear'
            }
        )

        # Load template
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.current_state = MLTrainingState.LOADING_TEMPLATE.value

        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.callback_query.data = "load_template:deferred_template"
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_template_selection(update_mock, context_mock)

        # User chooses "Defer Loading"
        session = await state_manager.get_session(user_id, conversation_id)
        update_mock.callback_query.data = "template_defer"
        await template_handlers.handle_template_load_option(update_mock, context_mock)

        # Verify session has deferred flag set
        session = await state_manager.get_session(user_id, conversation_id)
        assert session.load_deferred is True
        assert session.current_state == MLTrainingState.COMPLETE.value

    async def test_load_template_with_invalid_path(
        self,
        state_manager,
        template_handlers
    ):
        """Test template load when file path is invalid."""
        user_id = 12345
        conversation_id = "test_conv"

        # Create template with invalid path
        template_handlers.template_manager.save_template(
            user_id=user_id,
            template_name="invalid_path_template",
            config={
                'file_path': "/nonexistent/path/data.csv",
                'target_column': 'price',
                'feature_columns': ['sqft'],
                'model_type': 'linear'
            }
        )

        # Load template and try immediate loading
        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.current_state = MLTrainingState.LOADING_TEMPLATE.value

        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.callback_query.data = "load_template:invalid_path_template"
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()
        context_mock.bot_data = {}

        await template_handlers.handle_template_selection(update_mock, context_mock)

        # Try to load data immediately (should fail)
        session = await state_manager.get_session(user_id, conversation_id)
        update_mock.callback_query.data = "template_load_now"

        await template_handlers.handle_template_load_option(update_mock, context_mock)

        # Verify error message shown
        update_mock.callback_query.edit_message_text.assert_called()
        call_args = update_mock.callback_query.edit_message_text.call_args[0][0]
        assert "Invalid file path" in call_args or "Invalid Request" in call_args


@pytest.mark.asyncio
class TestTemplateCancellation:
    """Test template workflow cancellation."""

    async def test_cancel_template_save(
        self,
        state_manager,
        template_handlers
    ):
        """Test cancelling template save operation."""
        user_id = 12345
        conversation_id = "test_conv"

        session = await state_manager.get_or_create_session(user_id, conversation_id)
        session.workflow_type = WorkflowType.ML_TRAINING
        session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS.value

        # Save state snapshot
        session.save_state_snapshot()

        # Transition to saving template
        await state_manager.transition_state(session, MLTrainingState.SAVING_TEMPLATE.value)
        assert session.current_state == MLTrainingState.SAVING_TEMPLATE.value

        # User cancels
        update_mock = AsyncMock()
        update_mock.callback_query = AsyncMock()
        update_mock.callback_query.answer = AsyncMock()
        update_mock.callback_query.edit_message_text = AsyncMock()
        update_mock.callback_query.message.chat_id = conversation_id
        update_mock.effective_user.id = user_id
        context_mock = AsyncMock()

        await template_handlers.handle_cancel_template(update_mock, context_mock)

        # Verify state restored to previous
        session = await state_manager.get_session(user_id, conversation_id)
        assert session.current_state == MLTrainingState.COLLECTING_HYPERPARAMETERS.value


@pytest.mark.asyncio
class TestTemplateListOperations:
    """Test template listing and management."""

    async def test_list_multiple_templates(
        self,
        template_manager,
        test_csv_path
    ):
        """Test listing multiple templates sorted by last used."""
        user_id = 12345

        # Create multiple templates
        for i in range(3):
            template_manager.save_template(
                user_id=user_id,
                template_name=f"template_{i}",
                config={
                    'file_path': test_csv_path,
                    'target_column': 'price',
                    'feature_columns': ['sqft'],
                    'model_type': 'linear'
                }
            )

        templates = template_manager.list_templates(user_id)
        assert len(templates) == 3

    async def test_template_user_isolation(
        self,
        template_manager,
        test_csv_path
    ):
        """Test templates are isolated per user."""
        user_1 = 12345
        user_2 = 67890

        # User 1 creates template
        template_manager.save_template(
            user_id=user_1,
            template_name="user1_template",
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft'],
                'model_type': 'linear'
            }
        )

        # User 2 cannot see User 1's template
        templates_user_2 = template_manager.list_templates(user_2)
        assert len(templates_user_2) == 0

        # User 2 can create same template name
        template_manager.save_template(
            user_id=user_2,
            template_name="user1_template",  # Same name, different user
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['bathrooms'],
                'model_type': 'ridge'
            }
        )

        # Each user has their own template
        templates_user_1 = template_manager.list_templates(user_1)
        templates_user_2 = template_manager.list_templates(user_2)

        assert len(templates_user_1) == 1
        assert len(templates_user_2) == 1
        assert templates_user_1[0].feature_columns == ['sqft']
        assert templates_user_2[0].feature_columns == ['bathrooms']


@pytest.mark.asyncio
class TestTemplateUpdates:
    """Test template update operations."""

    async def test_update_template_preserves_created_at(
        self,
        template_manager,
        test_csv_path
    ):
        """Test updating template preserves created_at timestamp."""
        user_id = 12345
        template_name = "test_template"

        # Create template
        template_manager.save_template(
            user_id=user_id,
            template_name=template_name,
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft'],
                'model_type': 'linear'
            }
        )

        original = template_manager.load_template(user_id, template_name)
        original_created_at = original.created_at

        # Update template
        success, message = template_manager.save_template(
            user_id=user_id,
            template_name=template_name,
            config={
                'file_path': test_csv_path,
                'target_column': 'price',
                'feature_columns': ['sqft', 'bedrooms'],  # Changed
                'model_type': 'ridge'  # Changed
            }
        )

        assert success
        assert "updated" in message.lower()

        # Verify created_at preserved
        updated = template_manager.load_template(user_id, template_name)
        assert updated.created_at == original_created_at
        assert len(updated.feature_columns) == 2
        assert updated.model_type == "ridge"
