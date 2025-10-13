"""
Unit tests for TemplateManager.

Tests all CRUD operations for ML training templates including:
- Save template (create and update)
- Load template
- List templates
- Delete template
- Rename template
- Name validation
- Template existence checks
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from src.core.training_template import TemplateConfig, TrainingTemplate
from src.core.template_manager import TemplateManager


@pytest.fixture
def temp_templates_dir():
    """Create temporary templates directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def template_config(temp_templates_dir):
    """Create TemplateConfig for testing."""
    return TemplateConfig(
        enabled=True,
        templates_dir=temp_templates_dir,
        max_templates_per_user=5,
        allowed_name_pattern=r"^[a-zA-Z0-9_]{1,32}$",
        name_max_length=32
    )


@pytest.fixture
def template_manager(template_config):
    """Create TemplateManager instance for testing."""
    return TemplateManager(template_config)


@pytest.fixture
def sample_template_config() -> Dict[str, Any]:
    """Create sample template configuration."""
    return {
        "file_path": "/path/to/data/housing.csv",
        "target_column": "price",
        "feature_columns": ["sqft", "bedrooms", "bathrooms"],
        "model_category": "regression",
        "model_type": "random_forest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }


class TestTemplateManagerInit:
    """Test TemplateManager initialization."""

    def test_initialization(self, template_manager, temp_templates_dir):
        """Test manager initializes correctly."""
        assert template_manager.templates_dir == Path(temp_templates_dir)
        assert template_manager.max_templates_per_user == 5
        assert template_manager.templates_dir.exists()

    def test_creates_templates_directory(self, template_config):
        """Test templates directory is created if it doesn't exist."""
        manager = TemplateManager(template_config)
        assert manager.templates_dir.exists()


class TestSaveTemplate:
    """Test template save operations."""

    def test_save_template_success(self, template_manager, sample_template_config):
        """Test successful template save."""
        user_id = 12345
        template_name = "housing_rf"

        success, message = template_manager.save_template(user_id, template_name, sample_template_config)

        assert success is True
        assert "saved successfully" in message.lower()
        assert template_manager.template_exists(user_id, template_name)

    def test_save_template_creates_user_directory(self, template_manager, sample_template_config):
        """Test user directory is created on first save."""
        user_id = 12345
        user_dir = template_manager.templates_dir / f"user_{user_id}"

        assert not user_dir.exists()

        template_manager.save_template(user_id, "test_template", sample_template_config)

        assert user_dir.exists()

    def test_save_template_invalid_name(self, template_manager, sample_template_config):
        """Test template save with invalid name."""
        user_id = 12345
        invalid_names = [
            "",  # Empty
            "test template",  # Space
            "test-template",  # Hyphen
            "test.template",  # Dot
            "a" * 33,  # Too long
            "123 test"  # Space
        ]

        for invalid_name in invalid_names:
            success, message = template_manager.save_template(user_id, invalid_name, sample_template_config)
            assert success is False
            assert len(message) > 0

    def test_save_template_missing_required_field(self, template_manager):
        """Test template save with missing required fields."""
        user_id = 12345
        incomplete_config = {
            "file_path": "/path/to/data.csv",
            # Missing target_column
            "feature_columns": ["col1", "col2"]
        }

        success, message = template_manager.save_template(user_id, "test", incomplete_config)

        assert success is False
        assert "missing" in message.lower() or "required" in message.lower()

    def test_save_template_exceeds_max_count(self, template_manager, sample_template_config):
        """Test template save when max count exceeded."""
        user_id = 12345
        max_templates = template_manager.max_templates_per_user

        # Create max templates
        for i in range(max_templates):
            template_manager.save_template(user_id, f"template_{i}", sample_template_config)

        # Try to create one more
        success, message = template_manager.save_template(user_id, "excess_template", sample_template_config)

        assert success is False
        assert "maximum" in message.lower() or "exceeded" in message.lower()

    def test_update_existing_template(self, template_manager, sample_template_config):
        """Test updating existing template preserves created_at."""
        user_id = 12345
        template_name = "test_template"

        # Create initial template
        template_manager.save_template(user_id, template_name, sample_template_config)
        original = template_manager.load_template(user_id, template_name)

        # Update template with different config
        updated_config = sample_template_config.copy()
        updated_config["hyperparameters"] = {"n_estimators": 200}

        success, message = template_manager.save_template(user_id, template_name, updated_config)

        assert success is True
        assert "updated" in message.lower()

        updated = template_manager.load_template(user_id, template_name)
        assert updated.hyperparameters["n_estimators"] == 200


class TestLoadTemplate:
    """Test template load operations."""

    def test_load_template_success(self, template_manager, sample_template_config):
        """Test successful template load."""
        user_id = 12345
        template_name = "test_template"

        template_manager.save_template(user_id, template_name, sample_template_config)
        loaded = template_manager.load_template(user_id, template_name)

        assert loaded is not None
        assert loaded.template_name == template_name
        assert loaded.user_id == user_id
        assert loaded.file_path == sample_template_config["file_path"]
        assert loaded.target_column == sample_template_config["target_column"]
        assert loaded.feature_columns == sample_template_config["feature_columns"]

    def test_load_template_not_found(self, template_manager):
        """Test loading non-existent template."""
        user_id = 12345
        loaded = template_manager.load_template(user_id, "nonexistent")

        assert loaded is None

    def test_load_template_different_user(self, template_manager, sample_template_config):
        """Test loading template from different user."""
        user_id_1 = 12345
        user_id_2 = 67890
        template_name = "test_template"

        template_manager.save_template(user_id_1, template_name, sample_template_config)
        loaded = template_manager.load_template(user_id_2, template_name)

        assert loaded is None  # Different user should not see it

    def test_load_template_corrupted_json(self, template_manager, sample_template_config):
        """Test loading template with corrupted JSON."""
        user_id = 12345
        template_name = "corrupted"

        # Save valid template first
        template_manager.save_template(user_id, template_name, sample_template_config)

        # Corrupt the JSON file
        user_dir = template_manager._get_user_directory(user_id)
        template_file = user_dir / f"{template_name}.json"
        with open(template_file, 'w') as f:
            f.write("{ invalid json }")

        loaded = template_manager.load_template(user_id, template_name)
        assert loaded is None


class TestListTemplates:
    """Test template listing operations."""

    def test_list_templates_empty(self, template_manager):
        """Test listing templates with no templates."""
        user_id = 12345
        templates = template_manager.list_templates(user_id)

        assert templates == []

    def test_list_templates_multiple(self, template_manager, sample_template_config):
        """Test listing multiple templates."""
        user_id = 12345
        template_names = ["template_1", "template_2", "template_3"]

        for name in template_names:
            template_manager.save_template(user_id, name, sample_template_config)

        templates = template_manager.list_templates(user_id)

        assert len(templates) == 3
        loaded_names = [t.template_name for t in templates]
        assert set(loaded_names) == set(template_names)

    def test_list_templates_sorted_by_last_used(self, template_manager, sample_template_config):
        """Test templates are sorted by last_used."""
        user_id = 12345

        # Create templates with different last_used timestamps
        template_manager.save_template(user_id, "old", sample_template_config)
        template_manager.save_template(user_id, "new", sample_template_config)

        # Update last_used for "old" to be more recent
        old_template = template_manager.load_template(user_id, "old")
        old_config = sample_template_config.copy()
        old_config["last_used"] = "2025-10-12T10:00:00Z"
        template_manager.save_template(user_id, "old", old_config)

        new_config = sample_template_config.copy()
        new_config["last_used"] = "2025-10-11T10:00:00Z"
        template_manager.save_template(user_id, "new", new_config)

        templates = template_manager.list_templates(user_id)

        # Most recently used should be first
        assert templates[0].template_name == "old"
        assert templates[1].template_name == "new"

    def test_list_templates_user_isolation(self, template_manager, sample_template_config):
        """Test templates are isolated per user."""
        user_1 = 12345
        user_2 = 67890

        template_manager.save_template(user_1, "user1_template", sample_template_config)
        template_manager.save_template(user_2, "user2_template", sample_template_config)

        templates_1 = template_manager.list_templates(user_1)
        templates_2 = template_manager.list_templates(user_2)

        assert len(templates_1) == 1
        assert len(templates_2) == 1
        assert templates_1[0].template_name == "user1_template"
        assert templates_2[0].template_name == "user2_template"


class TestDeleteTemplate:
    """Test template deletion operations."""

    def test_delete_template_success(self, template_manager, sample_template_config):
        """Test successful template deletion."""
        user_id = 12345
        template_name = "test_template"

        template_manager.save_template(user_id, template_name, sample_template_config)
        assert template_manager.template_exists(user_id, template_name)

        success = template_manager.delete_template(user_id, template_name)

        assert success is True
        assert not template_manager.template_exists(user_id, template_name)

    def test_delete_template_not_found(self, template_manager):
        """Test deleting non-existent template."""
        user_id = 12345
        success = template_manager.delete_template(user_id, "nonexistent")

        assert success is False

    def test_delete_template_different_user(self, template_manager, sample_template_config):
        """Test deleting template from different user."""
        user_1 = 12345
        user_2 = 67890
        template_name = "test_template"

        template_manager.save_template(user_1, template_name, sample_template_config)

        success = template_manager.delete_template(user_2, template_name)

        assert success is False
        assert template_manager.template_exists(user_1, template_name)


class TestRenameTemplate:
    """Test template rename operations."""

    def test_rename_template_success(self, template_manager, sample_template_config):
        """Test successful template rename."""
        user_id = 12345
        old_name = "old_name"
        new_name = "new_name"

        template_manager.save_template(user_id, old_name, sample_template_config)

        success, message = template_manager.rename_template(user_id, old_name, new_name)

        assert success is True
        assert not template_manager.template_exists(user_id, old_name)
        assert template_manager.template_exists(user_id, new_name)

        # Verify data is preserved
        renamed = template_manager.load_template(user_id, new_name)
        assert renamed.file_path == sample_template_config["file_path"]

    def test_rename_template_invalid_new_name(self, template_manager, sample_template_config):
        """Test rename with invalid new name."""
        user_id = 12345
        old_name = "valid_name"

        template_manager.save_template(user_id, old_name, sample_template_config)

        success, message = template_manager.rename_template(user_id, old_name, "invalid name")

        assert success is False
        assert template_manager.template_exists(user_id, old_name)

    def test_rename_template_not_found(self, template_manager):
        """Test renaming non-existent template."""
        user_id = 12345

        success, message = template_manager.rename_template(user_id, "nonexistent", "new_name")

        assert success is False
        assert "not found" in message.lower()

    def test_rename_template_name_exists(self, template_manager, sample_template_config):
        """Test rename to existing template name."""
        user_id = 12345

        template_manager.save_template(user_id, "template1", sample_template_config)
        template_manager.save_template(user_id, "template2", sample_template_config)

        success, message = template_manager.rename_template(user_id, "template1", "template2")

        assert success is False
        assert "already exists" in message.lower()


class TestValidateTemplateName:
    """Test template name validation."""

    def test_validate_name_valid(self, template_manager):
        """Test validation with valid names."""
        valid_names = [
            "test",
            "test_template",
            "template123",
            "TEST_TEMPLATE",
            "a",
            "a" * 32
        ]

        for name in valid_names:
            is_valid, message = template_manager.validate_template_name(name)
            assert is_valid is True, f"Failed for: {name}"
            assert message == ""

    def test_validate_name_invalid(self, template_manager):
        """Test validation with invalid names."""
        invalid_names = [
            ("", "empty"),
            (" test", "whitespace"),
            ("test ", "whitespace"),
            ("test template", "space"),
            ("test-template", "hyphen"),
            ("test.template", "dot"),
            ("a" * 33, "too long"),
            ("CON", "reserved"),
            ("NUL", "reserved")
        ]

        for name, reason in invalid_names:
            is_valid, message = template_manager.validate_template_name(name)
            assert is_valid is False, f"Should fail for {reason}: {name}"
            assert len(message) > 0

    def test_validate_name_reserved_names(self, template_manager):
        """Test reserved name validation."""
        reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]

        for name in reserved:
            is_valid, message = template_manager.validate_template_name(name)
            assert is_valid is False
            assert "reserved" in message.lower()


class TestTemplateExists:
    """Test template existence checks."""

    def test_template_exists_true(self, template_manager, sample_template_config):
        """Test template exists check returns True."""
        user_id = 12345
        template_name = "test_template"

        template_manager.save_template(user_id, template_name, sample_template_config)

        assert template_manager.template_exists(user_id, template_name) is True

    def test_template_exists_false(self, template_manager):
        """Test template exists check returns False."""
        user_id = 12345

        assert template_manager.template_exists(user_id, "nonexistent") is False

    def test_template_exists_different_user(self, template_manager, sample_template_config):
        """Test template exists is user-specific."""
        user_1 = 12345
        user_2 = 67890
        template_name = "test_template"

        template_manager.save_template(user_1, template_name, sample_template_config)

        assert template_manager.template_exists(user_1, template_name) is True
        assert template_manager.template_exists(user_2, template_name) is False


class TestGetTemplateCount:
    """Test template count operations."""

    def test_get_template_count_zero(self, template_manager):
        """Test count with no templates."""
        user_id = 12345
        count = template_manager.get_template_count(user_id)

        assert count == 0

    def test_get_template_count_multiple(self, template_manager, sample_template_config):
        """Test count with multiple templates."""
        user_id = 12345

        for i in range(3):
            template_manager.save_template(user_id, f"template_{i}", sample_template_config)

        count = template_manager.get_template_count(user_id)

        assert count == 3
