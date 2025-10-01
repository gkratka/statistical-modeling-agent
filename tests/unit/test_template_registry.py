"""
Test suite for template registry system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


def test_template_registry_creation():
    """Test TemplateRegistry creation."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = TemplateRegistry(Path(tmpdir))
        assert registry is not None
        assert registry.template_dir == Path(tmpdir)


def test_template_registry_invalid_directory():
    """Test TemplateRegistry with invalid directory."""
    from src.generators.template_registry import TemplateRegistry

    with pytest.raises(FileNotFoundError):
        TemplateRegistry(Path("/nonexistent/directory"))


def test_get_template_success():
    """Test successful template retrieval."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test template structure
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_file = stats_dir / "descriptive.j2"
        template_content = """
import json
import pandas as pd

def main():
    data = json.loads(sys.stdin.read())
    result = {"mean": {{ columns | length }}}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        template_file.write_text(template_content)

        registry = TemplateRegistry(Path(tmpdir))
        template = registry.get_template("stats", "descriptive")

        assert template is not None
        rendered = template.render(columns=["age", "income"])
        assert "2" in rendered  # length of columns


def test_get_template_not_found():
    """Test template not found error."""
    from src.generators.template_registry import TemplateRegistry
    from src.utils.exceptions import ScriptGenerationError

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = TemplateRegistry(Path(tmpdir))

        with pytest.raises(ScriptGenerationError, match="Template not found"):
            registry.get_template("nonexistent", "template")


def test_template_caching():
    """Test template caching functionality."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test template
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_file = stats_dir / "test.j2"
        template_file.write_text("Test template content")

        registry = TemplateRegistry(Path(tmpdir))

        # First call should cache the template
        template1 = registry.get_template("stats", "test")

        # Second call should return cached template
        template2 = registry.get_template("stats", "test")

        # Should be the same object (cached)
        assert template1 is template2


def test_list_available_templates():
    """Test listing available templates."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test templates
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        ml_dir = Path(tmpdir) / "ml"
        ml_dir.mkdir()

        # Create template files
        (stats_dir / "descriptive.j2").write_text("content")
        (stats_dir / "correlation.j2").write_text("content")
        (ml_dir / "classifier.j2").write_text("content")

        registry = TemplateRegistry(Path(tmpdir))
        templates = registry.list_available_templates()

        assert "stats" in templates
        assert "ml" in templates
        assert "descriptive" in templates["stats"]
        assert "correlation" in templates["stats"]
        assert "classifier" in templates["ml"]


def test_template_validation():
    """Test template validation on load."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create invalid template
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_file = stats_dir / "invalid.j2"
        template_file.write_text("{% invalid jinja syntax %}")

        registry = TemplateRegistry(Path(tmpdir))

        # Should handle invalid template gracefully
        with pytest.raises(Exception):  # Jinja2 will raise TemplateSyntaxError
            registry.get_template("stats", "invalid")


def test_template_registry_reload():
    """Test template registry reload functionality."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_file = stats_dir / "test.j2"
        template_file.write_text("Original content")

        registry = TemplateRegistry(Path(tmpdir))

        # Load template
        template1 = registry.get_template("stats", "test")
        original_rendered = template1.render()

        # Modify template file
        template_file.write_text("Modified content")

        # Reload registry
        registry.reload()

        # Get template again (should be reloaded)
        template2 = registry.get_template("stats", "test")
        modified_rendered = template2.render()

        assert "Original" in original_rendered
        assert "Modified" in modified_rendered


def test_template_metadata():
    """Test template metadata extraction."""
    from src.generators.template_registry import TemplateRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_content = """
{#
Template: Descriptive Statistics
Description: Calculate basic descriptive statistics
Author: System
Version: 1.0
Required_params: columns, statistics
#}
import json
import pandas as pd
"""

        template_file = stats_dir / "descriptive.j2"
        template_file.write_text(template_content)

        registry = TemplateRegistry(Path(tmpdir))
        metadata = registry.get_template_metadata("stats", "descriptive")

        assert metadata["template"] == "Descriptive Statistics"
        assert metadata["description"] == "Calculate basic descriptive statistics"
        assert metadata["version"] == "1.0"