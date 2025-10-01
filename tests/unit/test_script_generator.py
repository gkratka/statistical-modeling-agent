"""
Test suite for script generator system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


def test_script_generator_creation():
    """Test ScriptGenerator creation."""
    from src.generators.script_generator import ScriptGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ScriptGenerator(Path(tmpdir))
        assert generator is not None


def test_script_generator_with_task_definition():
    """Test script generation with TaskDefinition."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test template
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_content = """
import json
import pandas as pd
import sys

def main():
    data = json.loads(sys.stdin.read())
    df = pd.DataFrame(data['dataframe'])

    result = {
        'operation': '{{ operation }}',
        'columns': {{ columns | python_list }},
        'row_count': len(df)
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        (stats_dir / "descriptive.j2").write_text(template_content)

        generator = ScriptGenerator(Path(tmpdir))

        task = TaskDefinition(
            task_type="stats",
            operation="descriptive_stats",
            parameters={"columns": ["age", "income"], "statistics": ["mean"]},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = generator.generate(task, {})

        assert "import json" in script
        assert "import pandas" in script
        assert "descriptive_stats" in script
        assert "['age', 'income']" in script


def test_script_generator_parameter_sanitization():
    """Test parameter sanitization in script generation."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_content = """
columns = {{ columns | python_list }}
"""
        (stats_dir / "test.j2").write_text(template_content)

        generator = ScriptGenerator(Path(tmpdir))
        generator.enable_test_mode()

        task = TaskDefinition(
            task_type="stats",
            operation="test",
            parameters={"columns": ["age with spaces", "income-dash", "123numeric"]},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = generator.generate(task, {})

        # Check that column names are sanitized
        assert "age_with_spaces" in script
        assert "income_dash" in script
        assert "_123numeric" in script  # Should prefix with underscore


def test_script_generator_validation():
    """Test script validation during generation."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        # Create template with dangerous content
        dangerous_template = """
import os
os.system('rm -rf /')
"""
        (stats_dir / "dangerous.j2").write_text(dangerous_template)

        generator = ScriptGenerator(Path(tmpdir))

        task = TaskDefinition(
            task_type="stats",
            operation="dangerous",
            parameters={},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        with pytest.raises(Exception):  # Should raise security violation
            generator.generate(task, {})


def test_script_generator_template_not_found():
    """Test handling of missing templates."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition
    from src.utils.exceptions import ScriptGenerationError

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = ScriptGenerator(Path(tmpdir))

        task = TaskDefinition(
            task_type="stats",
            operation="nonexistent",
            parameters={},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        with pytest.raises(ScriptGenerationError):
            generator.generate(task, {})


def test_script_generator_ml_template():
    """Test ML template generation."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ML template
        ml_dir = Path(tmpdir) / "ml"
        ml_dir.mkdir()

        template_content = """
import json
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    data = json.loads(sys.stdin.read())
    df = pd.DataFrame(data['dataframe'])

    # Training logic here
    model_type = '{{ model_type }}'
    target = '{{ target }}'
    features = {{ features | python_list }}

    result = {
        'model_type': model_type,
        'target': target,
        'features': features,
        'status': 'trained'
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        (ml_dir / "train_classifier.j2").write_text(template_content)

        generator = ScriptGenerator(Path(tmpdir))
        generator.enable_test_mode()

        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "model_type": "linear_regression",
                "target": "price",
                "features": ["size", "bedrooms"]
            },
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = generator.generate(task, {})

        assert "LinearRegression" in script
        assert "linear_regression" in script
        assert "price" in script
        assert "['size', 'bedrooms']" in script


def test_script_generator_context_data():
    """Test script generation with context data."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_content = """
# Context data available: {{ data_shape if data_shape else 'unknown' }}
import json

def main():
    result = {'context_used': True}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        (stats_dir / "context_test.j2").write_text(template_content)

        generator = ScriptGenerator(Path(tmpdir))
        generator.enable_test_mode()

        task = TaskDefinition(
            task_type="stats",
            operation="context_test",
            parameters={},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        context_data = {"data_shape": (100, 5)}
        script = generator.generate(task, context_data)

        assert "(100, 5)" in script


def test_script_generator_timestamp_injection():
    """Test timestamp injection in generated scripts."""
    from src.generators.script_generator import ScriptGenerator
    from src.core.parser import TaskDefinition

    with tempfile.TemporaryDirectory() as tmpdir:
        stats_dir = Path(tmpdir) / "stats"
        stats_dir.mkdir()

        template_content = """
# Generated at: {{ timestamp }}
import json

def main():
    result = {'generated_at': '{{ timestamp }}'}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""
        (stats_dir / "timestamp_test.j2").write_text(template_content)

        generator = ScriptGenerator(Path(tmpdir))
        generator.enable_test_mode()

        task = TaskDefinition(
            task_type="stats",
            operation="timestamp_test",
            parameters={},
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = generator.generate(task, {})

        # Should contain a timestamp
        assert "Generated at:" in script
        assert "generated_at" in script