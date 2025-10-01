"""
Comprehensive test suite for correlation command flow.

Tests the complete flow from parsing to execution to verify
correlation commands work correctly end-to-end.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, Any

from src.core.parser import RequestParser, TaskDefinition
from src.generators.script_generator import ScriptGenerator
from src.core.orchestrator import TaskOrchestrator
from src.utils.exceptions import ParseError, ScriptGenerationError
from pathlib import Path


class TestCorrelationParsing:
    """Test correlation command parsing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = RequestParser()

    def test_parse_correlation_basic(self):
        """Test basic correlation command parsing."""
        text = "/script correlation"
        task = self.parser.parse_request(
            text=text,
            user_id=123,
            conversation_id="test"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"

    def test_parse_correlation_with_columns(self):
        """Test correlation parsing with specific columns."""
        text = "/script correlation leads and sales"
        task = self.parser.parse_request(
            text=text,
            user_id=123,
            conversation_id="test"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"
        # Should extract columns from the text
        columns = task.parameters.get('columns', [])
        assert 'leads' in columns
        assert 'sales' in columns

    def test_parse_correlation_with_for_preposition(self):
        """Test correlation parsing with 'for' preposition."""
        text = "/script correlation for leads and sales"
        task = self.parser.parse_request(
            text=text,
            user_id=123,
            conversation_id="test"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"
        columns = task.parameters.get('columns', [])
        assert 'leads' in columns
        assert 'sales' in columns

    def test_parse_natural_language_correlation(self):
        """Test natural language correlation requests."""
        text = "calculate correlation between leads and sales"
        task = self.parser.parse_request(
            text=text,
            user_id=123,
            conversation_id="test"
        )

        assert task.task_type == "script"
        assert task.operation == "correlation"


class TestCorrelationTemplateMapping:
    """Test correlation template mapping in script generator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use actual template directory
        template_dir = Path(__file__).parent.parent / 'templates'
        self.generator = ScriptGenerator(template_dir)

    def test_operation_template_mapping(self):
        """Test that correlation maps to correct template."""
        category, template_name = self.generator._get_template_mapping("correlation")
        assert category == "stats"
        assert template_name == "correlation"

    def test_correlation_analysis_mapping(self):
        """Test correlation_analysis operation mapping."""
        category, template_name = self.generator._get_template_mapping("correlation_analysis")
        assert category == "stats"
        assert template_name == "correlation"

    def test_fallback_mapping_excludes_correlation(self):
        """Test that fallback logic doesn't incorrectly map correlation to descriptive."""
        # This test should FAIL initially, revealing the bug
        with patch.object(self.generator, 'operation_templates', {}):
            # With empty operation_templates, test fallback logic
            try:
                category, template_name = self.generator._get_template_mapping("correlation")
                # This should NOT map to descriptive template
                assert not (category == "stats" and template_name == "descriptive"), \
                    "BUG: Correlation incorrectly mapped to descriptive template"
            except ScriptGenerationError:
                # This is acceptable - should raise error, not map incorrectly
                pass


class TestCorrelationScriptGeneration:
    """Test correlation script generation."""

    def setup_method(self):
        """Set up test fixtures."""
        template_dir = Path(__file__).parent.parent / 'templates'
        self.generator = ScriptGenerator(template_dir)

    def test_generate_correlation_script(self):
        """Test correlation script generation."""
        task = TaskDefinition(
            task_type="script",
            operation="correlation",
            parameters={
                'columns': ['leads', 'sales'],
                'method': 'pearson'
            },
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = self.generator.generate(task)

        # Verify script contains correlation-specific elements
        assert 'correlation' in script.lower()
        assert 'pearson' in script.lower() or 'corr' in script.lower()
        assert 'leads' in script
        assert 'sales' in script

    def test_correlation_template_exists(self):
        """Test that correlation template file exists."""
        template_path = Path(__file__).parent.parent / 'templates' / 'stats' / 'correlation.j2'
        assert template_path.exists(), "Correlation template file not found"

    def test_correlation_script_parameters(self):
        """Test correlation script with various parameters."""
        task = TaskDefinition(
            task_type="script",
            operation="correlation",
            parameters={
                'columns': ['column1', 'column2', 'column3'],
                'method': 'spearman'
            },
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        script = self.generator.generate(task)

        # Should handle multiple columns
        assert 'column1' in script
        assert 'column2' in script
        assert 'column3' in script
        assert 'spearman' in script.lower()


class TestCorrelationEndToEnd:
    """Test end-to-end correlation execution."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'leads': [100, 120, 130, 110, 140],
            'sales': [50, 60, 65, 55, 70],
            'score': [75, 80, 85, 77, 88]
        })

    @pytest.mark.asyncio
    async def test_correlation_execution_flow(self):
        """Test complete correlation execution flow."""
        # Parse correlation request
        parser = RequestParser()
        task = parser.parse_request(
            text="/script correlation leads and sales",
            user_id=123,
            conversation_id="test"
        )

        # Generate script
        template_dir = Path(__file__).parent.parent / 'templates'
        generator = ScriptGenerator(template_dir)
        script = generator.generate(task)

        # Verify script was generated (not empty)
        assert len(script) > 100
        assert 'correlation' in script.lower()

    def test_correlation_parameter_validation(self):
        """Test correlation parameter validation."""
        parser = RequestParser()

        # Test single column (should fail or handle gracefully)
        task = parser.parse_request(
            text="/script correlation leads",
            user_id=123,
            conversation_id="test"
        )

        # Should have extracted at least one column
        columns = task.parameters.get('columns', [])
        assert len(columns) >= 1


class TestCorrelationErrorHandling:
    """Test correlation error handling scenarios."""

    def test_invalid_correlation_operation(self):
        """Test handling of invalid correlation operations."""
        template_dir = Path(__file__).parent.parent / 'templates'
        generator = ScriptGenerator(template_dir)

        # Test with invalid operation that contains 'correlation'
        with pytest.raises(ScriptGenerationError):
            generator._get_template_mapping("invalid_correlation_operation")

    def test_missing_correlation_parameters(self):
        """Test handling of missing correlation parameters."""
        template_dir = Path(__file__).parent.parent / 'templates'
        generator = ScriptGenerator(template_dir)

        task = TaskDefinition(
            task_type="script",
            operation="correlation",
            parameters={},  # No parameters
            data_source=None,
            user_id=123,
            conversation_id="test"
        )

        # Should handle gracefully or provide meaningful error
        try:
            script = generator.generate(task)
            # If it generates, should have default behavior
            assert 'correlation' in script.lower()
        except (ScriptGenerationError, Exception) as e:
            # If it fails, should have meaningful error message
            assert "parameter" in str(e).lower() or "column" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])