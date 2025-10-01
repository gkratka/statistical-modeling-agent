"""
Unit tests for language generator.

Tests plain language summary generation for statistical and ML results.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.processors.language_generator import LanguageGenerator
from src.processors.dataclasses import ProcessorConfig


class TestLanguageGenerator:
    """Test language generator."""

    @pytest.fixture
    def generator(self):
        """Create language generator with default config."""
        config = ProcessorConfig()
        return LanguageGenerator(config)

    @pytest.fixture
    def technical_generator(self):
        """Create language generator with technical style."""
        config = ProcessorConfig(language_style="technical", use_emojis=False)
        return LanguageGenerator(config)

    def test_generator_initialization(self, generator):
        """Test generator initializes with config."""
        assert generator.config is not None
        assert isinstance(generator.config, ProcessorConfig)

    def test_generate_descriptive_stats_summary(self, generator):
        """Test descriptive statistics summary."""
        stats = {
            'age': {
                'mean': 35.5,
                'median': 35.0,
                'std': 5.2,
                'min': 25.0,
                'max': 45.0
            }
        }

        summary = generator.generate_descriptive_stats_summary(stats)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'age' in summary.lower()
        # Friendly mode should have explanation
        assert any(word in summary.lower() for word in ['average', 'typical', 'spread'])

    def test_generate_correlation_summary(self, generator):
        """Test correlation summary."""
        correlations = {
            'age_income': 0.85,
            'age_score': 0.72,
            'income_score': 0.68
        }

        summary = generator.generate_correlation_summary(correlations)

        assert isinstance(summary, str)
        assert 'strong' in summary.lower() or 'high' in summary.lower()
        assert 'correlation' in summary.lower()

    def test_generate_regression_summary(self, generator):
        """Test regression results summary."""
        regression_result = {
            'r_squared': 0.82,
            'coefficients': {
                'intercept': 15000.0,
                'age': 1250.5,
                'experience': 3500.2
            },
            'p_values': {
                'age': 0.001,
                'experience': 0.003
            }
        }

        summary = generator.generate_regression_summary(regression_result)

        assert isinstance(summary, str)
        assert 'model' in summary.lower()
        assert '82' in summary or '0.82' in summary
        # Should explain variance
        assert 'variance' in summary.lower() or 'variation' in summary.lower()

    def test_generate_ml_training_summary(self, generator):
        """Test ML training summary."""
        training_result = {
            'model_type': 'neural_network',
            'metrics': {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.91,
                'f1_score': 0.90
            },
            'training_time': 45.2
        }

        summary = generator.generate_ml_training_summary(training_result)

        assert isinstance(summary, str)
        assert 'neural' in summary.lower() or 'network' in summary.lower()
        assert '92' in summary or '0.92' in summary
        # Should mention performance
        assert any(word in summary.lower() for word in ['accuracy', 'performance', 'correct'])

    def test_generate_ml_prediction_summary(self, generator):
        """Test ML prediction summary."""
        predictions = {
            'n_predictions': 150,
            'prediction_summary': {
                'mean': 45.2,
                'median': 44.5,
                'min': 10.2,
                'max': 89.7
            }
        }

        summary = generator.generate_ml_prediction_summary(predictions)

        assert isinstance(summary, str)
        assert '150' in summary
        assert 'prediction' in summary.lower()

    def test_technical_style_summary(self, technical_generator):
        """Test technical language style."""
        stats = {
            'value': {
                'mean': 50.0,
                'std': 10.0
            }
        }

        summary = technical_generator.generate_descriptive_stats_summary(stats)

        assert isinstance(summary, str)
        # Technical style should use precise terms
        assert 'mean' in summary.lower() or 'standard deviation' in summary.lower()
        # Should not have emojis
        assert '📊' not in summary
        assert '🎯' not in summary

    def test_friendly_style_summary(self, generator):
        """Test friendly language style."""
        stats = {
            'value': {
                'mean': 50.0,
                'std': 10.0
            }
        }

        summary = generator.generate_descriptive_stats_summary(stats)

        assert isinstance(summary, str)
        # Friendly style should have plain language
        assert any(word in summary.lower() for word in
                  ['average', 'typical', 'usually', 'spread', 'varies'])

    def test_interpret_correlation_strength(self, generator):
        """Test correlation strength interpretation."""
        # Strong positive
        assert 'strong' in generator.interpret_correlation_strength(0.85).lower()

        # Moderate
        assert 'moderate' in generator.interpret_correlation_strength(0.55).lower()

        # Weak
        assert 'weak' in generator.interpret_correlation_strength(0.25).lower()

        # Negative
        result = generator.interpret_correlation_strength(-0.75)
        assert 'strong' in result.lower()
        assert 'negative' in result.lower() or 'inverse' in result.lower()

    def test_interpret_p_value(self, generator):
        """Test p-value interpretation."""
        # Significant
        assert 'significant' in generator.interpret_p_value(0.01).lower()

        # Borderline
        assert 'borderline' in generator.interpret_p_value(0.08).lower() or \
               'marginally' in generator.interpret_p_value(0.08).lower()

        # Not significant
        result = generator.interpret_p_value(0.15)
        assert ('not' in result.lower() and 'significant' in result.lower()) or \
               'no strong evidence' in result.lower()

    def test_format_percentage(self, generator):
        """Test percentage formatting."""
        assert '85%' in generator.format_percentage(0.85)
        assert '92.5%' in generator.format_percentage(0.925)
        assert '100%' in generator.format_percentage(1.0)
        assert '0%' in generator.format_percentage(0.0)

    def test_format_large_number(self, generator):
        """Test large number formatting."""
        # Thousands
        result = generator.format_large_number(5000)
        assert '5' in result and 'thousand' in result.lower()

        # Millions
        result = generator.format_large_number(2500000)
        assert '2.5' in result and 'million' in result.lower()

    def test_error_handling_empty_input(self, generator):
        """Test error handling for empty input."""
        with pytest.raises(ValueError):
            generator.generate_descriptive_stats_summary({})

    def test_error_handling_invalid_metrics(self, generator):
        """Test error handling for invalid ML metrics."""
        invalid_result = {
            'model_type': 'unknown',
            'metrics': {}  # Empty metrics
        }

        with pytest.raises(ValueError):
            generator.generate_ml_training_summary(invalid_result)

    def test_emoji_usage_based_on_config(self):
        """Test emoji usage respects config."""
        # With emojis
        config_with = ProcessorConfig(use_emojis=True)
        gen_with = LanguageGenerator(config_with)

        # Without emojis
        config_without = ProcessorConfig(use_emojis=False)
        gen_without = LanguageGenerator(config_without)

        stats = {'x': {'mean': 50.0}}

        summary_with = gen_with.generate_descriptive_stats_summary(stats)
        summary_without = gen_without.generate_descriptive_stats_summary(stats)

        # Check that emojis appear only when enabled
        emoji_chars = ['📊', '🎯', '✅', '⚠️', '💡']
        has_emoji_with = any(emoji in summary_with for emoji in emoji_chars)
        has_emoji_without = any(emoji in summary_without for emoji in emoji_chars)

        # At least one should have emoji when enabled, none when disabled
        assert not has_emoji_without
