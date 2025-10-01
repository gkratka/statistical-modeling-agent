"""
Unit tests for visualization generator.

Tests chart generation for statistical and ML visualizations.
"""

import pytest
import numpy as np
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

from src.processors.visualization_generator import VisualizationGenerator
from src.processors.dataclasses import ProcessorConfig, ImageData


class TestVisualizationGenerator:
    """Test visualization generator."""

    @pytest.fixture
    def generator(self):
        """Create visualization generator with default config."""
        config = ProcessorConfig()
        return VisualizationGenerator(config)

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50],
            'income': [30000, 45000, 55000, 65000, 80000, 95000],
            'score': [75, 82, 88, 91, 95, 98]
        })

    def test_generator_initialization(self, generator):
        """Test generator initializes with config."""
        assert generator.config is not None
        assert isinstance(generator.config, ProcessorConfig)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_histogram(self, mock_close, mock_savefig, generator, sample_data):
        """Test histogram generation."""
        image = generator.generate_histogram(
            data=sample_data,
            column='age',
            bins=5
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'histogram' in image.caption.lower() or 'age' in image.caption.lower()
        assert image.format == 'png'
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_boxplot(self, mock_close, mock_savefig, generator, sample_data):
        """Test boxplot generation."""
        image = generator.generate_boxplot(
            data=sample_data,
            columns=['age', 'income']
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'boxplot' in image.caption.lower() or 'box' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_correlation_heatmap(self, mock_close, mock_savefig, generator, sample_data):
        """Test correlation heatmap generation."""
        corr_matrix = sample_data.corr()

        image = generator.generate_correlation_heatmap(corr_matrix)

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'correlation' in image.caption.lower() or 'heatmap' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_scatter_plot(self, mock_close, mock_savefig, generator, sample_data):
        """Test scatter plot generation."""
        image = generator.generate_scatter_plot(
            x=sample_data['age'],
            y=sample_data['income'],
            x_label='Age',
            y_label='Income'
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'scatter' in image.caption.lower() or 'age' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_confusion_matrix(self, mock_close, mock_savefig, generator):
        """Test confusion matrix generation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        labels = ['Class 0', 'Class 1']

        image = generator.generate_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'confusion' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_roc_curve(self, mock_close, mock_savefig, generator):
        """Test ROC curve generation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.85])

        image = generator.generate_roc_curve(
            y_true=y_true,
            y_proba=y_proba
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'roc' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_feature_importance(self, mock_close, mock_savefig, generator):
        """Test feature importance chart generation."""
        importances = {
            'age': 0.35,
            'income': 0.45,
            'score': 0.20
        }

        image = generator.generate_feature_importance(
            importances=importances,
            top_n=10
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'feature' in image.caption.lower() or 'importance' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_residual_plot(self, mock_close, mock_savefig, generator):
        """Test residual plot generation."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 19, 31, 38, 52])

        image = generator.generate_residual_plot(
            y_true=y_true,
            y_pred=y_pred
        )

        assert isinstance(image, ImageData)
        assert isinstance(image.buffer, BytesIO)
        assert 'residual' in image.caption.lower()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    def test_error_handling_empty_data(self, generator):
        """Test error handling for empty data."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            generator.generate_histogram(empty_df, 'nonexistent', 5)

    def test_error_handling_invalid_column(self, generator, sample_data):
        """Test error handling for invalid column."""
        with pytest.raises((ValueError, KeyError)):
            generator.generate_histogram(sample_data, 'nonexistent_column', 5)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_theme_application(self, mock_close, mock_savefig):
        """Test dark theme is applied."""
        config = ProcessorConfig(plot_theme='dark')
        generator = VisualizationGenerator(config)

        sample_data = pd.DataFrame({'x': [1, 2, 3]})
        generator.generate_histogram(sample_data, 'x', 3)

        # Verify savefig was called (theme application happens internally)
        assert mock_savefig.called
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_dpi_configuration(self, mock_close, mock_savefig):
        """Test DPI configuration is respected."""
        config = ProcessorConfig(image_dpi=150)
        generator = VisualizationGenerator(config)

        sample_data = pd.DataFrame({'x': [1, 2, 3]})
        generator.generate_histogram(sample_data, 'x', 3)

        # Verify savefig was called with dpi parameter
        call_kwargs = mock_savefig.call_args[1]
        assert 'dpi' in call_kwargs
        assert call_kwargs['dpi'] == 150
