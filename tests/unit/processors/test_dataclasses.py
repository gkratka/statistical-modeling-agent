"""
Unit tests for result processor dataclasses.

Tests ProcessedResult, ImageData, FileData, and ProcessorConfig.
"""

import pytest
from io import BytesIO
from dataclasses import FrozenInstanceError

from src.processors.dataclasses import (
    ProcessedResult,
    ImageData,
    FileData,
    ProcessorConfig,
    PaginationState
)


class TestImageData:
    """Test ImageData dataclass."""

    def test_image_data_creation(self):
        """Test creating ImageData with valid data."""
        buffer = BytesIO(b"fake image data")
        img = ImageData(
            buffer=buffer,
            caption="Test chart",
            format="png"
        )

        assert img.buffer == buffer
        assert img.caption == "Test chart"
        assert img.format == "png"

    def test_image_data_defaults(self):
        """Test ImageData default values."""
        buffer = BytesIO(b"data")
        img = ImageData(buffer=buffer, caption="Test")

        assert img.format == "png"  # Default format

    def test_image_data_immutable(self):
        """Test that ImageData is frozen/immutable."""
        img = ImageData(
            buffer=BytesIO(b"data"),
            caption="Test",
            format="jpeg"
        )

        with pytest.raises((FrozenInstanceError, AttributeError)):
            img.caption = "Modified"


class TestFileData:
    """Test FileData dataclass."""

    def test_file_data_creation(self):
        """Test creating FileData with valid data."""
        buffer = BytesIO(b"csv,data\n1,2")
        file = FileData(
            buffer=buffer,
            filename="data.csv",
            mime_type="text/csv"
        )

        assert file.buffer == buffer
        assert file.filename == "data.csv"
        assert file.mime_type == "text/csv"

    def test_file_data_defaults(self):
        """Test FileData default MIME type."""
        buffer = BytesIO(b"data")
        file = FileData(buffer=buffer, filename="test.csv")

        assert file.mime_type == "text/csv"  # Default

    def test_file_data_immutable(self):
        """Test that FileData is frozen/immutable."""
        file = FileData(
            buffer=BytesIO(b"data"),
            filename="test.csv"
        )

        with pytest.raises((FrozenInstanceError, AttributeError)):
            file.filename = "modified.csv"


class TestPaginationState:
    """Test PaginationState dataclass."""

    def test_pagination_state_creation(self):
        """Test creating PaginationState."""
        state = PaginationState(
            result_id="result_123",
            current_page=2,
            total_pages=5,
            chunk_size=10
        )

        assert state.result_id == "result_123"
        assert state.current_page == 2
        assert state.total_pages == 5
        assert state.chunk_size == 10

    def test_pagination_state_validation(self):
        """Test pagination state validation."""
        # Current page can't exceed total
        with pytest.raises(ValueError):
            PaginationState(
                result_id="test",
                current_page=6,
                total_pages=5,
                chunk_size=10
            )

        # Current page can't be zero
        with pytest.raises(ValueError):
            PaginationState(
                result_id="test",
                current_page=0,
                total_pages=5,
                chunk_size=10
            )

        # Total pages must be positive
        with pytest.raises(ValueError):
            PaginationState(
                result_id="test",
                current_page=1,
                total_pages=0,
                chunk_size=10
            )


class TestProcessorConfig:
    """Test ProcessorConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ProcessorConfig()

        assert config.enable_visualizations is True
        assert config.detail_level == "balanced"
        assert config.language_style == "friendly"
        assert config.plot_theme == "default"
        assert config.use_emojis is True
        assert config.max_charts_per_result == 5
        assert config.image_dpi == 100
        assert config.image_max_size_mb == 5

    def test_config_custom_values(self):
        """Test creating config with custom values."""
        config = ProcessorConfig(
            enable_visualizations=False,
            detail_level="compact",
            language_style="technical",
            plot_theme="dark",
            use_emojis=False,
            max_charts_per_result=3,
            image_dpi=150
        )

        assert config.enable_visualizations is False
        assert config.detail_level == "compact"
        assert config.language_style == "technical"
        assert config.plot_theme == "dark"
        assert config.use_emojis is False
        assert config.max_charts_per_result == 3
        assert config.image_dpi == 150

    def test_config_validation(self):
        """Test config validation for invalid values."""
        # Invalid detail level
        with pytest.raises(ValueError):
            ProcessorConfig(detail_level="invalid")

        # Invalid language style
        with pytest.raises(ValueError):
            ProcessorConfig(language_style="invalid")

        # Invalid plot theme
        with pytest.raises(ValueError):
            ProcessorConfig(plot_theme="invalid")

        # Invalid max charts (too low)
        with pytest.raises(ValueError):
            ProcessorConfig(max_charts_per_result=0)

        # Invalid DPI (too low)
        with pytest.raises(ValueError):
            ProcessorConfig(image_dpi=49)


class TestProcessedResult:
    """Test ProcessedResult dataclass."""

    def test_processed_result_creation(self):
        """Test creating ProcessedResult with all fields."""
        buffer = BytesIO(b"image")
        image = ImageData(buffer=buffer, caption="Chart")

        file_buffer = BytesIO(b"data")
        file = FileData(buffer=file_buffer, filename="data.csv")

        pagination = PaginationState(
            result_id="result_123",
            current_page=1,
            total_pages=3,
            chunk_size=10
        )

        result = ProcessedResult(
            text="Formatted results",
            images=[image],
            files=[file],
            summary="Plain language summary",
            needs_pagination=True,
            pagination_state=pagination
        )

        assert result.text == "Formatted results"
        assert len(result.images) == 1
        assert result.images[0] == image
        assert len(result.files) == 1
        assert result.files[0] == file
        assert result.summary == "Plain language summary"
        assert result.needs_pagination is True
        assert result.pagination_state == pagination

    def test_processed_result_minimal(self):
        """Test creating ProcessedResult with minimal fields."""
        result = ProcessedResult(
            text="Simple result",
            images=[],
            files=[],
            summary="Summary"
        )

        assert result.text == "Simple result"
        assert result.images == []
        assert result.files == []
        assert result.summary == "Summary"
        assert result.needs_pagination is False
        assert result.pagination_state is None

    def test_processed_result_defaults(self):
        """Test ProcessedResult default values."""
        result = ProcessedResult(
            text="Result",
            summary="Summary"
        )

        assert result.images == []
        assert result.files == []
        assert result.needs_pagination is False
        assert result.pagination_state is None

    def test_processed_result_immutable(self):
        """Test that ProcessedResult is frozen/immutable."""
        result = ProcessedResult(
            text="Result",
            summary="Summary"
        )

        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.text = "Modified"
