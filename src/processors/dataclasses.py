"""
Data structures for result processing.

Defines immutable dataclasses for processed results, images, files,
and processor configuration.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional, Any


def _validate_min_value(value: Any, min_value: Any, name: str) -> None:
    """Validate value is >= minimum."""
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")


def _validate_in_choices(value: Any, choices: List[Any], name: str) -> None:
    """Validate value is in allowed choices."""
    if value not in choices:
        raise ValueError(f"Invalid {name}: {value}. Must be one of {choices}")


def _validate_not_empty(value: str, name: str) -> None:
    """Validate string is not empty."""
    if not value:
        raise ValueError(f"{name} cannot be empty")


@dataclass(frozen=True)
class ImageData:
    """Image data for Telegram output with BytesIO buffer and caption."""
    buffer: BytesIO
    caption: str
    format: str = "png"

    def __post_init__(self):
        if self.format not in ["png", "jpeg", "jpg"]:
            raise ValueError(f"Invalid image format: {self.format}")


@dataclass(frozen=True)
class FileData:
    """File attachment for Telegram with buffer, filename, and MIME type."""
    buffer: BytesIO
    filename: str
    mime_type: str = "text/csv"

    def __post_init__(self):
        _validate_not_empty(self.filename, "Filename")


@dataclass(frozen=True)
class PaginationState:
    """Pagination state with result ID, page numbers, and chunk size."""
    result_id: str
    current_page: int
    total_pages: int
    chunk_size: int

    def __post_init__(self):
        _validate_min_value(self.current_page, 1, "Current page")
        _validate_min_value(self.total_pages, 1, "Total pages")
        _validate_min_value(self.chunk_size, 1, "Chunk size")

        if self.current_page > self.total_pages:
            raise ValueError(
                f"Current page ({self.current_page}) cannot exceed "
                f"total pages ({self.total_pages})"
            )


@dataclass
class ProcessorConfig:
    """Result processor configuration with visualization, styling, and output preferences."""
    enable_visualizations: bool = True
    detail_level: str = "balanced"
    language_style: str = "friendly"
    plot_theme: str = "default"
    use_emojis: bool = True
    max_charts_per_result: int = 5
    image_dpi: int = 100
    image_max_size_mb: int = 5

    def __post_init__(self):
        _validate_in_choices(self.detail_level, ["compact", "balanced", "detailed"], "detail_level")
        _validate_in_choices(self.language_style, ["technical", "friendly"], "language_style")
        _validate_in_choices(self.plot_theme, ["default", "dark"], "plot_theme")
        _validate_min_value(self.max_charts_per_result, 1, "max_charts_per_result")
        _validate_min_value(self.image_dpi, 50, "image_dpi")
        _validate_min_value(self.image_max_size_mb, 1, "image_max_size_mb")


@dataclass(frozen=True)
class ProcessedResult:
    """Complete processed result with text, images, files, summary, and pagination state."""
    text: str
    summary: str
    images: List[ImageData] = field(default_factory=list)
    files: List[FileData] = field(default_factory=list)
    needs_pagination: bool = False
    pagination_state: Optional[PaginationState] = None

    def __post_init__(self):
        _validate_not_empty(self.text, "Text")
        _validate_not_empty(self.summary, "Summary")

        if self.needs_pagination and self.pagination_state is None:
            raise ValueError(
                "pagination_state must be provided when needs_pagination is True"
            )
