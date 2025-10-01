"""
Result processing module for user-friendly output.

This module processes raw analysis results into Telegram-ready outputs
with visualizations, plain language summaries, and pagination.
"""

from src.processors.dataclasses import (
    ProcessedResult,
    ImageData,
    FileData,
    ProcessorConfig,
    PaginationState
)
from src.processors.result_processor import ResultProcessor
from src.processors.visualization_generator import VisualizationGenerator
from src.processors.language_generator import LanguageGenerator
from src.processors.pagination_manager import PaginationManager

__all__ = [
    "ProcessedResult",
    "ImageData",
    "FileData",
    "ProcessorConfig",
    "PaginationState",
    "ResultProcessor",
    "VisualizationGenerator",
    "LanguageGenerator",
    "PaginationManager",
]
