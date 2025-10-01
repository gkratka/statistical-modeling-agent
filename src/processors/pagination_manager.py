"""
Pagination manager for handling large result sets.

Chunks text and images into Telegram-friendly pages with navigation.
"""

import math
import re
from typing import List, Any
from uuid import uuid4

from src.processors.dataclasses import ProcessorConfig, PaginationState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PaginationManager:
    """Manages pagination for large result sets with smart text chunking and navigation."""

    # Telegram limits
    TELEGRAM_MAX_MESSAGE_LENGTH = 4096
    TELEGRAM_MAX_CAPTION_LENGTH = 1024

    # Safety margins for headers/footers
    TEXT_OVERHEAD = 296  # Space for headers, footers, formatting

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.text_chunk_size = self.TELEGRAM_MAX_MESSAGE_LENGTH - self.TEXT_OVERHEAD
        self.image_chunk_size = config.max_charts_per_result
        logger.info(f"PaginationManager initialized: text_chunk={self.text_chunk_size}, images={self.image_chunk_size}")

    def should_paginate(
        self,
        text: str = "",
        n_images: int = 0
    ) -> bool:
        """Determine if pagination is needed based on text length and image count."""
        # Check text length
        if len(text) > self.text_chunk_size:
            return True

        # Check image count
        if n_images > self.image_chunk_size:
            return True

        return False

    def chunk_text(
        self,
        text: str,
        max_length: int = None
    ) -> List[str]:
        """Chunk text at natural boundaries (paragraphs, sentences, words) preserving readability."""
        if not text:
            return []

        max_length = max_length or self.text_chunk_size

        # If text fits, return as single chunk
        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to find good break point
            chunk = remaining[:max_length]
            break_point = self._find_break_point(chunk, remaining)

            if break_point == -1:
                # Force break at max_length if no good point found
                break_point = max_length

            chunks.append(remaining[:break_point].rstrip())
            # Don't lstrip() as it might remove important content like headers
            remaining = remaining[break_point:]
            # Only remove leading whitespace if it's just spaces/tabs, not newlines with content
            if remaining.startswith((' ', '\t')):
                remaining = remaining.lstrip(' \t')

        logger.debug(f"Chunked text into {len(chunks)} pieces")
        return chunks

    def _find_break_point(self, chunk: str, full_text: str) -> int:
        """Find optimal break point preferring paragraphs, sentences, then words."""
        min_pos = int(len(chunk) * 0.5)

        # Try paragraph break
        if (pos := chunk.rfind('\n\n')) > min_pos:
            return pos + 2

        # Try section header
        if matches := list(re.finditer(r'\n#{1,3}\s', chunk)):
            if matches[-1].start() > min_pos:
                return matches[-1].start() + 1

        # Try sentence break
        for i in range(len(chunk) - 1, min_pos, -1):
            if chunk[i] in '.!?' and i + 1 < len(chunk) and chunk[i + 1] in ' \n':
                return i + 1

        # Try line/word breaks
        for break_char in ['\n', ' ']:
            if (pos := chunk.rfind(break_char)) > min_pos:
                return pos + 1

        return -1

    def chunk_images(
        self,
        images: List[Any],
        max_per_page: int = None
    ) -> List[List[Any]]:
        """Chunk images into pages based on maximum images per page."""
        if not images:
            return []

        max_per_page = max_per_page or self.image_chunk_size

        if len(images) <= max_per_page:
            return [images]

        chunks = [images[i:i + max_per_page] for i in range(0, len(images), max_per_page)]
        logger.debug(f"Chunked {len(images)} images into {len(chunks)} pages")
        return chunks

    def create_pagination_state(
        self,
        result_id: str,
        total_chunks: int,
        current_page: int = 1
    ) -> PaginationState:
        """Create pagination state object with validation."""
        return PaginationState(
            result_id=result_id,
            current_page=current_page,
            total_pages=total_chunks,
            chunk_size=self.text_chunk_size
        )

    def get_page_header(self, state: PaginationState) -> str:
        """Generate header for paginated page showing current position."""
        emoji = "📄 " if self.config.use_emojis else ""
        return f"{emoji}**Page {state.current_page} of {state.total_pages}**\n\n"

    def get_page_footer(self, state: PaginationState) -> str:
        """Generate footer with navigation hints based on current position."""
        if state.current_page < state.total_pages:
            emoji = "➡️ " if self.config.use_emojis else ""
            return f"\n\n{emoji}*More results available. Use /next to continue.*"
        else:
            emoji = "✅ " if self.config.use_emojis else ""
            return f"\n\n{emoji}*End of results*"

    def format_pagination_controls(self, state: PaginationState) -> str:
        """Format pagination navigation controls with progress and commands."""
        emoji = "📊 " if self.config.use_emojis else ""
        result = f"{emoji}Results: {state.current_page}/{state.total_pages}"

        if state.current_page < state.total_pages:
            result += " • /next for more"

        if state.current_page > 1:
            result += " • /prev for previous"

        return result

    def calculate_total_pages(
        self,
        total_length: int,
        chunk_size: int = None
    ) -> int:
        """Calculate total pages needed for given content length."""
        chunk_size = chunk_size or self.text_chunk_size
        return math.ceil(total_length / chunk_size)

    def generate_result_id(self) -> str:
        """Generate unique result ID for pagination session."""
        return str(uuid4())
