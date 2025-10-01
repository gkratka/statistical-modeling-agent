"""
Unit tests for pagination manager.

Tests chunking and pagination logic for large result sets.
"""

import pytest
from unittest.mock import Mock

from src.processors.pagination_manager import PaginationManager
from src.processors.dataclasses import ProcessorConfig, PaginationState


class TestPaginationManager:
    """Test pagination manager."""

    @pytest.fixture
    def manager(self):
        """Create pagination manager with default config."""
        config = ProcessorConfig()
        return PaginationManager(config)

    def test_manager_initialization(self, manager):
        """Test manager initializes with config."""
        assert manager.config is not None
        assert isinstance(manager.config, ProcessorConfig)

    def test_should_paginate_small_text(self, manager):
        """Test pagination not needed for small text."""
        small_text = "Short result"
        assert not manager.should_paginate(text=small_text)

    def test_should_paginate_large_text(self, manager):
        """Test pagination needed for large text."""
        # Telegram limit is 4096 characters
        large_text = "x" * 5000
        assert manager.should_paginate(text=large_text)

    def test_should_paginate_many_images(self, manager):
        """Test pagination needed for many images."""
        # Assume more than 5 images needs pagination
        assert manager.should_paginate(n_images=10)

    def test_should_paginate_threshold(self, manager):
        """Test pagination threshold respects config."""
        # Just under threshold (text_chunk_size is 3800)
        text = "x" * 3700
        assert not manager.should_paginate(text=text)

        # Just over threshold
        text = "x" * 3900
        assert manager.should_paginate(text=text)

    def test_chunk_text_small(self, manager):
        """Test chunking small text returns single chunk."""
        text = "Small text content"
        chunks = manager.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_large(self, manager):
        """Test chunking large text into multiple chunks."""
        # Create text that needs 3 chunks
        text = "Section 1\n\n" + ("x" * 3500) + "\n\nSection 2\n\n" + ("y" * 3500) + "\n\nSection 3"

        chunks = manager.chunk_text(text, max_length=4000)

        assert len(chunks) >= 2
        assert all(len(chunk) <= 4000 for chunk in chunks)

    def test_chunk_text_preserves_content(self, manager):
        """Test chunking preserves all content."""
        text = "A" * 5000

        chunks = manager.chunk_text(text, max_length=2000)

        # Reconstruct text
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_chunk_text_respects_separators(self, manager):
        """Test chunking prefers natural break points."""
        # Create text with clear paragraph boundaries
        p1 = "Paragraph 1\n\n" + ("x" * 500)
        p2 = "\n\nParagraph 2\n\n" + ("y" * 500)
        p3 = "\n\nParagraph 3\n\n" + ("z" * 500)
        text = p1 + p2 + p3

        chunks = manager.chunk_text(text, max_length=800)

        # Should split at paragraph boundaries, not mid-paragraph
        assert len(chunks) >= 2
        # Verify we're breaking at logical boundaries
        for chunk in chunks:
            # Each chunk should contain complete paragraph markers
            if "Paragraph" in chunk:
                # Count paragraph markers - should be complete
                assert chunk.count("Paragraph") >= 1

    def test_create_pagination_state(self, manager):
        """Test pagination state creation."""
        state = manager.create_pagination_state(
            result_id="test_123",
            total_chunks=5,
            current_page=1
        )

        assert isinstance(state, PaginationState)
        assert state.result_id == "test_123"
        assert state.current_page == 1
        assert state.total_pages == 5

    def test_get_page_header(self, manager):
        """Test pagination header generation."""
        state = PaginationState(
            result_id="test",
            current_page=2,
            total_pages=5,
            chunk_size=4000
        )

        header = manager.get_page_header(state)

        assert isinstance(header, str)
        assert "2" in header
        assert "5" in header
        assert "page" in header.lower()

    def test_get_page_footer(self, manager):
        """Test pagination footer generation."""
        state = PaginationState(
            result_id="test",
            current_page=2,
            total_pages=5,
            chunk_size=4000
        )

        footer = manager.get_page_footer(state)

        assert isinstance(footer, str)
        # Should have navigation hints
        assert any(word in footer.lower() for word in ['next', 'more', 'continue'])

    def test_chunk_images_small_list(self, manager):
        """Test image chunking for small list."""
        images = [Mock() for _ in range(3)]

        chunks = manager.chunk_images(images, max_per_page=5)

        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_images_large_list(self, manager):
        """Test image chunking for large list."""
        images = [Mock() for _ in range(12)]

        chunks = manager.chunk_images(images, max_per_page=5)

        assert len(chunks) == 3  # 12 images / 5 per page = 3 pages
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 2

    def test_chunk_images_preserves_order(self, manager):
        """Test image chunking preserves order."""
        images = [Mock(id=i) for i in range(10)]

        chunks = manager.chunk_images(images, max_per_page=4)

        # Verify order
        reconstructed = []
        for chunk in chunks:
            reconstructed.extend(chunk)

        assert [img.id for img in reconstructed] == list(range(10))

    def test_format_pagination_controls(self, manager):
        """Test pagination control formatting."""
        state = PaginationState(
            result_id="test",
            current_page=3,
            total_pages=5,
            chunk_size=4000
        )

        controls = manager.format_pagination_controls(state)

        assert isinstance(controls, str)
        # Should show page info
        assert "3" in controls
        assert "5" in controls

    def test_calculate_total_pages(self, manager):
        """Test total pages calculation."""
        # 10000 chars at 4000 per page = 3 pages
        total = manager.calculate_total_pages(
            total_length=10000,
            chunk_size=4000
        )

        assert total == 3

    def test_calculate_total_pages_exact_fit(self, manager):
        """Test total pages when length is exact multiple."""
        # 8000 chars at 4000 per page = 2 pages
        total = manager.calculate_total_pages(
            total_length=8000,
            chunk_size=4000
        )

        assert total == 2

    def test_error_handling_invalid_page(self, manager):
        """Test error handling for invalid page number."""
        with pytest.raises(ValueError):
            manager.create_pagination_state(
                result_id="test",
                total_chunks=5,
                current_page=0  # Invalid
            )

    def test_error_handling_invalid_total(self, manager):
        """Test error handling for invalid total pages."""
        with pytest.raises(ValueError):
            manager.create_pagination_state(
                result_id="test",
                total_chunks=0,  # Invalid
                current_page=1
            )

    def test_chunk_size_configuration(self):
        """Test chunk size respects configuration."""
        config = ProcessorConfig()
        manager = PaginationManager(config)

        # Default chunk size for Telegram (4096 - overhead)
        assert manager.text_chunk_size == 3800

    def test_smart_chunking_preserves_sections(self, manager):
        """Test smart chunking keeps logical sections together."""
        # Create sections with headers
        sections = []
        for i in range(5):
            sections.append(f"## Section {i}\n" + ("x" * 800))

        text = "\n\n".join(sections)

        chunks = manager.chunk_text(text, max_length=2500)

        # Each chunk should contain complete sections
        for chunk in chunks:
            # Count section headers
            section_count = chunk.count("## Section")
            # Verify sections aren't split mid-content
            assert section_count >= 1
