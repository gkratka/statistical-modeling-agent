"""
Unit tests for Markdown escaping utilities.

Tests the escape_markdown_v1 function to ensure it properly escapes
special Markdown characters for Telegram bot messages.
"""

import pytest

from src.bot.utils.markdown_escape import escape_markdown_v1


class TestEscapeMarkdownV1:
    """Test escape_markdown_v1 function with various inputs."""

    def test_escape_single_underscore(self):
        """Test escaping string with single underscore (Keras initializers)."""
        assert escape_markdown_v1("glorot_uniform") == "glorot\\_uniform"
        assert escape_markdown_v1("random_normal") == "random\\_normal"
        assert escape_markdown_v1("random_uniform") == "random\\_uniform"
        assert escape_markdown_v1("he_normal") == "he\\_normal"
        assert escape_markdown_v1("he_uniform") == "he\\_uniform"

    def test_escape_multiple_underscores(self):
        """Test escaping string with multiple underscores."""
        assert escape_markdown_v1("user_id_123") == "user\\_id\\_123"
        assert escape_markdown_v1("file_name_test") == "file\\_name\\_test"
        assert escape_markdown_v1("a_b_c_d") == "a\\_b\\_c\\_d"

    def test_escape_asterisks(self):
        """Test escaping asterisks (bold formatting)."""
        assert escape_markdown_v1("**bold**") == "\\*\\*bold\\*\\*"
        assert escape_markdown_v1("*italic*") == "\\*italic\\*"
        assert escape_markdown_v1("test*with*asterisks") == "test\\*with\\*asterisks"

    def test_escape_backticks(self):
        """Test escaping backticks (code formatting)."""
        assert escape_markdown_v1("`code`") == "\\`code\\`"
        assert escape_markdown_v1("inline `code` text") == "inline \\`code\\` text"

    def test_escape_brackets(self):
        """Test escaping square brackets (link formatting)."""
        # Only opening bracket needs escaping in Telegram Markdown v1
        assert escape_markdown_v1("[link text]") == "\\[link text]"
        assert escape_markdown_v1("text [with] brackets") == "text \\[with] brackets"

    def test_escape_backslash(self):
        """Test escaping backslashes."""
        assert escape_markdown_v1("path\\to\\file") == "path\\\\to\\\\file"
        assert escape_markdown_v1("C:\\Users\\test") == "C:\\\\Users\\\\test"

    def test_escape_mixed_characters(self):
        """Test escaping strings with multiple special characters."""
        assert escape_markdown_v1("_under*star`tick") == "\\_under\\*star\\`tick"
        assert escape_markdown_v1("[link](url)") == "\\[link](url)"  # Only [ needs escaping
        assert escape_markdown_v1("**bold** _italic_") == "\\*\\*bold\\*\\* \\_italic\\_"

    def test_no_escaping_needed(self):
        """Test strings without special characters."""
        assert escape_markdown_v1("normal text") == "normal text"
        assert escape_markdown_v1("abc123") == "abc123"
        assert escape_markdown_v1("Hello World!") == "Hello World!"
        assert escape_markdown_v1("") == ""

    def test_integer_input(self):
        """Test non-string input (should convert to string)."""
        assert escape_markdown_v1(100) == "100"
        assert escape_markdown_v1(32) == "32"

    def test_float_input(self):
        """Test float input."""
        assert escape_markdown_v1(0.2) == "0.2"
        assert escape_markdown_v1(3.14) == "3.14"

    def test_emojis_preserved(self):
        """Test that emojis are preserved (not special Markdown chars)."""
        assert escape_markdown_v1("✅ Success") == "✅ Success"
        assert escape_markdown_v1("🧠 Keras") == "🧠 Keras"
        assert escape_markdown_v1("📊 Stats") == "📊 Stats"

    def test_real_world_keras_message(self):
        """Test realistic Keras configuration message."""
        initializer = "glorot_uniform"
        message_part = f"✅ Initializer: {escape_markdown_v1(initializer)}"
        assert message_part == "✅ Initializer: glorot\\_uniform"

    def test_real_world_file_path(self):
        """Test file path with underscores."""
        path = "/Users/john_doe/data_science/test_file.csv"
        escaped = escape_markdown_v1(path)
        assert escaped == "/Users/john\\_doe/data\\_science/test\\_file.csv"

    def test_real_world_column_name(self):
        """Test column name with underscores."""
        column = "user_registration_date"
        escaped = escape_markdown_v1(column)
        assert escaped == "user\\_registration\\_date"
