"""
Markdown escaping utilities for Telegram bot messages.

This module provides functions to safely escape special Markdown characters
in dynamic content before sending messages with parse_mode="Markdown".
"""


def escape_markdown_v1(text: str) -> str:
    """
    Escape special Markdown characters for Telegram Markdown v1.

    Telegram's Markdown parser treats these characters as formatting:
    - _ (underscore): italic text (e.g., _text_ renders as italic)
    - * (asterisk): bold text (e.g., **text** renders as bold)
    - ` (backtick): code/monospace (e.g., `code` renders as monospace)
    - [ (bracket): links (e.g., [text](url) renders as hyperlink)

    This function escapes these characters so they display as literal text
    rather than being interpreted as formatting commands.

    Args:
        text: Raw text that may contain special characters

    Returns:
        Escaped text safe for parse_mode="Markdown"

    Examples:
        >>> escape_markdown_v1("glorot_uniform")
        'glorot\\_uniform'
        >>> escape_markdown_v1("random_normal")
        'random\\_normal'
        >>> escape_markdown_v1("user_id_123")
        'user\\_id\\_123'
        >>> escape_markdown_v1("**bold**")
        '\\*\\*bold\\*\\*'
        >>> escape_markdown_v1("`code`")
        '\\`code\\`'
        >>> escape_markdown_v1("normal text")
        'normal text'
    """
    if not isinstance(text, str):
        text = str(text)

    # Escape Markdown v1 special characters
    # Order matters: escape backslash first to avoid double-escaping
    special_chars = ['\\', '_', '*', '`', '[']

    for char in special_chars:
        text = text.replace(char, f'\\{char}')

    return text
