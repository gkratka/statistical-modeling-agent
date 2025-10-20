"""
Base message formatting utilities for Telegram bot workflows.

This module provides shared message formatting functions used across
all workflow message classes (score, local_path, prediction, etc.).
"""

from typing import List, Optional, Dict, Any


class BaseMessages:
    """Shared message formatting utilities for all workflow handlers."""

    @staticmethod
    def format_error(
        title: str,
        details: str,
        suggestions: Optional[List[str]] = None,
        icon: str = "❌"
    ) -> str:
        """
        Format error message with consistent structure.

        Args:
            title: Error title/type
            details: Detailed error description
            suggestions: Optional list of suggestions for fixing
            icon: Emoji icon (default: ❌)

        Returns:
            Formatted error message string
        """
        msg = f"{icon} **{title}**\n\n{details}\n\n"

        if suggestions:
            msg += "**💡 Suggestions:**\n"
            msg += "\n".join(f"• {suggestion}" for suggestion in suggestions)

        return msg

    @staticmethod
    def format_path_error(
        field_name: str,
        path: str,
        error_type: str,
        allowed_dirs: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        allowed_extensions: Optional[List[str]] = None
    ) -> str:
        """
        Format path-related error messages.

        Args:
            field_name: Name of the field (e.g., "TRAIN_DATA")
            path: The problematic path
            error_type: Type of error (not_found, not_allowed, too_large, etc.)
            allowed_dirs: List of allowed directories
            max_size: Maximum file size in MB
            allowed_extensions: List of allowed file extensions

        Returns:
            Formatted path error message
        """
        error_titles = {
            'not_found': 'File Not Found',
            'not_allowed': 'Path Not Allowed',
            'too_large': 'File Too Large',
            'invalid_extension': 'Invalid File Type',
            'not_absolute': 'Invalid Path Format',
            'permission_denied': 'Permission Denied'
        }

        title = error_titles.get(error_type, 'Path Error')
        details = f"**{field_name}**: `{path}`\n\n"

        if error_type == 'not_found':
            details += "The specified file does not exist or cannot be accessed."
        elif error_type == 'not_allowed':
            details += "This path is not in the allowed directories list."
        elif error_type == 'too_large':
            details += f"File exceeds maximum size limit of {max_size}MB."
        elif error_type == 'invalid_extension':
            details += "File extension is not allowed."
        elif error_type == 'not_absolute':
            details += "Path must be absolute (start with / or ./)."
        elif error_type == 'permission_denied':
            details += "Cannot read file due to permission restrictions."

        suggestions = []

        if allowed_dirs and error_type in ('not_found', 'not_allowed'):
            suggestions.append("**Allowed directories:**")
            for dir_path in allowed_dirs:
                suggestions.append(f"  • `{dir_path}`")

        if allowed_extensions and error_type == 'invalid_extension':
            suggestions.append(f"**Allowed formats:** {', '.join(allowed_extensions)}")

        if error_type == 'not_absolute':
            suggestions.append("Use absolute paths like `/home/user/data.csv` or `./data/file.csv`")

        return BaseMessages.format_error(title, details, suggestions if suggestions else None)

    @staticmethod
    def format_success(
        title: str,
        summary: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        next_steps: Optional[List[str]] = None,
        icon: str = "✅"
    ) -> str:
        """
        Format success message with consistent structure.

        Args:
            title: Success message title
            summary: Key-value pairs for summary section
            metrics: Optional performance metrics
            next_steps: Optional list of suggested next actions
            icon: Emoji icon (default: ✅)

        Returns:
            Formatted success message string
        """
        msg = f"{icon} **{title}**\n\n"

        # Summary section
        if summary:
            msg += "**📊 Summary:**\n"
            for key, value in summary.items():
                msg += f"• {key}: {value}\n"
            msg += "\n"

        # Metrics section
        if metrics:
            msg += "**📈 Metrics:**\n"
            for key, value in metrics.items():
                formatted_value = BaseMessages._format_metric_value(value)
                msg += f"• {key}: {formatted_value}\n"
            msg += "\n"

        # Next steps section
        if next_steps:
            msg += "**💡 Next Steps:**\n"
            msg += "\n".join(f"• {step}" for step in next_steps)

        return msg

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        """Format metric value based on type."""
        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, str):
            return value
        else:
            return str(value)

    @staticmethod
    def format_list(items: List[str], max_items: int = 10) -> str:
        """
        Format a list with optional truncation.

        Args:
            items: List of items to format
            max_items: Maximum items to show before truncating

        Returns:
            Formatted list string
        """
        if len(items) <= max_items:
            return "\n".join(f"• {item}" for item in items)
        else:
            shown = items[:max_items]
            remaining = len(items) - max_items
            result = "\n".join(f"• {item}" for item in shown)
            result += f"\n• ... and {remaining} more"
            return result

    @staticmethod
    def format_progress(
        phase: str,
        current: int,
        total: int,
        elapsed_seconds: Optional[float] = None,
        icon: str = "🔄"
    ) -> str:
        """
        Format progress message.

        Args:
            phase: Current phase name
            current: Current step number
            total: Total number of steps
            elapsed_seconds: Optional elapsed time
            icon: Emoji icon (default: 🔄)

        Returns:
            Formatted progress message
        """
        msg = f"{icon} **Phase {current}/{total}:** {phase}"

        if elapsed_seconds is not None:
            msg += f"\n⏱️ Elapsed: {elapsed_seconds:.1f}s"

        return msg
