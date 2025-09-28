"""
Telegram Result Formatter for the Statistical Modeling Agent.

This module formats statistical analysis results into Telegram-friendly
Markdown messages with proper formatting, emojis, and user experience.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramResultFormatter:
    """Formats analysis results for Telegram display with Markdown formatting."""

    MAX_MESSAGE_LENGTH = 4096
    MAX_CAPTION_LENGTH = 1024
    DEFAULT_PRECISION = 4
    MAX_COLUMNS_DETAILED = 5
    MAX_CORRELATIONS_DETAILED = 10

    def __init__(self, precision: int = DEFAULT_PRECISION, use_emojis: bool = True, compact_mode: bool = False) -> None:
        """Initialize the result formatter."""
        self.precision = precision
        self.use_emojis = use_emojis
        self.compact_mode = compact_mode
        self.logger = logger

    def format_stats_result(self, result: Dict[str, Any]) -> str:
        """Format statistics results for Telegram display."""
        try:
            if not result.get("success", False):
                return self._format_error(result)

            operation = result.get("operation", "")
            stats_data = result.get("result", {})

            if operation in ["descriptive_stats", "summary_analysis"]:
                return self._format_descriptive_stats(stats_data, result)
            elif operation == "correlation_analysis":
                return self._format_correlation_results(stats_data, result)
            elif operation in ["mean_analysis", "median_analysis", "std_analysis"]:
                return self._format_single_stat_analysis(stats_data, result)
            else:
                return self._format_generic_stats(stats_data, result)

        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            return self._format_formatting_error(str(e))

    def _format_descriptive_stats(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """Format descriptive statistics with tables and organization."""
        emoji = "📊" if self.use_emojis else ""
        message_parts = [f"{emoji} **Descriptive Statistics Results**\n"]

        # Get metadata for context
        metadata = result.get("metadata", {})
        execution_time = metadata.get("execution_time", 0)
        data_shape = metadata.get("data_shape", [0, 0])

        # Process columns (exclude summary)
        columns_data = {k: v for k, v in data.items() if k != "summary"}
        summary = data.get("summary", {})

        if not columns_data:
            return self._format_error({
                "error": "No statistical data found in results",
                "error_code": "NO_DATA"
            })

        # Determine if we should use compact mode for many columns
        use_compact = self.compact_mode or len(columns_data) > self.MAX_COLUMNS_DETAILED

        if use_compact:
            message_parts.extend(self._format_stats_compact(columns_data))
        else:
            message_parts.extend(self._format_stats_detailed(columns_data))

        # Add summary information
        message_parts.extend(self._format_stats_summary(summary, execution_time, data_shape))

        formatted_message = "\n".join(message_parts)
        return self._ensure_telegram_length(formatted_message)

    def _format_stats_detailed(self, columns_data: Dict[str, Any]) -> List[str]:
        """Format statistics in detailed mode with full information."""
        parts = []
        stat_formats = {
            'mean': '• Mean: **{}**',
            'median': '• Median: **{}**',
            'std': '• Std Dev: **{}**'
        }

        for column, stats in columns_data.items():
            if not isinstance(stats, dict):
                continue

            emoji = "📈" if self.use_emojis else "•"
            parts.append(f"\n{emoji} **{column.upper()}**")

            # Core statistics using template
            for stat, template in stat_formats.items():
                if stats.get(stat) is not None:
                    parts.append(template.format(self._format_number(stats[stat])))

            # Range and data info
            if stats.get('min') is not None and stats.get('max') is not None:
                parts.append(f"• Range: Min: {self._format_number(stats['min'])} | Max: {self._format_number(stats['max'])}")

            data_info = []
            if stats.get('count') is not None:
                data_info.append(f"Count: {stats['count']}")
            if stats.get('missing', 0) > 0:
                data_info.append(f"Missing: {stats['missing']}")
            if data_info:
                parts.append(f"• Data: {' | '.join(data_info)}")

            # Quartiles
            if q := stats.get('quartiles'):
                if q.get('q1') is not None and q.get('q3') is not None:
                    parts.append(f"• Quartiles: Q1={self._format_number(q['q1'])}, Q3={self._format_number(q['q3'])}")

        return parts

    def _format_stats_compact(self, columns_data: Dict[str, Any]) -> List[str]:
        """Format statistics in compact mode for many columns."""
        parts = []

        # Create table header
        parts.append("\n| Column | Mean | Median | Std Dev |")
        parts.append("|--------|------|--------|---------|")

        for column, stats in columns_data.items():
            if not isinstance(stats, dict):
                continue

            mean_val = self._format_number(stats.get('mean', 'N/A'), short=True)
            median_val = self._format_number(stats.get('median', 'N/A'), short=True)
            std_val = self._format_number(stats.get('std', 'N/A'), short=True)

            parts.append(f"| {column[:15]} | {mean_val} | {median_val} | {std_val} |")

        return parts

    def _format_correlation_results(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """Format correlation analysis results."""
        emoji = "🔍" if self.use_emojis else ""
        message_parts = [f"{emoji} **Correlation Analysis Results**\n"]

        correlation_matrix = data.get("correlation_matrix", {})
        significant_correlations = data.get("significant_correlations", [])
        summary = data.get("summary", {})

        # Format correlation matrix (compact for Telegram)
        if correlation_matrix:
            message_parts.extend(self._format_correlation_matrix(correlation_matrix))

        # Format significant correlations
        message_parts.extend(self._format_significant_correlations(significant_correlations))

        # Add correlation summary
        message_parts.extend(self._format_correlation_summary(summary))

        formatted_message = "\n".join(message_parts)
        return self._ensure_telegram_length(formatted_message)

    def _format_correlation_matrix(self, matrix: Dict[str, Any]) -> List[str]:
        """Format correlation matrix as a readable table."""
        parts = ["\n🔗 **Correlation Matrix**"]

        if not matrix:
            parts.append("• No correlations calculated")
            return parts

        # Get variables
        variables = list(matrix.keys())

        if len(variables) <= 4:  # Detailed matrix for small datasets
            parts.append("\n| Variable | " + " | ".join(variables) + " |")
            parts.append("|" + "---|" * (len(variables) + 1))

            for var1 in variables:
                row_values = []
                for var2 in variables:
                    corr_val = matrix.get(var1, {}).get(var2, 0.0)
                    row_values.append(self._format_number(corr_val, short=True))
                parts.append(f"| {var1} | " + " | ".join(row_values) + " |")
        else:  # Summary for large matrices
            parts.append("• Matrix too large for display")
            parts.append(f"• Variables: {', '.join(variables[:5])}")
            if len(variables) > 5:
                parts.append(f"• ... and {len(variables) - 5} more")

        return parts

    def _format_significant_correlations(self, correlations: List[Dict[str, Any]]) -> List[str]:
        """Format significant correlations list."""
        parts = ["\n⚡ **Significant Correlations**"]

        if not correlations:
            parts.append("• No significant correlations found")
            return parts

        # Show top correlations
        max_show = min(len(correlations), self.MAX_CORRELATIONS_DETAILED)

        for i, corr in enumerate(correlations[:max_show]):
            col1 = corr.get("column1", "")
            col2 = corr.get("column2", "")
            value = corr.get("correlation", 0.0)

            strength = self._get_correlation_strength(abs(value))
            emoji = self._get_correlation_emoji(value)

            parts.append(f"• {col1} ↔ {col2}: **{self._format_number(value)}** {emoji} ({strength})")

        if len(correlations) > max_show:
            parts.append(f"• ... and {len(correlations) - max_show} more correlations")

        return parts

    def _format_single_stat_analysis(
        self,
        data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """Format single statistic analysis (mean, median, std only)."""
        operation = result.get("operation", "")
        stat_name = operation.replace("_analysis", "").title()

        emoji = "📊" if self.use_emojis else ""
        message_parts = [f"{emoji} **{stat_name} Analysis Results**\n"]

        for column, stats in data.items():
            if column == "summary" or not isinstance(stats, dict):
                continue

            # Extract the specific statistic
            stat_key = operation.replace("_analysis", "")
            if stat_key in stats:
                value = stats[stat_key]
                message_parts.append(f"• **{column}**: {self._format_number(value)}")

        return "\n".join(message_parts)

    def _format_error(self, result: Dict[str, Any]) -> str:
        """Format error messages with helpful information."""
        error_msg = result.get("error", "Unknown error occurred")
        error_code = result.get("error_code", "UNKNOWN_ERROR")
        task_type = result.get("task_type", "")
        operation = result.get("operation", "")

        emoji = "❌" if self.use_emojis else ""
        message_parts = [f"{emoji} **Error Processing Request**\n"]

        # Main error message
        message_parts.append(f"**Issue:** {self._escape_markdown(error_msg)}")

        # Context information
        if task_type:
            message_parts.append(f"**Task:** {task_type}")
        if operation:
            message_parts.append(f"**Operation:** {operation}")

        # Add helpful suggestions based on error type
        suggestions = self._get_error_suggestions(error_code, error_msg)
        if suggestions:
            message_parts.append("\n**Suggestions:**")
            message_parts.extend([f"• {suggestion}" for suggestion in suggestions])

        return "\n".join(message_parts)

    def _format_stats_summary(
        self,
        summary: Dict[str, Any],
        execution_time: float,
        data_shape: List[int]
    ) -> List[str]:
        """Format summary information for statistics."""
        parts = ["\n📋 **Summary**"]

        if summary.get("total_columns"):
            parts.append(f"• Columns analyzed: **{summary['total_columns']}**")

        if summary.get("missing_strategy"):
            parts.append(f"• Missing data: {summary['missing_strategy']} strategy")

        if data_shape and len(data_shape) >= 2:
            parts.append(f"• Dataset: {data_shape[0]:,} rows × {data_shape[1]} columns")

        if execution_time:
            parts.append(f"• Processed in: {execution_time:.3f}s")

        return parts

    def _format_correlation_summary(self, summary: Dict[str, Any]) -> List[str]:
        """Format correlation analysis summary."""
        parts = ["\n📋 **Analysis Summary**"]

        method = summary.get("method", "").title()
        if method:
            parts.append(f"• Method: {method} correlation")

        total_pairs = summary.get("total_pairs", 0)
        significant_pairs = summary.get("significant_pairs", 0)
        if total_pairs > 0:
            parts.append(f"• Pairs analyzed: {total_pairs}")
            parts.append(f"• Significant pairs: {significant_pairs}")

        strongest = summary.get("strongest_correlation", {})
        if strongest and strongest.get("pair"):
            pair = strongest["pair"]
            value = strongest.get("value", 0)
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                parts.append(f"• Strongest: {pair[0]} ↔ {pair[1]} ({self._format_number(value)})")

        threshold = summary.get("significance_threshold")
        if threshold:
            parts.append(f"• Significance threshold: {threshold}")

        return parts

    def _format_number(self, value: Any, short: bool = False) -> str:
        """Format numbers with appropriate precision."""
        if value is None or value == 'N/A':
            return 'N/A'

        try:
            num_value = float(value)

            # Handle special values
            if pd.isna(num_value):
                return 'N/A'
            if abs(num_value) == float('inf'):
                return '∞' if num_value > 0 else '-∞'

            # Determine precision based on magnitude
            precision = 2 if short else self.precision

            if abs(num_value) >= 1000000:
                return f"{num_value/1000000:.{precision-2}f}M"
            elif abs(num_value) >= 1000:
                return f"{num_value/1000:.{precision-1}f}K"
            elif abs(num_value) < 0.01 and num_value != 0:
                return f"{num_value:.2e}"
            else:
                return f"{num_value:.{precision}f}"

        except (ValueError, TypeError):
            return str(value)

    def _get_correlation_strength(self, abs_value: float) -> str:
        """Get correlation strength description."""
        if abs_value >= 0.9:
            return "Very Strong"
        elif abs_value >= 0.7:
            return "Strong"
        elif abs_value >= 0.5:
            return "Moderate"
        elif abs_value >= 0.3:
            return "Weak"
        else:
            return "Very Weak"

    def _get_correlation_emoji(self, value: float) -> str:
        """Get emoji for correlation strength and direction."""
        if not self.use_emojis:
            return ""

        abs_val = abs(value)
        if abs_val >= 0.8:
            return "🔥"
        elif abs_val >= 0.6:
            return "⚡"
        elif abs_val >= 0.4:
            return "⭐"
        else:
            return "💫"

    def _get_error_suggestions(self, error_code: str, error_msg: str) -> List[str]:
        """Get helpful suggestions based on error type."""
        suggestions = []

        if "VALIDATION_ERROR" in error_code:
            if "column" in error_msg.lower():
                suggestions.append("Check available column names with your data")
                suggestions.append("Use exact column names (case sensitive)")
            elif "task" in error_msg.lower():
                suggestions.append("Try: 'Calculate statistics for column_name'")
                suggestions.append("Try: 'Show correlation matrix'")

        elif "DATA_ERROR" in error_code:
            if "missing" in error_msg.lower():
                suggestions.append("Consider removing rows with missing data")
                suggestions.append("Try uploading a cleaned dataset")
            elif "empty" in error_msg.lower():
                suggestions.append("Upload a file with data rows")
                suggestions.append("Check your CSV file format")

        elif "TIMEOUT_ERROR" in error_code:
            suggestions.append("Try with a smaller dataset")
            suggestions.append("Request analysis for fewer columns")

        if not suggestions:
            suggestions.append("Upload a CSV file if you haven't already")
            suggestions.append("Use clear language like 'calculate mean for sales'")

        return suggestions

    def _escape_markdown(self, text: str) -> str:
        """Escape special Markdown characters for Telegram."""
        # Escape Telegram MarkdownV2 special characters
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        escaped_text = text
        for char in special_chars:
            escaped_text = escaped_text.replace(char, f'\\{char}')
        return escaped_text

    def _ensure_telegram_length(self, message: str) -> str:
        """Ensure message fits within Telegram limits."""
        if len(message) <= self.MAX_MESSAGE_LENGTH:
            return message

        # Truncate and add notice
        truncated = message[:self.MAX_MESSAGE_LENGTH - 100]
        last_newline = truncated.rfind('\n')
        if last_newline > 0:
            truncated = truncated[:last_newline]

        truncated += f"\n\n⚠️ *Results truncated to fit message limits*"
        return truncated

    def _format_formatting_error(self, error: str) -> str:
        """Format error that occurred during result formatting."""
        emoji = "⚠️" if self.use_emojis else ""
        return (
            f"{emoji} **Formatting Error**\n\n"
            f"An error occurred while formatting the results:\n"
            f"`{self._escape_markdown(error)}`\n\n"
            f"The analysis may have completed successfully, but "
            f"there was an issue displaying the results."
        )

    def _format_generic_stats(self, data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Format generic statistics results as fallback."""
        emoji = "📊" if self.use_emojis else ""
        message_parts = [f"{emoji} **Statistical Analysis Results**\n"]

        # Try to format whatever data we have
        for key, value in data.items():
            if key == "summary":
                continue

            if isinstance(value, dict):
                message_parts.append(f"\n**{key.upper()}**")
                for stat_key, stat_value in value.items():
                    if isinstance(stat_value, (int, float)):
                        message_parts.append(f"• {stat_key}: {self._format_number(stat_value)}")
                    else:
                        message_parts.append(f"• {stat_key}: {stat_value}")
            else:
                message_parts.append(f"• {key}: {value}")

        return "\n".join(message_parts)


# Convenience functions
def format_stats_for_telegram(result: Dict[str, Any], **kwargs) -> str:
    """Format statistics result for Telegram using default formatter."""
    formatter = TelegramResultFormatter(**kwargs)
    return formatter.format_stats_result(result)


def format_error_for_telegram(error_msg: str, error_code: str = "ERROR", **kwargs) -> str:
    """Format error message for Telegram."""
    error_result = {
        "success": False,
        "error": error_msg,
        "error_code": error_code
    }
    formatter = TelegramResultFormatter(**kwargs)
    return formatter.format_stats_result(error_result)