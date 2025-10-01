"""
Script command handlers for Telegram bot integration.

This module provides handlers for script generation commands and processes
script execution requests with natural language parsing and result formatting.
"""

import json
from typing import Dict, Any, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.parser import RequestParser, TaskDefinition
from src.core.orchestrator import TaskOrchestrator
from src.utils.exceptions import ParseError, ScriptGenerationError, ExecutionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def escape_user_data(text: str) -> str:
    """
    Escape only user-provided data for safe Telegram Markdown display.
    Preserves numbers, decimals, and common punctuation while escaping markdown conflicts.

    Args:
        text: User data to escape (column names, values, etc.)

    Returns:
        Text with only problematic markdown characters escaped
    """
    if not isinstance(text, str):
        text = str(text)

    # Only escape characters that could cause markdown parsing issues
    # Preserve: . (decimals), - (negative numbers, ranges)
    # Escape: characters that conflict with Telegram Markdown syntax
    markdown_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '=', '|', '{', '}', '!']

    for char in markdown_chars:
        text = text.replace(char, f'\\{char}')

    return text


class ScriptHandler:
    """Handler for script generation and execution commands."""

    def __init__(self, parser: RequestParser, orchestrator: TaskOrchestrator):
        """Initialize script handler with core components."""
        self.parser = parser
        self.orchestrator = orchestrator

    async def script_command_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /script command - show available templates.

        Usage:
        /script - Show available templates
        /script descriptive - Generate descriptive statistics script
        /script correlation - Generate correlation analysis script
        /script train_classifier - Generate ML training script
        """
        try:
            # Get command arguments
            args = context.args if context.args else []

            if not args:
                # Show available templates
                message = self._get_template_listing()
                await update.message.reply_text(message, parse_mode='Markdown')
                return

            # Parse script request
            command_text = f"/script {' '.join(args)}"
            task = self.parser.parse_request(
                text=command_text,
                user_id=update.effective_user.id,
                conversation_id=str(update.effective_chat.id)
            )

            if task.task_type != "script":
                await update.message.reply_text(
                    "❌ Invalid script command. Use `/script` to see available templates."
                )
                return

            # Get user data from context using correct key format
            user_id = update.effective_user.id
            user_data_key = f'data_{user_id}'

            if hasattr(context, 'user_data') and user_data_key in context.user_data:
                user_data = context.user_data[user_data_key]
                data = user_data.get('dataframe')
                metadata = user_data.get('metadata', {})
            else:
                data = None

            if data is None:
                await update.message.reply_text(
                    "📁 Please upload a data file first before generating scripts."
                )
                return

            # Execute script generation and execution
            await update.message.reply_text("🔄 Generating and executing script...")

            result = await self.orchestrator.execute_task(task, data)
            response_message = self.format_script_results(result)

            await update.message.reply_text(response_message, parse_mode='Markdown')

        except ParseError as e:
            logger.error(f"Parse error in script command: {str(e)}")
            await update.message.reply_text(f"❌ Parse Error: {str(e)}")
        except Exception as e:
            import traceback
            logger.error(f"Script command error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Task details: task_type={getattr(task, 'task_type', 'unknown')}, operation={getattr(task, 'operation', 'unknown')}, parameters={getattr(task, 'parameters', {})}")
            await update.message.reply_text(f"❌ Script command error: {str(e)}")

    async def script_generation_handler(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle script generation requests from natural language.

        Processes messages like:
        - "Generate a script for correlation analysis"
        - "Create Python code for descriptive statistics"
        - "Make a script to train a classifier"
        """
        try:
            message_text = update.message.text

            # Parse the natural language request
            task = self.parser.parse_request(
                text=message_text,
                user_id=update.effective_user.id,
                conversation_id=str(update.effective_chat.id)
            )

            if task.task_type != "script":
                # Not a script request, ignore
                return

            # Get user data from context using correct key format
            user_id = update.effective_user.id
            user_data_key = f'data_{user_id}'

            if hasattr(context, 'user_data') and user_data_key in context.user_data:
                user_data = context.user_data[user_data_key]
                data = user_data.get('dataframe')
                metadata = user_data.get('metadata', {})
            else:
                data = None

            if data is None:
                await update.message.reply_text(
                    "📁 Please upload a data file first before generating scripts."
                )
                return

            # Optionally show script preview
            preview_shown = await self.display_script_preview(update, task)

            if not preview_shown:
                # Execute directly without preview
                await update.message.reply_text("🔄 Generating and executing script...")

                result = await self.orchestrator.execute_task(task, data)
                response_message = self.format_script_results(result)

                await update.message.reply_text(response_message, parse_mode='Markdown')

        except ParseError as e:
            await update.message.reply_text(f"❌ Parse Error: {str(e)}")
        except Exception as e:
            logger.error(f"Script generation error: {str(e)}")
            await update.message.reply_text("❌ An error occurred generating the script.")

    async def display_script_preview(
        self,
        update: Update,
        task: TaskDefinition
    ) -> bool:
        """
        Show generated script to user before execution (optional security feature).

        Args:
            update: Telegram update object
            task: Task definition for script generation

        Returns:
            True if preview was shown, False if script executed directly
        """
        try:
            # For now, execute directly without preview
            # Future enhancement: Generate script and show preview with confirm/cancel buttons
            return False

        except Exception as e:
            logger.error(f"Script preview error: {str(e)}")
            return False

    def format_script_results(self, result: Dict[str, Any]) -> str:
        """
        Format script execution results for Telegram display.

        Args:
            result: Script execution result dictionary

        Returns:
            Formatted message string
        """
        if not result.get('success', False):
            return f"❌ **Script Execution Failed**\n\n{result.get('error', 'Unknown error occurred')}"

        try:
            # Parse the output if it's JSON
            output = result.get('output', '')
            if output:
                try:
                    output_data = json.loads(output)
                    formatted_output = self._format_output_data(output_data)
                except json.JSONDecodeError:
                    formatted_output = f"```\n{output}\n```"
            else:
                formatted_output = "No output generated"

            # Format metadata
            metadata = result.get('metadata', {})
            execution_time = result.get('execution_time', 0)
            memory_usage = result.get('memory_usage', 0)

            # Escape only user data in metadata to prevent Markdown conflicts
            escaped_operation = escape_user_data(metadata.get('operation', 'Unknown'))
            escaped_template = escape_user_data(metadata.get('template_used', 'N/A'))

            message = f"""✅ **Script Executed Successfully**

**Operation**: {escaped_operation}
**Template**: {escaped_template}

**Results**:
{formatted_output}

**Performance**:
- Execution Time: {execution_time:.3f}s
- Memory Usage: {memory_usage}MB
- Security Validated: {metadata.get('security_validated', False)}
"""

            return message

        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")
            return f"✅ Script executed, but results formatting failed: {str(e)}"

    def _format_output_data(self, data: Dict[str, Any]) -> str:
        """Format structured output data for display."""
        try:
            # Handle nested result structure from script execution
            if 'result' in data and isinstance(data['result'], dict):
                return self._format_script_results(data['result'])
            elif isinstance(data, dict) and any(key in data for key in ['summary', 'correlationsummary', 'overall']):
                return self._format_script_results(data)
            else:
                # Fall back to basic formatting for other structures
                return self._format_basic_data(data)
        except Exception as e:
            logger.error(f"Output formatting error: {str(e)}")
            return f"**Raw Output**:\n```json\n{json.dumps(data, indent=2, default=str)}\n```"

    def _format_script_results(self, result: Dict[str, Any]) -> str:
        """Format statistical analysis results into readable summary."""
        sections = ["📊 **Statistical Analysis Results**"]

        # Format metadata section
        metadata_section = self._format_metadata(result)
        if metadata_section:
            sections.append(metadata_section)

        # Format column statistics
        if 'summary' in result and isinstance(result['summary'], dict):
            stats_section = self._format_statistics_summary(result['summary'])
            if stats_section:
                sections.append(stats_section)

        # Format correlation analysis
        if 'correlationsummary' in result and result['correlationsummary']:
            corr_section = self._format_correlation_summary(result['correlationsummary'])
            if corr_section:
                sections.append(corr_section)

        # Format overall dataset metrics
        if 'overall' in result and result['overall']:
            overall_section = self._format_overall_summary(result['overall'])
            if overall_section:
                sections.append(overall_section)

        return "\n\n".join(sections)

    def _format_statistics_summary(self, summary: Dict[str, Any]) -> str:
        """Format individual column statistics."""
        if not summary:
            return ""

        sections = ["**📈 Column Statistics**"]
        stat_order = ['count', 'mean', 'median', 'std', 'min', 'max', 'range']

        for column_name, stats in summary.items():
            if not isinstance(stats, dict):
                continue

            escaped_column_name = escape_user_data(column_name)
            column_lines = [f"\n**Column: {escaped_column_name}**"]

            # Format all statistics using helper method
            for stat_name in stat_order + [s for s in stats.keys() if s not in stat_order]:
                if stat_name in stats and isinstance(stats[stat_name], (int, float)):
                    formatted_stat = self._format_statistic(stat_name, stats[stat_name])
                    if formatted_stat:
                        column_lines.append(formatted_stat)

            sections.extend(column_lines)

        return "\n".join(sections)

    def _format_correlation_summary(self, corr_data: Dict[str, Any]) -> str:
        """Format correlation analysis results."""
        if not corr_data:
            return ""

        sections = ["**🔗 Correlation Analysis**"]

        # Handle correlation matrix
        if 'correlation_matrix' in corr_data:
            matrix = corr_data['correlation_matrix']
            if isinstance(matrix, dict):
                sections.append("\n**Correlation Matrix:**")
                for var1, correlations in matrix.items():
                    if isinstance(correlations, dict):
                        for var2, corr_value in correlations.items():
                            if var1 != var2 and isinstance(corr_value, (int, float)):
                                strength = self._get_correlation_strength(corr_value)
                                pair_str = self._format_variable_pair(var1, var2)
                                sections.append(f"• {pair_str}: {corr_value:.3f} ({strength})")

        # Handle strongest/weakest correlations using helper method
        for corr_type in ['strongest_correlation', 'weakest_correlation']:
            if corr_type in corr_data:
                formatted_corr = self._format_correlation_pair(corr_data[corr_type], corr_type)
                if formatted_corr:
                    sections.append(formatted_corr)

        return "\n".join(sections)

    def _format_overall_summary(self, overall: Dict[str, Any]) -> str:
        """Format overall dataset summary."""
        if not overall:
            return ""

        sections = ["**📋 Dataset Overview**"]

        # Format dataset-level metrics
        if 'total_rows' in overall:
            sections.append(f"• Total Rows: {overall['total_rows']}")
        if 'total_columns' in overall:
            sections.append(f"• Total Columns: {overall['total_columns']}")
        if 'numeric_columns' in overall:
            sections.append(f"• Numeric Columns: {overall['numeric_columns']}")
        if 'missing_values' in overall:
            sections.append(f"• Missing Values: {overall['missing_values']}")

        return "\n".join(sections)

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format analysis metadata."""
        if not metadata:
            return ""

        sections = []

        if 'columnsanalyzed' in metadata:
            columns = metadata['columnsanalyzed']
            if isinstance(columns, list) and columns:
                escaped_columns = [escape_user_data(col) for col in columns[:5]]
                column_list = ", ".join(escaped_columns)  # Show first 5 columns
                if len(columns) > 5:
                    column_list += f" and {len(columns) - 5} more"
                sections.append(f"**Columns Analyzed:** {column_list}")

        if 'statisticscalculated' in metadata:
            stats = metadata['statisticscalculated']
            if isinstance(stats, list) and stats:
                stats_list = ", ".join(stats)
                sections.append(f"**Statistics Calculated:** {stats_list}")

        return "\n".join(sections)

    def _format_statistic(self, stat_name: str, value: float) -> str:
        """Format a single statistic value consistently."""
        if stat_name == 'count':
            return f"• Count: {int(value)}"
        elif value >= 1000 or (value < 1 and value > 0):
            return f"• {stat_name.title()}: {value:.2f}"
        else:
            return f"• {stat_name.title()}: {value:.4f}"

    def _format_variable_pair(self, var1: str, var2: str) -> str:
        """Format a pair of variables consistently."""
        escaped_var1 = escape_user_data(var1)
        escaped_var2 = escape_user_data(var2)
        return f"{escaped_var1} <-> {escaped_var2}"

    def _format_correlation_pair(self, corr_data: Dict[str, Any], corr_type: str) -> str:
        """Format strongest/weakest correlation pairs."""
        if not isinstance(corr_data, dict) or 'correlation' not in corr_data:
            return ""

        title = "Strongest" if "strongest" in corr_type else "Weakest"
        variables = corr_data.get('variables', [])

        if isinstance(variables, list) and len(variables) >= 2:
            variables_str = f"{escape_user_data(variables[0])} ↔ {escape_user_data(variables[1])}"
        else:
            variables_str = escape_user_data(str(variables))

        return f"\n**{title} Correlation:**\n• {variables_str}: {corr_data['correlation']:.3f}"

    def _get_correlation_strength(self, correlation: float) -> str:
        """Get human-readable correlation strength description."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "Very Strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def _format_basic_data(self, data: Dict[str, Any]) -> str:
        """Format basic data structures (fallback method)."""
        if isinstance(data, dict):
            formatted_lines = []
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    formatted_lines.append(f"• **{key}**: {value:.4f}")
                elif isinstance(value, dict):
                    formatted_lines.append(f"• **{key}**:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            formatted_lines.append(f"  - {sub_key}: {sub_value:.4f}")
                        elif isinstance(sub_value, list):
                            formatted_values = ", ".join(str(v) for v in sub_value)
                            formatted_lines.append(f"  - {sub_key}: ({formatted_values})")
                        else:
                            formatted_lines.append(f"  - {sub_key}: {sub_value}")
                else:
                    formatted_lines.append(f"• **{key}**: {value}")
            return '\n'.join(formatted_lines)
        else:
            return str(data)

    def _get_template_listing(self) -> str:
        """Get formatted listing of available script templates."""
        return """🐍 **Available Script Templates**

**Statistical Analysis**:
• `/script descriptive` - Descriptive statistics (mean, median, std, etc.)
• `/script correlation` - Correlation analysis between variables
• `/script summary` - Comprehensive data summary

**Machine Learning**:
• `/script train_classifier` - Train classification models
• `/script predict` - Make predictions with trained models

**Usage Examples**:
• `/script descriptive for sales` - Statistics for specific column
• `/script correlation for leads and sales` - Correlation between columns
• `/script descriptive` - All numeric columns
• Natural language: "Generate a script for correlation analysis"

**Features**:
✅ Secure sandboxed execution
✅ JSON output format
✅ Statistical validation
✅ Performance monitoring

📁 **Note**: Upload your data file first before generating scripts.
"""


# Convenience functions for direct handler registration
async def script_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Convenience wrapper for script command handling."""
    # This will be initialized by the main bot with proper dependencies
    handler = context.bot_data.get('script_handler')
    if handler:
        await handler.script_command_handler(update, context)
    else:
        await update.message.reply_text("❌ Script handler not initialized.")


async def script_generation_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Convenience wrapper for script generation handling."""
    # This will be initialized by the main bot with proper dependencies
    handler = context.bot_data.get('script_handler')
    if handler:
        await handler.script_generation_handler(update, context)
    else:
        await update.message.reply_text("❌ Script handler not initialized.")