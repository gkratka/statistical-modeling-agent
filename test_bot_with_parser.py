#!/usr/bin/env python3
"""
Test bot with parser integration for real Telegram message testing.

This bot integrates the parser to show how real user messages are parsed
and what TaskDefinition objects are created. Perfect for testing and validation.
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import NoReturn, Optional
import json

from dotenv import load_dotenv
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram import Update

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.parser import RequestParser, DataSource, TaskDefinition
from src.utils.exceptions import ParseError, ConfigurationError
from src.utils.logger import setup_logger, get_logger

# Global storage for user data (in production this would be a database)
user_data_store = {}

class TestParserBot:
    """Test bot that demonstrates parser integration."""

    def __init__(self) -> None:
        """Initialize the test bot."""
        self.application: Application | None = None
        self.logger = get_logger(__name__)
        self.parser = RequestParser()
        self._shutdown_requested = False

    def _load_configuration(self) -> dict[str, str]:
        """Load configuration from environment."""
        env_path = Path(__file__).parent / ".env"
        load_dotenv(env_path)

        config = {}
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ConfigurationError(
                "TELEGRAM_BOT_TOKEN not found in .env file",
                config_key="TELEGRAM_BOT_TOKEN"
            )

        config["telegram_bot_token"] = token
        config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
        return config

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        self.logger.info(f"Parser test bot started by: {username} (ID: {user_id})")

        welcome_message = (
            "🤖 **Parser Testing Bot**\n\n"
            "This bot tests the natural language parser in real-time!\n\n"
            "**Try these commands:**\n"
            "📊 `calculate mean for age column`\n"
            "📈 `show correlation between age and income`\n"
            "🧠 `train a model to predict salary`\n"
            "📋 `what columns are available`\n\n"
            "**Features:**\n"
            "✅ Shows parsed task type and operation\n"
            "📊 Displays confidence scores\n"
            "🔍 Shows extracted parameters\n"
            "⚠️ Handles parsing errors gracefully\n\n"
            "**Upload a CSV file first**, then try analysis requests!"
        )

        try:
            await update.message.reply_text(welcome_message, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send welcome message: {e}")

    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not update.effective_user or not update.message:
            return

        help_message = (
            "🆘 **Parser Test Bot Help**\n\n"
            "**Statistical Requests:**\n"
            "• `calculate mean`, `find average`\n"
            "• `show correlation matrix`\n"
            "• `descriptive statistics`\n"
            "• `standard deviation for salary`\n\n"
            "**Machine Learning Requests:**\n"
            "• `train a model to predict income`\n"
            "• `build random forest classifier`\n"
            "• `predict house prices`\n"
            "• `neural network for classification`\n\n"
            "**Data Information Requests:**\n"
            "• `show me the data`\n"
            "• `what columns are available`\n"
            "• `data shape and info`\n\n"
            "**Debug Features:**\n"
            "Type `/debug` to see detailed parsing information\n"
            "Type `/data` to see uploaded file information\n\n"
            "The bot will show you exactly how each message is parsed!"
        )

        try:
            await update.message.reply_text(help_message, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send help message: {e}")

    async def debug_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show debug information about parser patterns."""
        if not update.effective_user or not update.message:
            return

        debug_info = (
            "🔧 **Parser Debug Information**\n\n"
            "**Statistics Patterns:**\n"
            f"• {len(self.parser.stats_patterns)} patterns loaded\n"
            f"• Examples: {list(self.parser.stats_patterns.keys())[:5]}\n\n"
            "**ML Patterns:**\n"
            f"• {len(self.parser.ml_patterns)} patterns loaded\n"
            f"• Examples: {list(self.parser.ml_patterns.keys())[:5]}\n\n"
            "**Column Patterns:**\n"
            f"• {len(self.parser.column_patterns)} patterns loaded\n\n"
            "**Confidence Thresholds:**\n"
            "• ML patterns: 0.5+ required\n"
            "• Stats patterns: 0.3+ required\n"
            "• Below 0.3: ParseError raised\n\n"
            "Send any message to see detailed parsing!"
        )

        try:
            await update.message.reply_text(debug_info, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send debug info: {e}")

    async def data_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show information about uploaded data."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        data_source = user_data_store.get(user_id)

        if not data_source:
            message = (
                "📁 **No Data Uploaded**\n\n"
                "Upload a CSV file to test parsing with data context.\n"
                "The parser can extract column names and use file information."
            )
        else:
            message = (
                f"📁 **Current Data Source**\n\n"
                f"**File:** {data_source.file_name}\n"
                f"**Type:** {data_source.file_type}\n"
                f"**Size:** {data_source.shape}\n"
                f"**Columns:** {data_source.columns}\n\n"
                "This data context is used when parsing your requests."
            )

        try:
            await update.message.reply_text(message, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send data info: {e}")

    async def document_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle CSV file uploads and create DataSource objects."""
        if not update.effective_user or not update.message or not update.message.document:
            return

        user_id = update.effective_user.id
        document = update.message.document
        file_name = document.file_name or "unknown"
        file_size = document.file_size or 0

        self.logger.info(f"File uploaded by user {user_id}: {file_name}")

        # Create a DataSource object (in real implementation, would process the file)
        try:
            # Simulate CSV processing
            if file_name.lower().endswith('.csv'):
                # Create mock DataSource with realistic data
                data_source = DataSource(
                    file_id=document.file_id,
                    file_name=file_name,
                    file_type="csv",
                    columns=["age", "income", "education", "satisfaction", "region"],  # Mock columns
                    shape=(1000, 5)  # Mock shape
                )

                # Store for user
                user_data_store[user_id] = data_source

                response = (
                    f"📁 **File Processed Successfully!**\n\n"
                    f"**File:** {file_name}\n"
                    f"**Size:** {file_size:,} bytes\n"
                    f"**Type:** CSV detected\n"
                    f"**Columns found:** {', '.join(data_source.columns)}\n"
                    f"**Rows:** {data_source.shape[0]:,}\n\n"
                    f"✅ **Ready for analysis!**\n"
                    f"Now try requests like:\n"
                    f"• `calculate mean for age`\n"
                    f"• `correlation between income and satisfaction`\n"
                    f"• `train model to predict satisfaction based on age and income`\n\n"
                    f"The parser will use this data context for better accuracy."
                )
            else:
                response = (
                    f"⚠️ **File Type Not Supported**\n\n"
                    f"Please upload a CSV file for testing.\n"
                    f"Received: {file_name} ({document.mime_type})"
                )

        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            response = f"❌ **Error processing file:** {str(e)}"

        try:
            await update.message.reply_text(response, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send file response: {e}")

    async def message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages using the parser."""
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        message_text = update.message.text or ""

        self.logger.info(f"Parsing message from {username}: {message_text[:100]}")

        # Get user's data source if available
        data_source = user_data_store.get(user_id)

        try:
            # Parse the message
            conversation_id = f"{update.effective_chat.id}_{update.message.message_id}"
            task = self.parser.parse_request(
                message_text,
                user_id,
                conversation_id,
                data_source
            )

            # Format the response with detailed parsing information
            response = self._format_parsing_success(task, message_text, data_source)

        except ParseError as e:
            response = self._format_parsing_error(e, message_text)
            self.logger.warning(f"Parse error for user {user_id}: {e.message}")

        except Exception as e:
            response = self._format_unexpected_error(e, message_text)
            self.logger.error(f"Unexpected error parsing message: {e}")

        try:
            await update.message.reply_text(response, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")

    def _format_parsing_success(self, task: TaskDefinition, original_text: str, data_source: Optional[DataSource]) -> str:
        """Format a successful parsing result."""
        # Confidence level emoji
        confidence_emoji = "🟢" if task.confidence_score >= 0.7 else "🟡" if task.confidence_score >= 0.4 else "🔴"

        # Task type emoji
        type_emoji = {
            "stats": "📊",
            "ml_train": "🧠",
            "ml_score": "🔮",
            "data_info": "📋"
        }.get(task.task_type, "❓")

        response = f"""✅ **Parsing Successful!** {confidence_emoji}

**Original:** "{original_text}"

{type_emoji} **Task Type:** {task.task_type}
🎯 **Operation:** {task.operation.replace('_', ' ').title()}
📈 **Confidence:** {task.confidence_score:.0%}

**📋 Parameters:**"""

        # Format parameters nicely
        for key, value in task.parameters.items():
            if value and value != ["all"]:
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                response += f"\n• **{key.title()}:** {value_str}"

        # Add data context information
        if data_source:
            response += f"\n\n📁 **Data Context:** {data_source.file_name} ({data_source.shape[0]} rows)"
        else:
            response += f"\n\n⚠️ **No data uploaded** - Upload CSV for column-specific parsing"

        # Add what would happen next
        response += f"\n\n🚀 **Next Steps:**\n"
        if task.task_type == "stats":
            response += f"• Route to Statistics Engine\n• Generate analysis script\n• Calculate: {', '.join(task.parameters.get('statistics', []))}"
        elif task.task_type == "ml_train":
            response += f"• Route to ML Training Engine\n• Build {task.parameters.get('model_type', 'auto')} model"
            if task.parameters.get('target'):
                response += f"\n• Target: {task.parameters['target']}"
        elif task.task_type == "ml_score":
            response += f"• Route to ML Scoring Engine\n• Make predictions"
        else:
            response += f"• Route to Data Information Engine\n• Provide data overview"

        return response

    def _format_parsing_error(self, error: ParseError, original_text: str) -> str:
        """Format a parsing error response."""
        return f"""❌ **Could Not Parse Request**

**Original:** "{original_text}"

**Issue:** {error.message}

💡 **Try these instead:**

**📊 Statistics:**
• "calculate mean for age column"
• "show correlation between income and education"
• "descriptive statistics for all columns"

**🧠 Machine Learning:**
• "train a model to predict satisfaction"
• "build random forest classifier"
• "predict income based on age and education"

**📋 Data Info:**
• "show me the data"
• "what columns are available"
• "data shape and size"

**Tips:**
• Be specific about what you want to analyze
• Mention column names if you know them
• Use action words like "calculate", "show", "train", "predict"
"""

    def _format_unexpected_error(self, error: Exception, original_text: str) -> str:
        """Format an unexpected error response."""
        return f"""⚠️ **Unexpected Error**

**Original:** "{original_text}"
**Error:** {str(error)}

This appears to be a bug in the parser. The error has been logged for investigation.

Please try a different request or contact support if this persists.
"""

    def _setup_handlers(self) -> None:
        """Set up all bot handlers."""
        if not self.application:
            raise ConfigurationError("Application not initialized")

        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_handler))
        self.application.add_handler(CommandHandler("help", self.help_handler))
        self.application.add_handler(CommandHandler("debug", self.debug_handler))
        self.application.add_handler(CommandHandler("data", self.data_handler))

        # Document handler for CSV uploads
        self.application.add_handler(
            MessageHandler(filters.Document.ALL, self.document_handler)
        )

        # Text message handler (must be last)
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler)
        )

        # Error handler
        self.application.add_error_handler(self._error_handler)

        self.logger.info("Parser test bot handlers configured")

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle bot errors."""
        error = context.error
        self.logger.error(f"Bot error: {error}", exc_info=error)

        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ An error occurred. Please try again or use /help for guidance."
                )
            except Exception:
                pass

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown."""
        def signal_handler(signum: int, frame) -> None:
            self.logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> NoReturn:
        """Start the test bot."""
        try:
            config = self._load_configuration()

            setup_logger(level=config.get("log_level", "INFO"))
            self.logger.info("Starting Parser Test Bot...")

            # Create application
            self.application = Application.builder().token(
                config["telegram_bot_token"]
            ).build()

            # Setup handlers and signals
            self._setup_handlers()
            self._setup_signal_handlers()

            # Start bot
            await self.application.initialize()
            await self.application.start()

            self.logger.info("Parser Test Bot started! Send messages to test parsing...")
            await self.application.updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True
            )

            # Wait for shutdown
            while not self._shutdown_requested:
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}")
            sys.exit(1)
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Shutdown the bot gracefully."""
        self.logger.info("Shutting down Parser Test Bot...")
        if self.application:
            try:
                if self.application.updater.running:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

    def run(self) -> None:
        """Run the bot synchronously."""
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\nParser Test Bot stopped by user")
        except Exception as e:
            print(f"Fatal error: {e}")


if __name__ == "__main__":
    bot = TestParserBot()
    bot.run()