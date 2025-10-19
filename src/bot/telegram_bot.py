#!/usr/bin/env python3
"""
Main Telegram bot application for the Statistical Modeling Agent.

This module implements the core bot application following the architecture
specified in CLAUDE.md. It handles bot initialization, configuration,
and graceful shutdown.
"""

import asyncio
import os
import signal
import sys
import time
import yaml
from pathlib import Path
from typing import NoReturn, Dict, Any, Optional

from dotenv import load_dotenv
from telegram import error as telegram_error
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bot import handlers
from src.bot.ml_handlers.ml_training_local_path import register_local_path_handlers
from src.bot.ml_handlers.prediction_handlers import register_prediction_handlers
from src.bot.ml_handlers.prediction_template_handlers import PredictionTemplateHandlers

# Import handler functions from handlers.py file
start_handler = handlers.start_handler
help_handler = handlers.help_handler
message_handler = handlers.message_handler
document_handler = handlers.document_handler
diagnostic_handler = handlers.diagnostic_handler
version_handler = handlers.version_handler
cancel_handler = handlers.cancel_handler
train_handler = handlers.train_handler
error_handler = handlers.error_handler
from src.processors.data_loader import DataLoader
from src.utils.exceptions import ConfigurationError
from src.utils.logger import setup_logger, get_logger

# PID file management
PID_FILE = Path(".bot.pid")


def cleanup_pid_file() -> None:
    """Remove PID file on shutdown."""
    if PID_FILE.exists():
        PID_FILE.unlink()
        logger = get_logger(__name__)
        logger.info("PID file removed")


def check_running_instance() -> bool:
    """Check if another instance is running."""
    if not PID_FILE.exists():
        return False

    logger = get_logger(__name__)
    try:
        existing_pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        return False

    # Check if process is actually running
    try:
        os.kill(existing_pid, 0)  # Signal 0 checks existence
        return True  # Process exists
    except OSError:
        # Process doesn't exist (stale PID)
        logger.warning(f"Stale PID file detected (PID {existing_pid} dead)")
        PID_FILE.unlink()
        return False


def create_pid_file() -> None:
    """Create PID file with current process ID."""
    PID_FILE.write_text(str(os.getpid()))
    logger = get_logger(__name__)
    logger.info(f"PID file created: {os.getpid()}")


class StatisticalModelingBot:
    """
    Main bot application class.

    Handles bot lifecycle including initialization, configuration,
    and graceful shutdown following async/await patterns.
    """

    def __init__(self) -> None:
        """Initialize the bot application."""
        self.application: Optional[Application] = None
        self.logger = get_logger(__name__)
        self._shutdown_requested = False

    def _load_configuration(self) -> Dict[str, str]:
        """
        Load and validate configuration from environment.

        Returns:
            Dictionary containing validated configuration

        Raises:
            ConfigurationError: If required configuration is missing
        """
        # Load .env file
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)

        config = {}

        # Required configuration
        required_vars = ["TELEGRAM_BOT_TOKEN"]

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ConfigurationError(
                    f"Required environment variable {var} is not set. "
                    f"Please check your .env file.",
                    config_key=var
                )
            config[var.lower()] = value

        # Optional configuration with defaults
        config["log_level"] = os.getenv("LOG_LEVEL", "INFO")
        config["log_file"] = os.getenv("LOG_FILE", "./data/logs/bot.log")

        return config

    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Load configuration from config.yaml.

        Returns:
            Dictionary containing YAML configuration

        Raises:
            ConfigurationError: If config file not found or invalid
        """
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        try:
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return {}

            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config or {}

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {str(e)}",
                config_key="config.yaml"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load config.yaml: {e}")
            return {}

    def _setup_handlers(self) -> None:
        """Set up message handlers for the bot."""
        if not self.application:
            raise ConfigurationError("Application not initialized")

        # Load YAML configuration for local path feature
        yaml_config = self._load_yaml_config()

        # Initialize core components
        from src.bot.script_handler import ScriptHandler
        from src.core.parser import RequestParser
        from src.core.orchestrator import TaskOrchestrator
        from src.core.state_manager import StateManager

        parser = RequestParser()
        orchestrator = TaskOrchestrator()
        script_handler = ScriptHandler(parser, orchestrator)
        state_manager = StateManager()

        # Create DataLoader with YAML config for local path support
        data_loader = DataLoader(config=yaml_config)

        # Store in bot_data for access by handlers
        self.application.bot_data['script_handler'] = script_handler
        self.application.bot_data['state_manager'] = state_manager
        self.application.bot_data['data_loader'] = data_loader
        self.application.bot_data['config'] = yaml_config

        # Command handlers
        self.application.add_handler(CommandHandler("start", start_handler))
        self.application.add_handler(CommandHandler("help", help_handler))
        self.application.add_handler(CommandHandler("version", version_handler))
        self.application.add_handler(CommandHandler("diagnostic", diagnostic_handler))
        self.application.add_handler(CommandHandler("cancel", cancel_handler))

        # Register local path training handlers (NEW)
        # This replaces the old train_handler with the enhanced version
        register_local_path_handlers(self.application, state_manager, data_loader)

        # Register prediction handlers (NEW)
        register_prediction_handlers(self.application, state_manager, data_loader)

        # Register prediction template handlers (NEW)
        from src.core.prediction_template import PredictionTemplateConfig
        from src.core.prediction_template_manager import PredictionTemplateManager
        from src.utils.path_validator import PathValidator
        from src.engines.ml_engine import MLEngine
        from src.engines.ml_config import MLEngineConfig
        from telegram.ext import CallbackQueryHandler

        # Initialize template system components
        template_config = PredictionTemplateConfig()
        template_manager = PredictionTemplateManager(template_config)
        path_validator = PathValidator(
            allowed_directories=data_loader.allowed_directories,
            max_size_mb=data_loader.local_max_size_mb,
            allowed_extensions=data_loader.local_extensions
        )
        ml_engine = MLEngine(MLEngineConfig.get_default())

        # Create template handler instance
        template_handlers = PredictionTemplateHandlers(
            state_manager=state_manager,
            template_manager=template_manager,
            data_loader=data_loader,
            path_validator=path_validator,
            ml_engine=ml_engine
        )

        # Register template-specific callback handlers
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_template_source_selection,
                pattern=r"^use_pred_template$"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_template_selection,
                pattern=r"^load_pred_template:"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_template_confirmation,
                pattern=r"^confirm_pred_template$"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_back_to_templates,
                pattern=r"^back_to_pred_templates$"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_template_save_request,
                pattern=r"^save_pred_template$"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_cancel_template,
                pattern=r"^cancel_pred_template$"
            )
        )

        # Register template name text input handler (group 3 to avoid collisions)
        from src.core.state_manager import MLPredictionState

        async def template_name_text_wrapper(update, context):
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await state_manager.get_session(user_id, f"chat_{chat_id}")

            if session and session.current_state == MLPredictionState.SAVING_PRED_TEMPLATE.value:
                await template_handlers.handle_template_name_input(update, context)

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, template_name_text_wrapper),
            group=3  # Separate group for template text input
        )

        # Store template manager in bot_data for access by other handlers
        self.application.bot_data['prediction_template_manager'] = template_manager
        self.application.bot_data['prediction_template_handlers'] = template_handlers

        # Register score workflow handler (NEW)
        from src.bot.score_workflow import ScoreWorkflowHandler
        from telegram.ext import CallbackQueryHandler
        score_handler = ScoreWorkflowHandler(state_manager)
        self.application.add_handler(CommandHandler("score", score_handler.handle_score_command))

        # Create callback wrapper for score workflow
        async def score_callback_wrapper(update, context):
            user_id = update.effective_user.id
            conversation_id = f"chat_{update.effective_chat.id}"
            session = await state_manager.get_or_create_session(user_id, conversation_id)

            # Determine if confirmed or cancelled based on callback_data
            confirmed = update.callback_query.data == "score_confirm"
            await score_handler.handle_confirmation(update, context, session, confirmed)

        # Add callback query handlers for score workflow buttons
        self.application.add_handler(
            CallbackQueryHandler(score_callback_wrapper, pattern="^score_(confirm|cancel)$")
        )
        self.application.bot_data['score_handler'] = score_handler

        # Script command handler
        from src.bot.script_handler import script_command_handler
        self.application.add_handler(CommandHandler("script", script_command_handler))

        # Document handler (for CSV files, etc.)
        self.application.add_handler(
            MessageHandler(filters.Document.ALL, document_handler)
        )

        # Text message handler (must be last to catch all other messages)
        # Use group=1 to ensure CommandHandlers (group=0) have priority
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler),
            group=1
        )

        # Error handler
        self.application.add_error_handler(error_handler)

        self.logger.info("Bot handlers configured successfully")
        if data_loader.local_enabled:
            self.logger.info(f"Local file path training: ENABLED ({len(data_loader.allowed_directories)} allowed directories)")
        else:
            self.logger.info("Local file path training: DISABLED")

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown on SIGINT and SIGTERM."""
        def signal_handler(signum: int, frame) -> None:
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> NoReturn:
        """
        Start the bot application.

        This method runs indefinitely until a shutdown signal is received.
        """
        try:
            # Load configuration
            config = self._load_configuration()

            # Setup logging
            setup_logger(
                level=config["log_level"],
                log_file=config.get("log_file")
            )

            self.logger.info("Starting Statistical Modeling Agent bot...")
            self.logger.info(f"Log level: {config['log_level']}")

            # Create application
            self.application = Application.builder().token(
                config["telegram_bot_token"]
            ).build()

            # Setup handlers
            self._setup_handlers()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Initialize and start the bot
            await self.application.initialize()
            await self.application.start()

            # Start polling with exponential backoff retry logic
            self.logger.info("Bot started successfully. Polling for messages...")

            max_retries = 10
            retry_count = 0
            polling_started = False

            while not polling_started and retry_count < max_retries:
                try:
                    await self.application.updater.start_polling(
                        poll_interval=1.0,
                        timeout=30,
                        drop_pending_updates=True
                    )
                    polling_started = True
                    self.logger.info("✓ Polling started successfully")

                except telegram_error.Conflict as e:
                    retry_count += 1
                    backoff_time = min(2 ** retry_count, 30)  # Exponential backoff capped at 30s
                    self.logger.warning(
                        f"⚠️ Telegram API Conflict detected (attempt {retry_count}/{max_retries}). "
                        f"Another bot instance may be running. Retrying in {backoff_time}s..."
                    )
                    await asyncio.sleep(backoff_time)

                except telegram_error.NetworkError as e:
                    retry_count += 1
                    backoff_time = min(2 ** retry_count, 30)
                    self.logger.warning(
                        f"⚠️ Network error (attempt {retry_count}/{max_retries}): {e}. "
                        f"Retrying in {backoff_time}s..."
                    )
                    await asyncio.sleep(backoff_time)

                except Exception as e:
                    self.logger.error(f"❌ Unexpected error starting polling: {e}", exc_info=True)
                    raise

            if not polling_started:
                raise RuntimeError(
                    f"Failed to start polling after {max_retries} attempts. "
                    "Please ensure no other bot instances are running."
                )

            # Wait for shutdown signal with periodic health logging
            self.logger.info("Bot is now running and processing messages...")
            while not self._shutdown_requested:
                await asyncio.sleep(1)

        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e.message}")
            sys.exit(1)

        except Exception as e:
            self.logger.error(f"Unexpected error during startup: {e}")
            sys.exit(1)

        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Perform graceful shutdown of the bot."""
        self.logger.info("Shutting down bot...")

        if self.application:
            try:
                # Stop updater
                if self.application.updater.running:
                    await self.application.updater.stop()
                    self.logger.info("Updater stopped")

                # Stop application
                await self.application.stop()
                self.logger.info("Application stopped")

                # Shutdown application
                await self.application.shutdown()
                self.logger.info("Application shutdown complete")

            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")

        self.logger.info("Statistical Modeling Agent bot stopped")

    def run(self) -> None:
        """
        Synchronous run method for compatibility.

        This method provides a simple interface for running the bot
        from synchronous code.
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        except Exception as e:
            print(f"Fatal error: {e}")


async def main() -> None:
    """Main entry point for the bot application with instance management."""
    logger = get_logger(__name__)

    # INSTANCE CHECK
    if check_running_instance():
        logger.error("❌ Bot instance already running!")
        logger.error("   Use ./scripts/start_bot_clean.sh to restart")
        sys.exit(1)

    # CREATE PID FILE
    create_pid_file()

    # REGISTER CLEANUP
    def cleanup_handler(signum: int, frame) -> None:
        cleanup_pid_file()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

    try:
        bot = StatisticalModelingBot()
        await bot.start()
    finally:
        # ALWAYS CLEANUP
        cleanup_pid_file()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        cleanup_pid_file()
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        cleanup_pid_file()
        sys.exit(1)