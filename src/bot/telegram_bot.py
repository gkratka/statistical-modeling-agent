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
import yaml
from pathlib import Path
from typing import NoReturn, Dict, Any

from dotenv import load_dotenv
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
        self.application: Application | None = None
        self.logger = get_logger(__name__)
        self._shutdown_requested = False

    def _load_configuration(self) -> dict[str, str]:
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

        # Script command handler
        from src.bot.script_handler import script_command_handler
        self.application.add_handler(CommandHandler("script", script_command_handler))

        # Document handler (for CSV files, etc.)
        self.application.add_handler(
            MessageHandler(filters.Document.ALL, document_handler)
        )

        # Text message handler (must be last to catch all other messages)
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
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

            # Start polling
            self.logger.info("Bot started successfully. Polling for messages...")
            await self.application.updater.start_polling(
                poll_interval=1.0,
                timeout=30,
                drop_pending_updates=True
            )

            # Wait for shutdown signal
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