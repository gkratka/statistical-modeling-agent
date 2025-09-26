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
from pathlib import Path
from typing import NoReturn

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

from src.bot.handlers import (
    start_handler,
    help_handler,
    message_handler,
    document_handler,
    error_handler
)
from src.utils.exceptions import ConfigurationError
from src.utils.logger import setup_logger, get_logger


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

    def _setup_handlers(self) -> None:
        """Set up message handlers for the bot."""
        if not self.application:
            raise ConfigurationError("Application not initialized")

        # Command handlers
        self.application.add_handler(CommandHandler("start", start_handler))
        self.application.add_handler(CommandHandler("help", help_handler))

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
    """Main entry point for the bot application."""
    bot = StatisticalModelingBot()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)