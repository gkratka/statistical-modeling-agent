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
from telegram.request import HTTPXRequest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local Worker imports (must be after sys.path.insert)
from src.worker.websocket_server import WebSocketServer
from src.worker.http_server import HTTPServer
from src.worker.job_queue import JobQueue

from src.bot import main_handlers as handlers
from src.bot.ml_handlers.ml_training_local_path import register_local_path_handlers
from src.bot.ml_handlers.prediction_handlers import register_prediction_handlers
from src.bot.ml_handlers.prediction_template_handlers import PredictionTemplateHandlers
from src.bot.ml_handlers.models_browser_handler import ModelsBrowserHandler

# Import handler functions from main_handlers.py file
start_handler = handlers.start_handler
help_handler = handlers.help_handler
pt_handler = handlers.pt_handler
en_handler = handlers.en_handler
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

        # Local Worker servers
        self._websocket_server: Optional[WebSocketServer] = None
        self._http_server: Optional[HTTPServer] = None
        self._worker_enabled = False

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

        # Initialize i18n for translations
        from src.utils.i18n_manager import I18nManager
        I18nManager.initialize(
            locales_dir="./locales",
            default_locale=yaml_config.get("i18n", {}).get("default_language", "en")
        )

        # Store in bot_data for access by handlers
        self.application.bot_data['script_handler'] = script_handler
        self.application.bot_data['state_manager'] = state_manager
        self.application.bot_data['data_loader'] = data_loader
        self.application.bot_data['config'] = yaml_config

        # Check XGBoost availability and log status
        from src.utils.xgboost_checker import log_xgboost_status
        log_xgboost_status()

        # Command handlers
        self.application.add_handler(CommandHandler("start", start_handler))
        self.application.add_handler(CommandHandler("help", help_handler))
        self.application.add_handler(CommandHandler("pt", pt_handler))
        self.application.add_handler(CommandHandler("en", en_handler))
        self.application.add_handler(CommandHandler("version", version_handler))
        self.application.add_handler(CommandHandler("diagnostic", diagnostic_handler))
        self.application.add_handler(CommandHandler("cancel", cancel_handler))

        # Local worker command handlers
        from src.bot.handlers.connect_handler import (
            handle_connect_command,
            handle_disconnect_command,
            handle_worker_connect_button,
            handle_worker_autostart_command
        )
        from telegram.ext import CallbackQueryHandler
        self.application.add_handler(CommandHandler("connect", handle_connect_command))
        self.application.add_handler(CommandHandler("disconnect", handle_disconnect_command))
        self.application.add_handler(CommandHandler("worker", handle_worker_autostart_command))
        self.application.add_handler(
            CallbackQueryHandler(handle_worker_connect_button, pattern="^worker_connect$")
        )

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
                template_handlers.handle_upload_pred_template_request,
                pattern=r"^upload_pred_template$"
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
                template_handlers.handle_back_to_pred_template_list,
                pattern=r"^back_to_pred_templates$"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_template_delete,
                pattern=r"^delete_pred_template:"
            )
        )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_overwrite_confirmation,
                pattern=r"^confirm_overwrite_pred_template$"
            )
        )
        # COMMENTED OUT: Replaced by unified template system
        # self.application.add_handler(
        #     CallbackQueryHandler(
        #         template_handlers.handle_template_save_request,
        #         pattern=r"^save_pred_template$"
        #     )
        # )
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_cancel_template,
                pattern=r"^cancel_pred_template$"
            )
        )
        # Prediction template management (delete)
        self.application.add_handler(
            CallbackQueryHandler(
                template_handlers.handle_manage_pred_templates,
                pattern=r"^manage_pred_templates$"
            )
        )

        # COMMENTED OUT: Replaced by unified template system (now in group 6)
        # Register template name text input handler (group 3 to avoid collisions)
        # from src.core.state_manager import MLPredictionState

        # async def template_name_text_wrapper(update, context):
        #     user_id = update.effective_user.id
        #     chat_id = update.effective_chat.id
        #     session = await state_manager.get_session(user_id, f"chat_{chat_id}")

        #     if session and session.current_state == MLPredictionState.SAVING_PRED_TEMPLATE.value:
        #         await template_handlers.handle_template_name_input(update, context)

        # self.application.add_handler(
        #     MessageHandler(filters.TEXT & ~filters.COMMAND, template_name_text_wrapper),
        #     group=3  # Separate group for template text input
        # )

        # Document handler for prediction template uploads (group 4 to avoid conflicts)
        from src.core.state_manager import MLPredictionState

        async def pred_template_document_handler(update, context):
            """Handle document uploads for prediction template upload workflow."""
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await state_manager.get_session(user_id, f"chat_{chat_id}")

            if session and session.current_state == MLPredictionState.AWAITING_PRED_TEMPLATE_UPLOAD.value:
                await template_handlers.handle_pred_template_upload(update, context)

        self.application.add_handler(
            MessageHandler(filters.Document.ALL, pred_template_document_handler),
            group=4  # Separate group to avoid conflicts
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

        # Register models browser handler (NEW - /models command)
        models_handler = ModelsBrowserHandler(state_manager)
        self.application.add_handler(CommandHandler("models", models_handler.handle_models_command))

        # Register unified template handler (NEW - /template command)
        from src.bot.handlers.template_handlers import TemplateHandlers
        template_handler = TemplateHandlers(state_manager)
        self.application.add_handler(CommandHandler("template", template_handler.handle_template_command))

        # Add callback query handler for "Save as Template" button clicks
        # Support both train and predict template save buttons
        self.application.add_handler(
            CallbackQueryHandler(template_handler.handle_save_template_button, pattern="^save_as_template$")
        )
        # Route prediction template save to PredictionTemplateHandlers (stored in bot_data)
        async def save_pred_template_wrapper(update, context):
            prediction_handlers = context.bot_data.get('prediction_template_handlers')
            if prediction_handlers:
                await prediction_handlers.handle_template_save_request(update, context)

        self.application.add_handler(
            CallbackQueryHandler(save_pred_template_wrapper, pattern="^save_pred_template$")
        )

        # Add callback query handler for template save cancellation
        self.application.add_handler(
            CallbackQueryHandler(
                lambda update, context: update.callback_query.answer() or update.callback_query.edit_message_text("Template save cancelled."),
                pattern="^cancel_template$"
            )
        )

        # Template name text input handler (group 6 to avoid collisions)
        from src.core.state_manager import MLTrainingState, MLPredictionState

        async def template_name_text_wrapper(update, context):
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await state_manager.get_session(user_id, f"chat_{chat_id}")

            if session:
                if session.current_state == MLTrainingState.SAVING_TEMPLATE.value:
                    # Training templates use unified handler
                    await template_handler.handle_template_name_input(update, context)
                elif session.current_state == MLPredictionState.SAVING_PRED_TEMPLATE.value:
                    # Prediction templates use PredictionTemplateHandlers
                    prediction_handlers = context.bot_data.get('prediction_template_handlers')
                    if prediction_handlers:
                        await prediction_handlers.handle_template_name_input(update, context)

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, template_name_text_wrapper),
            group=6  # Separate group for template text input
        )

        self.application.bot_data['template_handler'] = template_handler

        # Add callback query handlers for models browser navigation
        self.application.add_handler(
            CallbackQueryHandler(models_handler.handle_model_selection, pattern="^model:")
        )
        self.application.add_handler(
            CallbackQueryHandler(models_handler.handle_pagination, pattern="^page:")
        )
        self.application.add_handler(
            CallbackQueryHandler(models_handler.handle_back_to_list, pattern="^back_to_list:")
        )
        self.application.add_handler(
            CallbackQueryHandler(models_handler.handle_cancel, pattern="^cancel_models$")
        )
        self.application.bot_data['models_handler'] = models_handler

        # Register join workflow handlers (NEW - /join command)
        from src.bot.handlers.join_handlers import JoinHandler
        from src.core.state_manager import JoinWorkflowState

        join_handler = JoinHandler(
            state_manager=state_manager,
            data_loader=data_loader,
            path_validator=path_validator,
            websocket_server=self._websocket_server
        )

        # Command handler for /join
        self.application.add_handler(CommandHandler("join", join_handler.handle_start_join))

        # Callback query handlers for join workflow buttons
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_operation_selection, pattern=r"^join_op_")
        )
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_dataframe_count, pattern=r"^join_count_")
        )
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_dataframe_source, pattern=r"^join_df_source_")
        )
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_key_column_selection, pattern=r"^join_key_")
        )
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_filter_selection, pattern=r"^join_filter_")
        )
        self.application.add_handler(
            CallbackQueryHandler(join_handler.handle_output_path, pattern=r"^join_output_")
        )

        # Text input handlers for join workflow (file path and custom output path)
        async def join_text_input_wrapper(update, context):
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await state_manager.get_session(user_id, f"chat_{chat_id}")

            if not session or session.workflow_type is None:
                return

            from src.core.state_manager import WorkflowType
            if session.workflow_type != WorkflowType.JOIN_WORKFLOW:
                return

            current_state = session.current_state
            # Handle file path input states
            if "DF" in current_state.upper() and "PATH" in current_state.upper():
                await join_handler.handle_file_path_input(update, context)
            # Handle custom output path input
            elif current_state == JoinWorkflowState.AWAITING_CUSTOM_OUTPUT_PATH.value:
                await join_handler.handle_custom_output_path(update, context)
            # Handle filter input (user can type filter expressions in CHOOSING_FILTER state)
            elif current_state == JoinWorkflowState.CHOOSING_FILTER.value:
                await join_handler.handle_filter_input(update, context)

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, join_text_input_wrapper),
            group=4  # Separate group for join text input
        )

        # File upload handler for join workflow
        async def join_upload_wrapper(update, context):
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
            session = await state_manager.get_session(user_id, f"chat_{chat_id}")

            if not session or session.workflow_type is None:
                return

            from src.core.state_manager import WorkflowType
            if session.workflow_type != WorkflowType.JOIN_WORKFLOW:
                return

            current_state = session.current_state
            # Handle file upload states
            if "DF" in current_state.upper() and "UPLOAD" in current_state.upper():
                await join_handler.handle_file_upload(update, context)

        self.application.add_handler(
            MessageHandler(filters.Document.ALL, join_upload_wrapper),
            group=5  # Separate group for join file uploads
        )

        self.application.bot_data['join_handler'] = join_handler
        self.logger.info("Join workflow handlers registered")

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

    async def _setup_worker_servers(self, yaml_config: Dict[str, Any]) -> None:
        """Set up WebSocket and HTTP servers for local worker connections.

        Args:
            yaml_config: YAML configuration dict
        """
        worker_config = yaml_config.get("worker", {})
        self._worker_enabled = worker_config.get("enabled", False)

        if not self._worker_enabled:
            self.logger.info("Local Worker feature: DISABLED")
            return

        # Get server configuration
        ws_host = worker_config.get("websocket_host", "0.0.0.0")
        ws_port = worker_config.get("websocket_port", 8765)
        http_host = worker_config.get("http_host", "0.0.0.0")
        http_port = worker_config.get("http_port", 8080)
        worker_script_path = Path(worker_config.get("script_path", "worker/statsbot_worker.py"))

        # Create WebSocket server
        self._websocket_server = WebSocketServer(
            host=ws_host,
            port=ws_port,
        )

        # Create job queue and attach to websocket server
        job_queue = JobQueue(default_timeout=3600.0)  # 1 hour for large model training
        job_queue.set_worker_manager(self._websocket_server.worker_manager)
        self._websocket_server.job_queue = job_queue

        # Register message handlers for job progress/results
        self._websocket_server.register_message_handler(
            "progress",
            lambda user_id, msg: asyncio.create_task(job_queue.handle_progress(user_id, msg))
        )
        self._websocket_server.register_message_handler(
            "result",
            lambda user_id, msg: asyncio.create_task(job_queue.handle_result(user_id, msg))
        )

        # Create HTTP server for worker script
        self._http_server = HTTPServer(
            host=http_host,
            port=http_port,
            worker_script_path=worker_script_path,
        )

        # Setup worker connection callbacks
        from src.bot.handlers.connect_handler import notify_worker_connected, notify_worker_disconnected

        async def on_worker_connect(user_id: int, machine_name: str):
            """Callback when worker connects."""
            if self.application:
                await notify_worker_connected(self.application.bot, user_id, machine_name)

        async def on_worker_disconnect(user_id: int):
            """Callback when worker disconnects."""
            if self.application:
                await notify_worker_disconnected(self.application.bot, user_id)

        self._websocket_server.worker_manager.on_connect(on_worker_connect)
        self._websocket_server.worker_manager.on_disconnect(on_worker_disconnect)

        # Start servers
        await self._websocket_server.start()
        await self._http_server.start()

        # Store in bot_data for handler access
        if self.application:
            self.application.bot_data['websocket_server'] = self._websocket_server
            self.application.bot_data['job_queue'] = job_queue
            self.application.bot_data['worker_enabled'] = True
            # Store worker HTTP URL for /connect command
            # Use env var for production (Railway), fallback to localhost for dev
            display_host = "localhost" if http_host == "0.0.0.0" else http_host
            default_http_url = f"http://{display_host}:{http_port}"
            self.application.bot_data['worker_http_url'] = os.environ.get('WORKER_HTTP_URL', default_http_url)
            # Store worker WebSocket URL for /connect command
            default_ws_url = f"ws://{display_host}:{ws_port}/ws"
            self.application.bot_data['worker_ws_url'] = os.environ.get('WORKER_WS_URL', default_ws_url)

        self.logger.info(f"Local Worker feature: ENABLED")
        self.logger.info(f"  WebSocket server: ws://{ws_host}:{ws_port}")
        self.logger.info(f"  HTTP server: http://{http_host}:{http_port}/worker")

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

            # Create application with increased timeouts for long-running predictions
            request = HTTPXRequest(
                connect_timeout=30.0,
                read_timeout=60.0,
                write_timeout=60.0,
            )
            self.application = Application.builder().token(
                config["telegram_bot_token"]
            ).request(request).build()

            # Setup handlers
            self._setup_handlers()

            # Setup Local Worker servers (WebSocket + HTTP)
            yaml_config = self._load_yaml_config()
            await self._setup_worker_servers(yaml_config)

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

        # Stop Local Worker servers
        if self._websocket_server:
            try:
                await self._websocket_server.stop()
                self.logger.info("WebSocket server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket server: {e}")

        if self._http_server:
            try:
                await self._http_server.stop()
                self.logger.info("HTTP server stopped")
            except Exception as e:
                self.logger.error(f"Error stopping HTTP server: {e}")

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