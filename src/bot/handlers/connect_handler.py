"""Handler for /connect command and worker connection management.

This module implements the /connect command that generates one-time tokens
for local worker authentication and displays platform-specific connection
commands.

Security:
    - One-time tokens with 5-minute expiry
    - Token invalidation after successful use
    - User-specific worker association

Features:
    - Platform-specific commands (Mac/Linux curl, Windows irm)
    - Worker connection/disconnection notifications
    - Start command integration with worker status
"""

import logging
from typing import Optional, Tuple

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.utils.decorators import telegram_handler, log_user_action

logger = logging.getLogger(__name__)


# ============================================================================
# /connect Command Handler
# ============================================================================


@telegram_handler
@log_user_action("/connect command")
async def handle_connect_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /connect command - generate token and show connection instructions.

    This command:
    1. Checks if worker feature is enabled
    2. Generates a one-time authentication token
    3. Displays platform-specific commands to connect worker
    4. Shows token expiry warning (5 minutes)

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id

    # Check if worker feature is enabled
    worker_enabled = context.bot_data.get('worker_enabled', False)
    if not worker_enabled:
        await update.message.reply_text(
            "❌ **Local Worker Not Available**\n\n"
            "The local worker feature is not enabled on this bot.\n\n"
            "Please contact the bot administrator.",
            parse_mode="Markdown"
        )
        return

    # Check if websocket server is available
    websocket_server = context.bot_data.get('websocket_server')
    if not websocket_server:
        await update.message.reply_text(
            "❌ **Worker Service Unavailable**\n\n"
            "The worker connection service is not running.\n\n"
            "Please contact the bot administrator.",
            parse_mode="Markdown"
        )
        return

    # Check if worker already connected
    worker_manager = websocket_server.worker_manager
    if worker_manager.is_user_connected(user_id):
        machine_name = worker_manager.get_machine_name(user_id)
        await update.message.reply_text(
            f"✅ **Worker Already Connected**\n\n"
            f"**Machine:** {machine_name}\n\n"
            f"Your local worker is already connected and ready to execute jobs.\n\n"
            f"To disconnect, stop the worker script on your machine.",
            parse_mode="Markdown"
        )
        return

    # Generate one-time token
    token_manager = websocket_server.token_manager
    token = token_manager.generate_token(user_id)

    # Get server URL (from config or environment)
    # For Railway deployment, this will be the public URL
    # For local development, this will be localhost
    server_url = context.bot_data.get('worker_http_url', 'http://localhost:8080')

    # Build connection commands
    mac_linux_command = (
        f"curl -s {server_url}/worker | python3 - --token={token}"
    )
    windows_command = (
        f"irm {server_url}/worker | python - --token={token}"
    )

    # Send instructions
    message = (
        "🔌 **Connect Local Worker**\n\n"
        "**Step 1:** Choose your platform:\n\n"
        "**Mac/Linux:**\n"
        f"```bash\n{mac_linux_command}\n```\n\n"
        "**Windows (PowerShell):**\n"
        f"```powershell\n{windows_command}\n```\n\n"
        "**Step 2:** Copy the command above and run it in your terminal.\n\n"
        "**Step 3:** Wait for the \"Worker connected!\" message.\n\n"
        "⏱️ **Token expires in 5 minutes**\n\n"
        "Once connected, you can use local file paths in `/train` and `/predict` workflows."
    )

    await update.message.reply_text(message, parse_mode="Markdown")
    logger.info(f"Generated connection token for user {user_id}")


# ============================================================================
# Start Command Integration
# ============================================================================


def get_start_message_with_worker_status(
    user_id: int,
    context: ContextTypes.DEFAULT_TYPE
) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    """Get start message with worker connection status.

    This function enhances the /start message to show:
    - "Connect local worker" button when no worker connected
    - "Worker: Connected (machine-name)" status when connected

    Args:
        user_id: Telegram user ID
        context: Bot context

    Returns:
        Tuple of (message_suffix, inline_keyboard)
        - message_suffix: Text to append to start message
        - inline_keyboard: Optional keyboard with connect button
    """
    worker_enabled = context.bot_data.get('worker_enabled', False)
    if not worker_enabled:
        return "", None

    websocket_server = context.bot_data.get('websocket_server')
    if not websocket_server:
        return "", None

    worker_manager = websocket_server.worker_manager

    # Check if worker connected
    if worker_manager.is_user_connected(user_id):
        machine_name = worker_manager.get_machine_name(user_id)
        status_message = (
            f"\n\n🖥️ **Local Worker Status**\n"
            f"✅ Connected: `{machine_name}`\n\n"
            f"You can use local file paths in `/train` and `/predict`."
        )
        return status_message, None
    else:
        status_message = (
            "\n\n🖥️ **Local Worker Status**\n"
            "❌ Not connected\n\n"
            "Connect a local worker to train models on your own machine."
        )

        # Create inline keyboard with connect button
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔌 Connect Local Worker", callback_data="worker_connect")]
        ])

        return status_message, keyboard


# ============================================================================
# Worker Notifications
# ============================================================================


async def notify_worker_connected(
    bot,
    user_id: int,
    machine_name: str
) -> None:
    """Send notification to user when worker connects.

    Args:
        bot: Telegram bot instance
        user_id: Telegram user ID
        machine_name: Hostname of connected worker
    """
    message = (
        "✅ **Local Worker Connected!**\n\n"
        f"**Machine:** {machine_name}\n\n"
        "Your local worker is now connected and ready to execute ML jobs.\n\n"
        "You can now use local file paths in `/train` and `/predict` workflows."
    )

    try:
        await bot.send_message(
            chat_id=user_id,
            text=message,
            parse_mode="Markdown"
        )
        logger.info(f"Sent connection notification to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send connection notification to user {user_id}: {e}")


async def notify_worker_disconnected(
    bot,
    user_id: int
) -> None:
    """Send notification to user when worker disconnects.

    Args:
        bot: Telegram bot instance
        user_id: Telegram user ID
    """
    message = (
        "⚠️ **Local Worker Disconnected**\n\n"
        "Your local worker has disconnected.\n\n"
        "To reconnect, use `/connect` and run the connection command on your machine."
    )

    try:
        await bot.send_message(
            chat_id=user_id,
            text=message,
            parse_mode="Markdown"
        )
        logger.info(f"Sent disconnection notification to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send disconnection notification to user {user_id}: {e}")


# ============================================================================
# Callback Query Handler
# ============================================================================


@telegram_handler
async def handle_worker_connect_button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle 'Connect Local Worker' button press from /start.

    This is triggered when user presses the inline button in /start message.

    Args:
        update: Telegram update object (with callback_query)
        context: Bot context
    """
    query = update.callback_query
    await query.answer()

    # Create a fake update that looks like /connect command
    # This allows reusing the connect command handler
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Check if worker feature is enabled
    worker_enabled = context.bot_data.get('worker_enabled', False)
    if not worker_enabled:
        await query.edit_message_text(
            "❌ **Local Worker Not Available**\n\n"
            "The local worker feature is not enabled on this bot.",
            parse_mode="Markdown"
        )
        return

    websocket_server = context.bot_data.get('websocket_server')
    if not websocket_server:
        await query.edit_message_text(
            "❌ **Worker Service Unavailable**\n\n"
            "The worker connection service is not running.",
            parse_mode="Markdown"
        )
        return

    # Check if worker already connected
    worker_manager = websocket_server.worker_manager
    if worker_manager.is_user_connected(user_id):
        machine_name = worker_manager.get_machine_name(user_id)
        await query.edit_message_text(
            f"✅ **Worker Already Connected**\n\n"
            f"**Machine:** {machine_name}\n\n"
            f"Your local worker is already connected.",
            parse_mode="Markdown"
        )
        return

    # Generate token and show connection instructions
    token_manager = websocket_server.token_manager
    token = token_manager.generate_token(user_id)

    server_url = context.bot_data.get('worker_http_url', 'http://localhost:8080')

    mac_linux_command = f"curl -s {server_url}/worker | python3 - --token={token}"
    windows_command = f"irm {server_url}/worker | python - --token={token}"

    message = (
        "🔌 **Connect Local Worker**\n\n"
        "**Mac/Linux:**\n"
        f"```bash\n{mac_linux_command}\n```\n\n"
        "**Windows:**\n"
        f"```powershell\n{windows_command}\n```\n\n"
        "⏱️ Token expires in 5 minutes"
    )

    await query.edit_message_text(message, parse_mode="Markdown")
    logger.info(f"Generated connection token for user {user_id} (via button)")


# ============================================================================
# /worker autostart Command Handler
# ============================================================================


@telegram_handler
@log_user_action("/worker command")
async def handle_worker_autostart_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /worker command - show worker status or autostart instructions.

    If worker is connected: shows status with machine name.
    If worker is not connected: shows autostart/setup instructions.

    Args:
        update: Telegram update object
        context: Bot context
    """
    user_id = update.effective_user.id

    # Check if worker feature is enabled
    worker_enabled = context.bot_data.get('worker_enabled', False)
    if not worker_enabled:
        await update.message.reply_text(
            "❌ **Local Worker Not Available**\n\n"
            "The local worker feature is not enabled on this bot.",
            parse_mode="Markdown"
        )
        return

    # Check if websocket server is available
    websocket_server = context.bot_data.get('websocket_server')
    if not websocket_server:
        await update.message.reply_text(
            "❌ **Worker Service Unavailable**\n\n"
            "The worker connection service is not running.",
            parse_mode="Markdown"
        )
        return

    # Check if worker is connected - show status instead of setup
    worker_manager = websocket_server.worker_manager
    if worker_manager.is_user_connected(user_id):
        machine_name = worker_manager.get_machine_name(user_id)
        await update.message.reply_text(
            f"✅ **Local Worker Connected**\n\n"
            f"**Machine:** `{machine_name}`\n\n"
            f"Your worker is connected and ready to execute ML jobs.\n\n"
            f"**Available Commands:**\n"
            f"• `/train` - Train ML model using local files\n"
            f"• `/predict` - Make predictions with trained models\n\n"
            f"To disconnect, stop the worker script on your machine.",
            parse_mode="Markdown"
        )
        return

    # If not connected, show autostart/setup instructions
    # Build instructions message
    message = (
        "🔧 **Auto-start Worker on Boot**\n\n"
        "To have the worker start automatically when your computer boots:\n\n"
        "**Step 1:** Connect your worker first using `/connect`\n\n"
        "**Step 2:** Once connected, run the setup command on your machine:\n\n"
        "**Mac:**\n"
        "```bash\n"
        "curl -s <server-url>/worker | python3 - --token=<your-token> --autostart on\n"
        "```\n\n"
        "**Linux:**\n"
        "```bash\n"
        "curl -s <server-url>/worker | python3 - --token=<your-token> --autostart on\n"
        "```\n\n"
        "**Windows (PowerShell):**\n"
        "```powershell\n"
        "irm <server-url>/worker | python - --token=<your-token> --autostart on\n"
        "```\n\n"
        "**Platform-specific Setup:**\n"
        "- **Mac**: Creates launchd service in `~/Library/LaunchAgents/`\n"
        "- **Linux**: Creates systemd user service in `~/.config/systemd/user/`\n"
        "- **Windows**: Creates Task Scheduler task to run on login\n\n"
        "**To Remove Auto-start:**\n"
        "Replace `--autostart on` with `--autostart off`\n\n"
        "**Requirements:**\n"
        "- Worker must be connected at least once before setting up auto-start\n"
        "- Token in command must be valid (get fresh one from `/connect`)\n"
        "- Worker script will be saved to `~/.statsbot/worker/statsbot_worker.py`\n\n"
        "**Verify Setup:**\n"
        "After setup, restart your computer and check if worker reconnects automatically."
    )

    await update.message.reply_text(message, parse_mode="Markdown")
    logger.info(f"Sent autostart instructions to user {user_id}")
