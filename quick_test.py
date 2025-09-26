#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Set up paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

print("🤖 Quick Bot Test")
print("================")

# Test token
token = os.getenv("TELEGRAM_BOT_TOKEN")
if token:
    print(f"✅ Token: {token[:10]}...")
else:
    print("❌ No token found!")
    sys.exit(1)

# Test basic imports
try:
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
    from telegram import Update
    print("✅ Telegram imports working")
except Exception as e:
    print(f"❌ Telegram import error: {e}")
    sys.exit(1)

# Test our imports
try:
    from src.bot.handlers import start_handler, message_handler
    print("✅ Handler imports working")
except Exception as e:
    print(f"❌ Handler import error: {e}")
    sys.exit(1)

# Simple bot test
print("\n🚀 Starting simple bot...")
print("Send /start or any message to test!")
print("Press Ctrl+C to stop\n")

async def simple_start(update, context):
    await update.message.reply_text("✅ /start works!")
    print(f"📨 Received /start from {update.effective_user.username}")

async def simple_message(update, context):
    text = update.message.text
    await update.message.reply_text(f"✅ Got your message: '{text}'")
    print(f"📨 Received message: '{text}' from {update.effective_user.username}")

# Create and run bot
app = Application.builder().token(token).build()
app.add_handler(CommandHandler("start", simple_start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, simple_message))

print("Bot starting...")
app.run_polling()