#!/usr/bin/env python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get token
token = os.getenv("TELEGRAM_BOT_TOKEN")

if not token:
    print("ERROR: No TELEGRAM_BOT_TOKEN found in .env file!")
    print("Please check your .env file")
else:
    print(f"Token found: {token[:10]}...")  # Print first 10 chars to verify

    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram import Update

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text('Bot is working! 🎉')

    # Create and run bot
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))

    print("Bot starting... Press Ctrl+C to stop")
    print("Go to Telegram and send /start to your bot")
    app.run_polling()