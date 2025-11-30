#!/bin/bash

# Start the Parser Test Bot for Telegram testing
echo "🧪 Starting Parser Test Bot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with TELEGRAM_BOT_TOKEN"
    exit 1
fi

# Set PYTHONPATH to ensure imports work
export PYTHONPATH="$(pwd):$(pwd)/src:$PYTHONPATH"

# Start the parser test bot
echo "🚀 Starting Parser Test Bot..."
echo "This bot will show you exactly how messages are parsed!"
echo "Use /start, /help, /debug, and /data commands"
echo "Press Ctrl+C to stop the bot"
echo "---"

python test_bot_with_parser.py