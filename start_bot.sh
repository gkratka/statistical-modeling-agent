#!/bin/bash

# Start the Statistical Modeling Agent Telegram Bot
echo "🤖 Starting Statistical Modeling Agent Bot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating it..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
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

# Check dependencies
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Start the bot
echo "🚀 Starting bot..."
echo "Press Ctrl+C to stop the bot"
echo "---"

python src/bot/telegram_bot.py