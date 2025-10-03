#!/bin/bash

echo "🔄 Statistical Modeling Bot Restart Script"
echo "=========================================="

# Find and kill existing bot process
echo "📍 Finding bot process..."
BOT_PID=$(pgrep -f "telegram_bot.py")

if [ -z "$BOT_PID" ]; then
    echo "⚠️  No bot process found running"
else
    echo "🛑 Stopping bot (PID: $BOT_PID)..."
    kill $BOT_PID
    sleep 2

    # Force kill if still running
    if pgrep -f "telegram_bot.py" > /dev/null; then
        echo "⚡ Force killing bot..."
        pkill -9 -f "telegram_bot.py"
        sleep 1
    fi

    echo "✅ Bot stopped"
fi

# Verify Python environment
echo "🐍 Checking Python environment..."
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if TELEGRAM_BOT_TOKEN is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "⚠️  TELEGRAM_BOT_TOKEN not set in environment"
    echo "Loading from .env file..."
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "❌ .env file not found!"
        exit 1
    fi
fi

# Start bot
echo "🚀 Starting bot..."
nohup python3 src/bot/telegram_bot.py > bot.log 2>&1 &
NEW_PID=$!

sleep 2

# Verify bot started
if pgrep -f "telegram_bot.py" > /dev/null; then
    echo "✅ Bot started successfully (PID: $NEW_PID)"
    echo "📋 Logs: tail -f bot.log"
    echo ""
    echo "📱 Test with:"
    echo "   1. Send /version to bot"
    echo "   2. Upload housing_data.csv"
    echo "   3. Send: Train a model to predict house prices"
    echo ""
    echo "Expected: Column selection prompt (NOT error message)"
else
    echo "❌ Bot failed to start!"
    echo "Check bot.log for errors"
    exit 1
fi
