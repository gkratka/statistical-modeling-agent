#!/bin/bash
echo "🛑 Stopping existing bot processes..."
# Use SIGTERM first for graceful shutdown
pkill -TERM -f telegram_bot.py
sleep 2

# Check if process still exists
if pgrep -f telegram_bot.py > /dev/null; then
    echo "⚠️ Process still running, using force kill..."
    pkill -9 -f telegram_bot.py
    sleep 1
fi

# Verify cleanup
if pgrep -f telegram_bot.py > /dev/null; then
    echo "❌ Failed to stop bot processes"
    exit 1
else
    echo "✅ All bot processes stopped"
fi

# Clean Python cache
echo "🧹 Cleaning Python cache..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "🚀 Starting fresh bot instance..."
echo "Version: DataLoader-v2.0-NUCLEAR-FIX"
source venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 exec python -B src/bot/telegram_bot.py