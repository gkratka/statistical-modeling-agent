#!/bin/bash
echo "🛑 Ensuring clean environment..."
pkill -f telegram_bot.py
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

echo "🚀 Starting bot with version verification..."
echo "Expected version: DataLoader-v2.0-NUCLEAR-FIX"
source venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -B src/bot/telegram_bot.py