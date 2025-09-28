#!/bin/bash
echo "🔍 Bot Health Check - $(date)"
echo "================================"

# Check process count
PROCESS_COUNT=$(pgrep -f telegram_bot.py | wc -l)
echo "Bot processes running: $PROCESS_COUNT"

# Check working directory
if [ $PROCESS_COUNT -eq 1 ]; then
    PID=$(pgrep -f telegram_bot.py)
    CWD=$(lsof -p $PID 2>/dev/null | grep cwd | awk '{print $NF}')
    echo "Working directory: $CWD"
fi

# Run quick test
cd /Users/gkratka/Documents/statistical-modeling-agent
source venv/bin/activate
python -m pytest tests/test_bot_instance_fix.py::TestBotInstanceFix::test_no_development_mode_text_in_codebase -q
echo "================================"