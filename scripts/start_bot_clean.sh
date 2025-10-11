#!/bin/bash
# TDD-Driven Bot Startup Script
# Tests: Cleanup verification, startup success, conflict detection

set -e  # Exit on any error

echo "🧪 TDD Bot Startup - Running Tests"
echo "=================================="

# Test 1: Kill existing instances
echo ""
echo "🧹 TEST 1: Cleanup existing processes..."
killall -9 python3 python Python 2>/dev/null || true
killall -9 Python 2>/dev/null || true  # macOS specific
sleep 3

# Test 1 Verification
echo "✓ Verifying cleanup..."
if ps aux | grep -E "[Pp]ython.*telegram_bot" > /dev/null; then
    echo "❌ TEST 1 FAILED: Processes still running after cleanup"
    ps aux | grep -E "[Pp]ython.*telegram_bot"
    exit 1
fi
echo "✅ TEST 1 PASSED: All processes cleaned up"

# Test 1.5: Remove stale PID file
echo ""
echo "🗑️  TEST 1.5: Removing stale PID file..."
if [ -f ".bot.pid" ]; then
    PID_IN_FILE=$(cat .bot.pid)
    if ! ps -p $PID_IN_FILE > /dev/null 2>&1; then
        echo "   Stale PID file found (PID $PID_IN_FILE dead)"
        rm .bot.pid
        echo "✅ TEST 1.5 PASSED: Stale PID file removed"
    else
        echo "❌ TEST 1.5 FAILED: Process $PID_IN_FILE still running"
        exit 1
    fi
else
    echo "✅ TEST 1.5 PASSED: No PID file found"
fi

# Test 2: Clear old logs
echo ""
echo "📝 TEST 2: Clearing logs..."
: > bot_output.log
if [ -s bot_output.log ]; then
    echo "❌ TEST 2 FAILED: Log file not empty"
    exit 1
fi
echo "✅ TEST 2 PASSED: Log file cleared"

# Wait for Telegram API to reset (critical!)
echo ""
echo "⏳ Waiting 30 seconds for Telegram API reset..."
echo "   (This ensures the API releases the getUpdates lock)"
for i in {30..1}; do
    printf "\r   Waiting... %2d seconds remaining" $i
    sleep 1
done
printf "\n"
echo "✅ API reset wait complete"

# Test 3: Start bot
echo ""
echo "🚀 TEST 3: Starting bot..."
python3 src/bot/telegram_bot.py > bot_output.log 2>&1 &
BOT_PID=$!
echo "   Bot started with PID: $BOT_PID"

# Test 3 Verification
echo "✓ Verifying process started..."
sleep 3
if ! ps -p $BOT_PID > /dev/null 2>&1; then
    echo "❌ TEST 3 FAILED: Bot process died immediately"
    echo ""
    echo "Last 20 lines of log:"
    tail -20 bot_output.log
    exit 1
fi
echo "✅ TEST 3 PASSED: Bot process running (PID: $BOT_PID)"

# Test 4: Check for conflicts
echo ""
echo "🔍 TEST 4: Checking for conflicts..."
sleep 5
if grep -q "Conflict" bot_output.log 2>/dev/null; then
    echo "❌ TEST 4 FAILED: Conflict detected in logs"
    echo ""
    echo "Conflict errors found:"
    grep "Conflict" bot_output.log | head -5
    echo ""
    echo "This means another bot instance is running somewhere."
    echo "Try waiting 60 seconds and running this script again."
    exit 1
fi
echo "✅ TEST 4 PASSED: No conflicts detected"

# Test 5: Verify single instance
echo ""
echo "🔢 TEST 5: Verifying single instance..."
PROCESS_COUNT=$(ps aux | grep -E "[Pp]ython.*telegram_bot" | wc -l | tr -d ' ')
if [ "$PROCESS_COUNT" -ne 1 ]; then
    echo "❌ TEST 5 FAILED: Expected 1 process, found $PROCESS_COUNT"
    ps aux | grep -E "[Pp]ython.*telegram_bot"
    exit 1
fi
echo "✅ TEST 5 PASSED: Exactly 1 bot instance running"

# All tests passed!
echo ""
echo "=================================="
echo "✅ ALL TESTS PASSED! Bot is healthy"
echo "=================================="
echo ""
echo "📊 Status:"
echo "   PID: $BOT_PID"
echo "   Log: bot_output.log"
echo ""
echo "🔍 Monitor with:"
echo "   tail -f bot_output.log"
echo ""
echo "✅ Health check:"
echo "   ./scripts/check_bot_health.sh"
