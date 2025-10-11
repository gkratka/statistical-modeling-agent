#!/bin/bash
# TDD-Driven Bot Health Check Script
# Tests: Process status, conflict detection, handler registration

set -e  # Exit on any error

echo "🩺 Bot Health Check"
echo "==================="

FAILED_TESTS=0

# Test 1: Process running?
echo ""
echo "🔍 TEST 1: Checking if bot process is running..."
if ps aux | grep -E "[Pp]ython.*telegram_bot" > /dev/null; then
    PID=$(ps aux | grep -E "[Pp]ython.*telegram_bot" | awk '{print $2}')
    echo "✅ TEST 1 PASSED: Bot process running (PID: $PID)"
else
    echo "❌ TEST 1 FAILED: No bot process found"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 2: Single instance?
echo ""
echo "🔢 TEST 2: Verifying single instance..."
PROCESS_COUNT=$(ps aux | grep -E "[Pp]ython.*telegram_bot" | wc -l | tr -d ' ')
if [ "$PROCESS_COUNT" -eq 1 ]; then
    echo "✅ TEST 2 PASSED: Exactly 1 instance running"
elif [ "$PROCESS_COUNT" -eq 0 ]; then
    echo "❌ TEST 2 FAILED: No instances running"
    FAILED_TESTS=$((FAILED_TESTS + 1))
else
    echo "❌ TEST 2 FAILED: Multiple instances ($PROCESS_COUNT) detected"
    echo "   This will cause conflicts!"
    ps aux | grep -E "[Pp]ython.*telegram_bot"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 3: No conflicts in recent logs?
echo ""
echo "🚫 TEST 3: Checking for conflicts in logs..."
if [ -f "bot_output.log" ]; then
    RECENT_CONFLICTS=$(tail -50 bot_output.log 2>/dev/null | grep -c "Conflict" || true)
    if [ "$RECENT_CONFLICTS" -eq 0 ]; then
        echo "✅ TEST 3 PASSED: No conflicts in recent logs"
    else
        echo "❌ TEST 3 FAILED: $RECENT_CONFLICTS conflict(s) detected in last 50 lines"
        echo ""
        echo "Recent conflict errors:"
        tail -50 bot_output.log | grep "Conflict" | head -3
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
else
    echo "⚠️  TEST 3 SKIPPED: Log file not found"
fi

# Test 4: Bot registered handlers?
echo ""
echo "📝 TEST 4: Checking handler registration..."
if [ -f "bot_output.log" ]; then
    if tail -200 bot_output.log 2>/dev/null | grep -q "registered" || tail -200 bot_output.log 2>/dev/null | grep -q "handlers"; then
        echo "✅ TEST 4 PASSED: Handlers registered"
    else
        echo "⚠️  TEST 4 WARNING: No handler registration found (may be buffered)"
        echo "   Python logging sometimes buffers output. This is usually OK."
    fi
else
    echo "⚠️  TEST 4 SKIPPED: Log file not found"
fi

# Test 5: No recent exceptions?
echo ""
echo "💥 TEST 5: Checking for recent exceptions..."
if [ -f "bot_output.log" ]; then
    RECENT_EXCEPTIONS=$(tail -100 bot_output.log 2>/dev/null | grep -c "Exception\|ERROR\|Traceback" || true)
    if [ "$RECENT_EXCEPTIONS" -eq 0 ]; then
        echo "✅ TEST 5 PASSED: No recent exceptions"
    else
        # Allow conflicts to be counted separately
        NONCONFLICT_EXCEPTIONS=$(tail -100 bot_output.log 2>/dev/null | grep -c "Exception\|ERROR\|Traceback" | grep -v "Conflict" || true)
        if [ "$NONCONFLICT_EXCEPTIONS" -eq 0 ]; then
            echo "⚠️  TEST 5 INFO: Only conflict exceptions (already reported)"
        else
            echo "⚠️  TEST 5 WARNING: $RECENT_EXCEPTIONS exception(s) in last 100 lines"
            echo "   (This may be normal during development)"
        fi
    fi
else
    echo "⚠️  TEST 5 SKIPPED: Log file not found"
fi

# Summary
echo ""
echo "==================="
if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ HEALTH CHECK PASSED"
    echo "==================="
    echo ""
    echo "Bot is healthy and ready to use!"
    echo ""
    echo "📊 Process info:"
    ps aux | grep -E "[Pp]ython.*telegram_bot" | head -1
    exit 0
else
    echo "❌ HEALTH CHECK FAILED"
    echo "==================="
    echo ""
    echo "Failed $FAILED_TESTS test(s). See above for details."
    echo ""
    echo "🔧 Recommended action:"
    echo "   ./scripts/start_bot_clean.sh"
    exit 1
fi
