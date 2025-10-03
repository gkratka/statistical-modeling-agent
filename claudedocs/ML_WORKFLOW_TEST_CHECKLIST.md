# ML Workflow Test Checklist

## Prerequisites
- [ ] Bot has been restarted using `./scripts/restart_bot.sh`
- [ ] Bot is running (check with `ps aux | grep telegram_bot.py`)
- [ ] Logs are accessible (`tail -f bot.log`)

## Test Sequence

### Test 1: Verify Bot Version
**Action**: Send `/version` to bot
**Expected**:
```
🤖 Bot Version Information

Code Version: v2.1.0-ml-workflow-fix
ML Workflow: ✅ Enabled
Error Handling: ✅ Enhanced
Parameter Keys: target_column, feature_columns

If you see an old version, the bot needs to be restarted.
```
**Actual**: _____________________
**Status**: ⬜ Pass / ⬜ Fail

### Test 2: Upload Data
**Action**: Send `housing_data.csv` file to bot
**Expected**: "✅ Data Successfully Loaded - Shape: 20 rows × 7 columns"
**Actual**: _____________________
**Status**: ⬜ Pass / ⬜ Fail

### Test 3: ML Training Request (Invalid Target)
**Action**: Send "Train a model to predict house prices"
**Expected**:
```
🎯 Select Target Column:

1. sqft
2. bedrooms
3. bathrooms
4. age
5. price
6. location
7. condition

Type the column name or number.
```
**Actual**: _____________________
**Status**: ⬜ Pass / ⬜ Fail

**⚠️ CRITICAL**: This should show column selection, NOT "❌ Error Processing Request"

### Test 4: Select Target
**Action**: Reply "5" or "price"
**Expected**: Asks for feature selection
**Actual**: _____________________
**Status**: ⬜ Pass / ⬜ Fail

### Test 5: Check Logs
**Action**: `grep "ML WORKFLOW" bot.log`
**Expected**: Shows workflow initiation logs like:
```
🔧 CODE VERSION: v2.1.0-ml-workflow-fix
🔧 ML WORKFLOW CHECK: task_type=ml_train, target=house, in_columns=False
🔧 WORKFLOW SHOULD START: True
🔧 STARTING ML TRAINING WORKFLOW...
🔧 ML WORKFLOW INITIATED - RETURNED TO USER
```
**Actual**: _____________________
**Status**: ⬜ Pass / ⬜ Fail

## Failure Recovery

### If Test 1 Fails (Version command not found)
1. Bot hasn't been restarted or old code is running
2. Actions:
   ```bash
   # Kill bot
   pkill -9 -f telegram_bot.py

   # Restart with script
   ./scripts/restart_bot.sh

   # Verify running
   ps aux | grep telegram_bot.py

   # Try Test 1 again
   ```

### If Test 3 Fails (Still shows error)
1. Check logs for version marker:
   ```bash
   tail -50 bot.log | grep "CODE VERSION"
   ```

2. If no "v2.1.0" appears:
   - Bot hasn't restarted properly
   - Python cached old modules

3. Fix:
   ```bash
   # Force kill all Python processes running telegram_bot.py
   pkill -9 -f telegram_bot.py

   # Wait 5 seconds
   sleep 5

   # Restart
   ./scripts/restart_bot.sh

   # Verify logs show new version
   tail -f bot.log
   ```

4. Look for in logs:
   - ✅ "CODE VERSION: v2.1.0-ml-workflow-fix" → Good!
   - ❌ No version line → Old code running

### If Test 3 Shows Different Error
1. Check exact error message in bot response
2. Check logs for exception traceback:
   ```bash
   grep -A 20 "UNEXPECTED ERROR" bot.log
   ```
3. Look for import errors or missing dependencies

## Success Criteria

All tests must pass:
- ✅ Bot shows correct version (v2.1.0-ml-workflow-fix)
- ✅ Data uploads successfully
- ✅ ML request triggers workflow (column selection prompt)
- ✅ Workflow completes target selection
- ✅ Logs show "ML WORKFLOW INITIATED"

## Common Issues

### Issue: "Unknown command '/version'"
**Cause**: Bot running old code without version handler
**Fix**: Restart bot with `./scripts/restart_bot.sh`

### Issue: Still shows "Error Processing Request"
**Cause**: Python module cache, bot not restarted
**Fix**:
```bash
pkill -9 -f telegram_bot.py
sleep 5
./scripts/restart_bot.sh
grep "CODE VERSION" bot.log  # Verify shows v2.1.0
```

### Issue: Bot won't start
**Cause**: Missing dependencies, environment issues
**Fix**:
```bash
# Check bot.log for errors
cat bot.log

# Verify virtual environment
source venv/bin/activate
python3 -c "import telegram; print('OK')"

# Check TELEGRAM_BOT_TOKEN
grep TELEGRAM_BOT_TOKEN .env
```

### Issue: "ModuleNotFoundError" in logs
**Cause**: Missing Python packages
**Fix**:
```bash
source venv/bin/activate
pip install -r requirements.txt
./scripts/restart_bot.sh
```

## Diagnostic Commands

```bash
# Check if bot is running
ps aux | grep telegram_bot.py

# Check bot logs (last 50 lines)
tail -50 bot.log

# Watch logs in real-time
tail -f bot.log

# Search for specific log messages
grep "CODE VERSION" bot.log
grep "ML WORKFLOW" bot.log
grep "ERROR" bot.log

# Check bot process start time
ps -p $(pgrep -f telegram_bot.py) -o lstart

# Verify Python packages
source venv/bin/activate && pip list | grep telegram
```

## Expected Timeline

- **Test 1-2**: < 30 seconds
- **Test 3-4**: < 1 minute
- **Test 5**: < 30 seconds
- **Total**: < 3 minutes

If tests take longer, check for:
- Network issues (bot can't reach Telegram API)
- Performance issues (check `top` for CPU/memory)
- Database/storage issues

## Notes

- All tests use the SAME uploaded dataset (housing_data.csv)
- Tests must be run in sequence (don't skip Test 2)
- If any test fails, STOP and fix before continuing
- Logs are critical for debugging - always check them first
- Version marker (v2.1.0) is the definitive proof bot restarted

## Contact

If all tests fail after multiple restart attempts:
1. Capture full bot.log
2. Note exact error messages
3. Document all steps taken
4. Check GitHub issues or submit new issue with:
   - Bot log (last 100 lines)
   - Test results
   - System info (`python3 --version`, `uname -a`)
