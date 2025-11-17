# XGBoost Setup Guide for macOS

## Problem
XGBoost training fails with error:
```
XGBoost Library (libxgboost.dylib) could not be loaded.
Library not loaded: @rpath/libomp.dylib
```

## Root Cause
XGBoost requires OpenMP runtime (`libomp`) which is not installed on your system.

## Solution (One-Time Setup)

### Step 1: Install Homebrew (if not already installed)

Open Terminal and run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**After installation**, add Homebrew to your PATH:
```bash
# For Apple Silicon (M1/M2/M3):
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# For Intel Macs:
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/usr/local/bin/brew shellenv)"
```

### Step 2: Install OpenMP Runtime

```bash
brew install libomp
```

### Step 3: Verify Installation

```bash
# Check libomp is installed
ls /opt/homebrew/opt/libomp/lib/libomp.dylib  # Apple Silicon
# OR
ls /usr/local/opt/libomp/lib/libomp.dylib     # Intel

# Test XGBoost import
cd /Users/gkratka/Documents/statistical-modeling-agent
source venv/bin/activate
python3 -c "import xgboost; print(f'XGBoost {xgboost.__version__} ready!')"
```

### Step 4: Restart Bot

```bash
pkill -9 -f "python.*telegram_bot"
rm -f .bot.pid
./scripts/dev_start.sh
```

## Quick Test

After setup, test the full XGBoost workflow:

1. Open Telegram bot
2. Send `/train`
3. Upload CSV with classification task
4. Configure schema
5. Select **"XGBoost Classification"** model
6. Configure 5 parameters (n_estimators, max_depth, etc.)
7. Training should complete successfully!

## Verification Commands

```bash
# Run installation tests
source venv/bin/activate
pytest tests/unit/test_xgboost_installation.py -v

# Should show: 7 passed
```

## Alternative: Use sklearn Gradient Boosting

If you prefer not to install Homebrew, you can use sklearn's gradient boosting instead:

1. Select **"Gradient Boosting (sklearn)"** instead of XGBoost
2. This uses pure Python implementation (slower but no dependencies)
3. Syntax: sklearn's GradientBoostingClassifier/Regressor

## Status

- ✅ XGBoost library installed (v2.1.4)
- ✅ XGBoost workflow code working (all 5 parameter steps complete)
- ❌ libomp.dylib missing (requires Homebrew installation)
- ⏳ **Action Required**: Install Homebrew and libomp (see above)

## Support

After completing steps, verify with:
```bash
source venv/bin/activate
python3 -c "import xgboost; from xgboost import XGBClassifier; print('✓ XGBoost ready')"
```
