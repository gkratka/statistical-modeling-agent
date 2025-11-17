# Development vs Production Bot Setup

Guide for managing separate development and production Telegram bots to prevent conflicts.

## System Overview

**Two-Bot Architecture:**
- **Development Bot** (`@statsmodeldev_bot`) - For local testing
- **Production Bot** (your original bot) - Deployed on Railway for users

**Key Principle:** Same codebase, different `TELEGRAM_BOT_TOKEN` environment variables.

---

## Quick Setup

### 1. You Already Have Development Bot ✅

Bot: `@statsmodeldev_bot`
Token: `8328756008:AAHwHiOfR-mPtdc14fAREcphJmTY4d9xmgc`

### 2. Configure Local Environment

Edit your local `.env` file:

```bash
# Development Bot Token (for local testing)
TELEGRAM_BOT_TOKEN=8328756008:AAHwHiOfR-mPtdc14fAREcphJmTY4d9xmgc

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# File Path Password
FILE_PATH_PASSWORD=senha123
```

**Security:** `.env` is in `.gitignore` - never committed to GitHub.

### 3. Railway Environment Variables

Keep production bot token in Railway:

1. Go to Railway dashboard → Your project
2. Click **Variables** tab
3. Verify `TELEGRAM_BOT_TOKEN` = your **production** bot token (NOT dev token)
4. Railway will use production token when deployed

---

## Development Workflow

### Local Testing
```bash
# Start local bot (uses dev token from .env)
source venv/bin/activate
python src/bot/telegram_bot.py
```

**Result:** Dev bot `@statsmodeldev_bot` starts responding in Telegram.

### Test Changes
1. Make code changes locally
2. Run local bot
3. Test in Telegram with `@statsmodeldev_bot`
4. Verify features work correctly

### Deploy to Production
```bash
# Commit and push to GitHub main branch
git add .
git commit -m "Add new feature"
git push origin main
```

**Result:** Railway auto-deploys, production bot updates with new code.

---

## Bot Identification

| Environment | Bot Username | Token Location | Who Uses |
|-------------|-------------|----------------|----------|
| **Local Dev** | `@statsmodeldev_bot` | Local `.env` file | You (testing) |
| **Railway Prod** | Your original bot | Railway env vars | Public users |

### How to Know Which Bot is Responding
- Check bot username in Telegram chat header
- Dev bot: `@statsmodeldev_bot`
- Prod bot: Your original bot username

---

## Common Scenarios

### Scenario 1: Testing New Feature Locally
```bash
# 1. Update code
vim src/bot/handlers/new_feature.py

# 2. Start dev bot
python src/bot/telegram_bot.py

# 3. Test with @statsmodeldev_bot in Telegram
# 4. Stop bot (Ctrl+C) when done
```

### Scenario 2: Deploying to Production
```bash
# 1. Ensure local tests pass
python src/bot/telegram_bot.py  # Test with dev bot

# 2. Commit and push
git add .
git commit -m "Implement new feature"
git push origin main

# 3. Railway auto-deploys (3-5 min)
# 4. Test with production bot in Telegram
```

### Scenario 3: Running Both Simultaneously
- **Dev bot (local):** Testing new features
- **Prod bot (Railway):** Serving real users
- No conflicts - different tokens, different Telegram chats

---

## Token Management

### Security Best Practices

✅ **DO:**
- Keep dev token in local `.env` (gitignored)
- Keep prod token in Railway environment variables
- Never commit tokens to GitHub
- Rotate tokens periodically

❌ **DON'T:**
- Commit `.env` to version control
- Share tokens publicly
- Use prod token in local `.env`
- Use dev token in Railway

### Token Storage Locations

| Token Type | Location | Committed to Git? |
|------------|----------|-------------------|
| Dev token | `.env` (local) | ❌ No (gitignored) |
| Prod token | Railway env vars | ❌ No (Railway only) |

---

## Troubleshooting

### Issue: "Conflict: terminated by other getUpdates request"

**Cause:** Two bots using same token.

**Fix:**
1. Check local `.env` has **dev** token
2. Check Railway has **prod** token
3. Kill any running local bot: `pkill -f telegram_bot.py`
4. Restart with correct token

### Issue: Changes Not Appearing in Production

**Cause:** Railway not redeployed or using cached build.

**Fix:**
1. Verify commit pushed to GitHub: `git log --oneline -1`
2. Check Railway Deployments tab shows new commit
3. Force redeploy: Railway Settings → Redeploy

### Issue: Can't Find Dev Bot in Telegram

**Solution:**
1. Search for `@statsmodeldev_bot` in Telegram
2. Send `/start` to activate
3. Bot must be running locally to respond

---

## Verification Checklist

After setup, verify both bots work:

- [ ] Local `.env` contains **dev** token (`8328756008:...`)
- [ ] Railway Variables contain **prod** token
- [ ] Start local bot → Dev bot responds in Telegram
- [ ] Railway bot still responds to users
- [ ] Both bots can run simultaneously
- [ ] No "conflict" errors in logs

---

## Quick Reference

### Start Local Dev Bot
```bash
source venv/bin/activate
python src/bot/telegram_bot.py
```

### Check Which Token is Configured Locally
```bash
grep TELEGRAM_BOT_TOKEN .env
```

### Check Railway Token
1. Railway dashboard → Project → Variables
2. Look for `TELEGRAM_BOT_TOKEN`
3. Should be **prod** token (different from dev)

### Stop Local Bot
```bash
# Ctrl+C in terminal, or:
pkill -f telegram_bot.py
```

---

## Environment Variables Summary

**Local `.env` (Development):**
```bash
TELEGRAM_BOT_TOKEN=8328756008:AAHwHiOfR-mPtdc14fAREcphJmTY4d9xmgc  # Dev bot
ANTHROPIC_API_KEY=your_key_here
FILE_PATH_PASSWORD=senha123
```

**Railway (Production):**
```bash
TELEGRAM_BOT_TOKEN=your_production_bot_token_here  # Prod bot
ANTHROPIC_API_KEY=your_key_here
FILE_PATH_PASSWORD=senha123
```

---

## Support Resources

- **Telegram Bot API:** https://core.telegram.org/bots/api
- **@BotFather Commands:** Message @BotFather in Telegram, send `/help`
- **Railway Docs:** https://docs.railway.app
- **Project Deployment:** `RAILWAY_DEPLOYMENT.md`

---

**Last Updated:** 2025-11-16
**Status:** ✅ Two-bot system active
