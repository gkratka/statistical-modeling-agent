# Railway Deployment Checklist

Quick reference checklist for deploying to Railway.

## Pre-Deployment Setup

### Local Repository
- [x] `runtime.txt` exists with `python-3.12`
- [x] `requirements-railway.txt` has all dependencies
- [x] `railway.json` configured (already exists)
- [x] `Procfile` exists (already exists)
- [x] `.railwayignore` configured (already exists)
- [ ] Latest code committed to git
- [ ] Code pushed to GitHub main branch

### GitHub Repository
- [x] Repository URL: `https://github.com/gkratka/statistical-modeling-agent.git`
- [ ] Main branch up to date
- [ ] No merge conflicts
- [ ] `.env` not committed (verify `.gitignore`)

### API Keys Ready
- [ ] Telegram Bot Token (from @BotFather)
- [ ] Anthropic API Key (from https://console.anthropic.com)

---

## Railway Project Setup

### 1. Create Project (5 min)
- [ ] Sign in to https://railway.app
- [ ] Click "New Project"
- [ ] Select "Deploy from GitHub repo"
- [ ] Choose `gkratka/statistical-modeling-agent`
- [ ] Railway detects build configuration ✓

### 2. Configure Volumes (3 min)
**IMPORTANT**: Create BEFORE first deployment

Go to Settings → Volumes and create:

- [ ] **Volume 1**: Models Storage
  - Name: `models-storage`
  - Mount Path: `/app/models`
  - Size: 10 GB

- [ ] **Volume 2**: Data Storage
  - Name: `data-storage`
  - Mount Path: `/app/data`
  - Size: 5 GB

- [ ] **Volume 3**: Sessions Storage
  - Name: `sessions-storage`
  - Mount Path: `/app/.sessions`
  - Size: 1 GB

### 3. Set Environment Variables (2 min)
Go to Variables tab:

- [ ] `TELEGRAM_BOT_TOKEN` = `<paste_your_token>`
- [ ] `ANTHROPIC_API_KEY` = `<paste_your_key>`
- [ ] `LOG_LEVEL` = `INFO` (optional)

### 4. Deploy (Auto)
- [ ] Railway auto-triggers deployment
- [ ] Wait 3-5 minutes for build
- [ ] Check Deployments tab shows "Success"

---

## Post-Deployment Verification

### Check Logs (2 min)
Go to Logs tab and verify:

- [ ] ✅ "Initializing Telegram bot..."
- [ ] ✅ "Loading configuration..."
- [ ] ✅ "ML Engine initialized"
- [ ] ✅ "Bot started successfully"
- [ ] ✅ "Polling for updates..."
- [ ] ❌ No error messages

### Test Bot Basic Functions (5 min)

- [ ] Open Telegram, find bot
- [ ] Send `/start` → Receives welcome message
- [ ] Send `/help` → Shows command list
- [ ] Upload CSV file → Confirms upload
- [ ] Send `show me the data` → Displays first rows

### Test ML Training (10 min)

- [ ] Upload training data (CSV with numeric columns)
- [ ] Send `/train`
- [ ] Follow workflow:
  - [ ] Choose task type (regression/classification)
  - [ ] Select target column
  - [ ] Select feature columns
  - [ ] Choose model type
  - [ ] Training completes successfully
- [ ] Verify metrics displayed (MSE, R², etc.)
- [ ] Send `/models` → Shows trained model in list

### Test Persistence (5 min)

- [ ] Go to Railway → Settings → Restart
- [ ] Wait for bot to restart (30 seconds)
- [ ] Send `/models` → Trained model still listed ✅
- [ ] Make prediction with existing model
- [ ] Verify logs persisted in `/app/data/logs`

---

## Health Monitoring (First 24 Hours)

### Day 1 Monitoring Checklist

**Hour 1:**
- [ ] No crashes in logs
- [ ] Bot responds to commands
- [ ] Memory usage stable (<80%)

**Hour 6:**
- [ ] Check Metrics tab for resource usage
- [ ] Verify no repeated errors in logs
- [ ] Test file upload still working

**Hour 24:**
- [ ] Review full day logs for patterns
- [ ] Check volume storage usage
- [ ] Verify uptime >99%
- [ ] Confirm no OOM (Out of Memory) errors

---

## Troubleshooting Quick Reference

### Build Failed
- [ ] Check `runtime.txt` has `python-3.12`
- [ ] Verify `requirements-railway.txt` syntax
- [ ] Check build logs for specific error

### Bot Crashes on Start
- [ ] Verify `TELEGRAM_BOT_TOKEN` is set correctly
- [ ] Verify `ANTHROPIC_API_KEY` is set correctly
- [ ] Check no other bot instance running (Telegram conflict)

### Models Don't Persist
- [ ] Verify volumes created (`/app/models`, `/app/data`, `/app/.sessions`)
- [ ] Check volume mount paths are exact
- [ ] Restart deployment after adding volumes

### "Another instance running" Error
- [ ] Stop local bot instance
- [ ] Check no duplicate Railway deployments
- [ ] Railway → Settings → Restart

---

## Optimization Checklist (After 1 Week)

### Performance
- [ ] Review average response time in logs
- [ ] Check memory usage patterns (Metrics tab)
- [ ] Identify slow operations (ML training times)

### Cost Optimization
- [ ] Review actual volume usage vs allocated
- [ ] Reduce volume sizes if usage <50%
- [ ] Clean old models to save space

### Feature Flags
- [ ] Confirm `local_data.enabled: false` (security)
- [ ] Review which ML models users actually use
- [ ] Consider enabling cloud training for heavy models

---

## Security Audit (Monthly)

- [ ] Rotate `TELEGRAM_BOT_TOKEN`
- [ ] Rotate `ANTHROPIC_API_KEY`
- [ ] Review Railway access logs
- [ ] Verify no secrets in logs
- [ ] Check `.env` still in `.gitignore`
- [ ] Audit Railway team access permissions

---

## Rollback Plan

If critical issue occurs:

1. [ ] Go to Railway → Deployments tab
2. [ ] Find last known good deployment
3. [ ] Click "⋯" → "Redeploy"
4. [ ] Monitor logs for success
5. [ ] Fix issue in local branch
6. [ ] Test locally before re-deploying

---

## Success Criteria

**Deployment is successful when:**

✅ Build completes without errors (3-5 min)
✅ Bot responds to `/start` in Telegram
✅ File upload and parsing works
✅ ML training completes successfully
✅ Trained models persist after restart
✅ No crashes for 24 hours
✅ Memory usage stable (<80%)
✅ Logs show no critical errors
✅ All 3 volumes mounted and accessible

---

## Next Steps After Successful Deployment

1. [ ] **Beta Testing**: Invite 3-5 users to test
2. [ ] **Documentation**: Share bot commands and examples
3. [ ] **Monitoring**: Set up Railway alerts for downtime
4. [ ] **Backup**: Export important models weekly
5. [ ] **Scale**: Monitor usage, upgrade plan if needed

---

## Quick Commands

```bash
# Push code to GitHub
git add runtime.txt
git commit -m "Add Railway runtime specification"
git push origin main

# Railway CLI (optional)
npm i -g @railway/cli
railway login
railway link
railway logs
railway status

# Check local bot still works
python src/bot/telegram_bot.py
```

---

## Support Resources

- Railway Dashboard: https://railway.app/dashboard
- Railway Docs: https://docs.railway.app
- Project Deployment Guide: `RAILWAY_DEPLOYMENT.md`
- Bot Deployment Docs: `docs/bot-deployment-server.md`

---

**Status**: Ready for deployment ✅
**Estimated Setup Time**: 25 minutes
**Monthly Cost**: ~$10.40 (Hobby plan + volumes)
