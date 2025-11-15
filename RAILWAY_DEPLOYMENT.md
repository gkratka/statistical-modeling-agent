# Railway Deployment Guide

Complete step-by-step guide to deploy the Statistical Modeling Telegram Bot on Railway.

## Prerequisites

- [x] GitHub account with repository: `https://github.com/gkratka/statistical-modeling-agent.git`
- [x] Railway account (sign up at https://railway.app)
- [x] Telegram Bot Token (from @BotFather)
- [x] Anthropic API Key (from https://console.anthropic.com)

## Deployment Steps

### Step 1: Push Code to GitHub (if not already done)

```bash
# Add runtime.txt and commit
git add runtime.txt
git commit -m "Add Python 3.12 runtime specification for Railway"

# Push to main branch
git push origin main
```

### Step 2: Create Railway Project

1. Go to https://railway.app and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose repository: `gkratka/statistical-modeling-agent`
5. Railway will automatically detect configuration files:
   - ✅ `railway.json` - Build and start configuration
   - ✅ `Procfile` - Process definition
   - ✅ `requirements-railway.txt` - Python dependencies
   - ✅ `runtime.txt` - Python 3.12 specification

### Step 3: Configure Persistent Volumes

**CRITICAL**: Create volumes BEFORE first deployment to prevent data loss.

Navigate to your Railway project → **Settings** → **Volumes**:

| Volume Name | Mount Path | Size | Purpose |
|-------------|-----------|------|---------|
| `models-storage` | `/app/models` | 10 GB | Trained ML models persistence |
| `data-storage` | `/app/data` | 5 GB | Logs, temp files, cache |
| `sessions-storage` | `/app/.sessions` | 1 GB | User session state |

**To create each volume:**
1. Click **"+ New Volume"**
2. Enter mount path (e.g., `/app/models`)
3. Set size
4. Click **"Add"**

### Step 4: Set Environment Variables

Navigate to **Variables** tab and add:

#### Required Variables

```bash
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### Optional Variables (with defaults)

```bash
LOG_LEVEL=INFO
LOG_FILE=/app/data/logs/bot.log
```

#### How to add variables:
1. Click **"+ New Variable"**
2. Enter `TELEGRAM_BOT_TOKEN` as name
3. Paste your bot token as value
4. Click **"Add"**
5. Repeat for `ANTHROPIC_API_KEY`

### Step 5: Deploy

1. Railway automatically triggers deployment after variable setup
2. Monitor deployment in **Deployments** tab
3. Expected build time: 3-5 minutes
4. Watch for **"Bot started successfully"** in logs

### Step 6: Verify Deployment

#### Check Logs
Navigate to **Logs** tab and verify:
```
✅ "Initializing Telegram bot..."
✅ "Loading configuration..."
✅ "ML Engine initialized"
✅ "Bot started successfully"
✅ "Polling for updates..."
```

#### Test Bot
1. Open Telegram
2. Find your bot (@your_bot_username)
3. Send `/start`
4. Expected response: Welcome message with available commands

#### Test ML Training
1. Upload a CSV file to bot
2. Send `/train`
3. Follow training workflow
4. Verify model is saved (check logs)

#### Test Persistence
1. Train a model
2. Go to Railway dashboard → **Settings** → **Restart**
3. After restart, list models with `/models`
4. Verify trained model still exists ✅

## Configuration Details

### Railway Build Configuration (`railway.json`)

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python src/bot/telegram_bot.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

### Process Definition (`Procfile`)

```
web: python src/bot/telegram_bot.py
```

### Python Runtime (`runtime.txt`)

```
python-3.12
```

### Application Configuration (`config/config.yaml`)

All paths are configured for Railway volumes:
- Models: `/app/models`
- Data: `/app/data`
- Logs: `/app/data/logs`
- Temp: `/app/data/temp`
- Cache: `/app/data/cache`

## Cost Breakdown

### Railway Hobby Plan

| Component | Cost |
|-----------|------|
| Hobby Plan (base) | $5.00/month |
| Persistent Volumes (16GB) | ~$5.40/month |
| **Total** | **~$10.40/month** |

### Volume Pricing
- $0.25/GB per month
- 10GB (models) + 5GB (data) + 1GB (sessions) = 16GB × $0.25 = $4.00/month
- Plus Railway overhead ≈ $5.40/month total

### Cost Optimization Tips
- Start with smaller volumes (5GB models, 2GB data)
- Clean old models periodically
- Monitor usage in Railway dashboard
- Scale volumes as needed

## Troubleshooting

### Issue: Build Fails with "Python version not found"

**Solution**: Verify `runtime.txt` exists with:
```
python-3.12
```

### Issue: Bot starts but crashes immediately

**Cause**: Missing environment variables

**Solution**:
1. Check Railway **Variables** tab
2. Verify `TELEGRAM_BOT_TOKEN` and `ANTHROPIC_API_KEY` are set
3. Check logs for specific error message

### Issue: Models don't persist after restart

**Cause**: Volumes not configured

**Solution**:
1. Go to Railway **Settings** → **Volumes**
2. Add volume with mount path `/app/models`
3. Redeploy application

### Issue: "Another instance is running" error

**Cause**: Multiple bot instances trying to poll Telegram

**Solution**:
1. Stop all other bot instances (local, other servers)
2. Go to Railway → **Settings** → **Restart**
3. Bot has built-in conflict detection with exponential backoff

### Issue: ImportError or ModuleNotFoundError

**Cause**: Missing dependency in `requirements-railway.txt`

**Solution**:
1. Add missing package to `requirements-railway.txt`
2. Commit and push to GitHub
3. Railway auto-redeploys

### Issue: File upload fails with "File too large"

**Telegram Limit**: 10MB for regular files, 50MB for premium users

**Solution**: Use local file path feature (requires configuration):
1. Set `local_data.enabled: true` in config
2. Add `FILE_PATH_PASSWORD` environment variable
3. Configure allowed directories in config
4. **Security Warning**: Only enable if you understand security implications

### Check Deployment Health

```bash
# View real-time logs
railway logs

# Check service status
railway status

# View environment variables
railway variables
```

## Monitoring

### Key Metrics to Monitor

1. **Memory Usage**: Watch for OOM (Out of Memory) errors
   - ML training can be memory-intensive
   - Upgrade Railway plan if needed

2. **CPU Usage**: Neural network training is CPU-intensive
   - Monitor build logs for performance
   - Consider RunPod/AWS for heavy training

3. **Volume Usage**: Track storage consumption
   - Railway dashboard shows volume usage
   - Clean old models when space low

4. **Error Rate**: Monitor logs for exceptions
   - Set up log alerts in Railway
   - Check for repeated errors

### Railway Dashboard Monitoring

1. **Deployments Tab**: Build success/failure history
2. **Logs Tab**: Real-time application logs
3. **Metrics Tab**: CPU, memory, network usage
4. **Settings → Volumes**: Storage usage per volume

## Security Best Practices

### Environment Variables
- ✅ Never commit `.env` to GitHub (already in `.gitignore`)
- ✅ Use Railway's environment variable UI
- ✅ Rotate API keys periodically
- ✅ Use different bot tokens for dev/prod

### Local File Path Feature
- ⚠️ Disabled by default in production (`local_data.enabled: false`)
- ⚠️ Only enable if absolutely necessary
- ⚠️ Requires `FILE_PATH_PASSWORD` for access
- ⚠️ Restrict `allowed_directories` to Railway volumes only

### Data Security
- User data stored only in ephemeral sessions
- No persistent user data without consent
- All script execution is sandboxed
- Input validation on all user inputs

## Advanced: Cloud ML Training Integration

### Optional: Add RunPod for Heavy ML Training

If Railway resources are insufficient for large ML models:

1. Add RunPod API key to Railway environment:
   ```bash
   RUNPOD_API_KEY=your_runpod_key
   ```

2. Enable cloud training in `config/config.yaml`:
   ```yaml
   ml_engine:
     cloud_training:
       enabled: true
       provider: runpod
   ```

3. Large models automatically route to RunPod
4. Results synced back to Railway volumes

See `docs/CLOUD_TRAINING_GUIDE.md` for full setup.

## Deployment Checklist

Use this checklist before going live:

- [ ] `runtime.txt` created with `python-3.12`
- [ ] Code pushed to GitHub main branch
- [ ] Railway project created and connected to GitHub
- [ ] 3 persistent volumes created (`/app/models`, `/app/data`, `/app/.sessions`)
- [ ] `TELEGRAM_BOT_TOKEN` environment variable set
- [ ] `ANTHROPIC_API_KEY` environment variable set
- [ ] Build completes successfully (3-5 minutes)
- [ ] Logs show "Bot started successfully"
- [ ] `/start` command works in Telegram
- [ ] File upload works (test with sample CSV)
- [ ] `/train` workflow completes successfully
- [ ] Model persists after Railway restart
- [ ] No errors in logs for 24 hours
- [ ] Local file path feature disabled (`local_data.enabled: false`)
- [ ] Monitoring set up (Railway alerts)

## Next Steps After Deployment

1. **User Testing**: Invite beta users to test
2. **Monitor Performance**: Watch logs for 48 hours
3. **Scale if Needed**: Upgrade Railway plan for more resources
4. **Set Up Alerts**: Configure Railway notifications
5. **Documentation**: Share bot commands with users
6. **Backup Strategy**: Export trained models periodically

## Support Resources

- **Railway Docs**: https://docs.railway.app
- **Project Docs**: `docs/bot-deployment-server.md`
- **ML Engine Docs**: `docs/ML_ENGINE_SUMMARY.md`
- **Cloud Training**: `docs/CLOUD_TRAINING_GUIDE.md`
- **API Reference**: `docs/API_REFERENCE.md`

## Railway CLI (Optional)

Install for command-line management:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link to project
railway link

# View logs
railway logs

# Set environment variable
railway variables set TELEGRAM_BOT_TOKEN=your_token

# Deploy manually
railway up
```

## Rollback Procedure

If deployment fails:

1. Go to Railway **Deployments** tab
2. Find last successful deployment
3. Click **"⋯"** → **"Redeploy"**
4. Fix issue locally
5. Push fix to GitHub
6. Railway auto-deploys new version

## Migration from Local to Railway

If you have existing local models:

1. **Backup Models**:
   ```bash
   tar -czf models_backup.tar.gz models/
   ```

2. **Upload to Railway**:
   - Use Railway CLI: `railway files upload models/ /app/models/`
   - Or: Retrain models after deployment

3. **Verify Migration**:
   - List models with `/models` command
   - Test predictions with uploaded data

---

**Deployment Status**: ✅ Ready for Railway deployment

**Last Updated**: 2025-11-15
