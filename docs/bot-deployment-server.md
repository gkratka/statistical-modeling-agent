# Bot Deployment Guide: Making Your Telegram Bot Available to Others

This guide explains how to deploy the Statistical Modeling Agent Telegram bot to a server so it can run 24/7 and be accessible to other users.

## Table of Contents
1. [Understanding the Current Setup](#understanding-the-current-setup)
2. [Deployment Options](#deployment-options)
3. [Recommended: Railway.app Deployment](#recommended-railwayapp-deployment)
4. [Alternative: Render.com](#alternative-rendercom)
5. [Alternative: Self-Hosted Server](#alternative-self-hosted-server)
6. [Post-Deployment Configuration](#post-deployment-configuration)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Cost Comparison](#cost-comparison)

---

## Understanding the Current Setup

**Current State**: Your bot runs locally on your computer
- Start: `python src/bot/telegram_bot.py`
- Stop: When you close the terminal or shut down your computer
- **Problem**: Others can message the bot, but won't get responses when your computer is off

**What You Need**: A server that runs 24/7 to keep the bot online continuously

**Key Requirements**:
- Persistent storage for trained ML models (users expect models to persist)
- Environment variables (TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY, cloud credentials)
- Python 3.9+ runtime environment
- 2GB+ RAM for ML operations
- Long-running process support (bot uses polling, not webhooks)

---

## Deployment Options

| Platform | Monthly Cost | Difficulty | Best For |
|----------|-------------|------------|----------|
| **Railway.app** | $10-12 | Easy | Most users - git push to deploy, persistent volumes |
| **Render.com** | $7-10 | Easy | Budget option - similar to Railway |
| **Fly.io** | $5-8 | Medium | Global edge deployment |
| **DigitalOcean** | $12 | Medium | Full control, manual setup |
| **AWS Lightsail** | $10 | Medium | AWS ecosystem integration |
| **Heroku** | $25 | Easy | Not recommended - expensive |

**Recommendation**: **Railway.app** - Best balance of ease, features, and cost.

---

## Recommended: Railway.app Deployment

Railway.app is recommended because it offers:
- Git-based deployments (push to deploy)
- Built-in persistent volumes for model storage
- Automatic health checks and restarts
- No cold starts (unlike serverless platforms)
- Affordable pricing (~$10-12/month including storage)

### Prerequisites

1. **GitHub Account** - To store your code
2. **Railway Account** - Sign up at [railway.app](https://railway.app)
3. **Required Credentials**:
   - Telegram Bot Token (from @BotFather)
   - Anthropic API Key (from console.anthropic.com)
   - Cloud provider keys (AWS or RunPod) if using cloud training

### Step 1: Prepare Your Repository

#### 1.1 Create Railway Configuration Files

**Create `Procfile`** in the project root:
```
web: python src/bot/telegram_bot.py
```

**Create `railway.json`** in the project root:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
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

**Create `.railwayignore`** in the project root:
```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
*.egg-info/
.DS_Store
*.log
.bot.pid
catboost_info/
test_data/
tests/
docs/
scripts/*.sh
.git/
```

#### 1.2 Optimize Dependencies

**Create `requirements-railway.txt`**:
```txt
python-telegram-bot>=20.0
anthropic>=0.18.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
python-dotenv>=1.0.0
openpyxl>=3.1.0
tensorflow-cpu>=2.12.0,<2.16.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.2.0
boto3>=1.28.0
runpod>=1.0.0
```

**Note**: We use `tensorflow-cpu` (150MB) instead of `tensorflow` (500MB) to reduce deployment size and cost.

#### 1.3 Update Configuration for Railway

**Modify `config/config.yaml`**:
```yaml
# Update paths to use Railway volumes
local_data:
  enabled: false  # Disable local file paths for security
  allowed_directories:
    - /app/data
  max_file_size_mb: 1000

logging:
  level: INFO
  file: /app/data/logs/agent.log
  max_size: 10485760
  backup_count: 5

ml_engine:
  models_dir: /app/models
  max_models_per_user: 50
  max_model_size_mb: 100

data:
  temp_dir: /app/data/temp
  cache_dir: /app/data/cache
```

### Step 2: Push to GitHub

```bash
# Initialize git if not already done
git init
git add .
git commit -m "feat: add Railway deployment configuration"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/statistical-modeling-agent.git
git branch -M main
git push -u origin main
```

### Step 3: Deploy to Railway

#### 3.1 Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authorize GitHub and select your repository
5. Railway will automatically detect Python and start building

#### 3.2 Create Persistent Volumes

**Important**: Without volumes, your trained models will be deleted on every restart.

In Railway Dashboard:
1. Click on your project
2. Go to "Variables" tab
3. Click "New Volume"

Create 3 volumes:

| Volume Name | Mount Path | Size | Purpose |
|-------------|-----------|------|---------|
| `models-storage` | `/app/models` | 10GB | Trained ML models |
| `data-storage` | `/app/data` | 5GB | Logs, temp files, cache |
| `sessions-storage` | `/app/.sessions` | 1GB | User session state |

**To create each volume**:
- Click "New Volume"
- Name: (e.g., `models-storage`)
- Mount Path: (e.g., `/app/models`)
- Size: (e.g., 10GB)
- Click "Add"

#### 3.3 Configure Environment Variables

In Railway Dashboard → Variables → Add Variable:

**Required Variables (Minimum)**:
```bash
TELEGRAM_BOT_TOKEN=<your_bot_token_from_BotFather>
ANTHROPIC_API_KEY=<your_anthropic_key>
PYTHON_VERSION=3.9
```

**Optional: Cloud Training with RunPod**:
```bash
CLOUD_PROVIDER=runpod
RUNPOD_API_KEY=<your_runpod_key>
RUNPOD_NETWORK_VOLUME_ID=<volume_id>
RUNPOD_STORAGE_ACCESS_KEY=<access_key>
RUNPOD_STORAGE_SECRET_KEY=<secret_key>
MAX_TRAINING_COST_DOLLARS=10.0
```

**Optional: Cloud Training with AWS**:
```bash
CLOUD_PROVIDER=aws
AWS_ACCESS_KEY_ID=<aws_access_key>
AWS_SECRET_ACCESS_KEY=<aws_secret_key>
AWS_REGION=us-east-1
S3_BUCKET=ml-agent-data-<your_account_id>
EC2_INSTANCE_TYPE=m5.xlarge
```

**Optional: Local File Path Support**:
```bash
FILE_PATH_PASSWORD=<your_password>
```

### Step 4: Deploy and Monitor

1. **Trigger Deployment**: Railway deploys automatically when you push to GitHub
2. **Monitor Build Logs**: Railway Dashboard → Deployments → View Logs
3. **Check Status**: Should show "Running" after 3-5 minutes

**Expected Build Output**:
```
Installing dependencies from requirements-railway.txt...
Successfully installed tensorflow-cpu-2.15.0 scikit-learn-1.3.0...
Starting application...
```

**Expected Runtime Logs**:
```
Starting Statistical Modeling Agent bot...
Log level: INFO
Bot handlers configured successfully
Bot started successfully. Polling for messages...
✓ Polling started successfully
Bot is now running and processing messages...
```

### Step 5: Test Your Deployed Bot

1. Open Telegram and find your bot (@your_bot_name)
2. Send `/start`
3. Expected response: Welcome message with available features
4. Try uploading a CSV and requesting analysis
5. Check Railway logs for activity

### Step 6: Enable Auto-Deploy

Railway automatically deploys when you push to GitHub:

```bash
# Make changes to your code
git add .
git commit -m "feat: add new feature"
git push origin main
# Railway automatically builds and deploys
```

---

## Alternative: Render.com

Render is similar to Railway but slightly cheaper ($7/month vs $10-12/month).

### Quick Setup

1. **Create `render.yaml`** in project root:
```yaml
services:
  - type: web
    name: statistical-modeling-agent
    env: python
    plan: starter
    buildCommand: pip install -r requirements-railway.txt
    startCommand: python src/bot/telegram_bot.py
    envVars:
      - key: PYTHON_VERSION
        value: "3.9"
      - key: TELEGRAM_BOT_TOKEN
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
```

2. **Connect GitHub**: Go to [render.com](https://render.com) → New Web Service → Connect GitHub repo

3. **Add Persistent Disks**: Dashboard → Disks → New Disk
   - Name: `models`
   - Mount Path: `/app/models`
   - Size: 10GB

4. **Set Environment Variables**: Dashboard → Environment → Add

5. **Deploy**: Render auto-deploys on git push

---

## Alternative: Self-Hosted Server

For full control, deploy to a Linux server (DigitalOcean, AWS Lightsail, etc.).

### Prerequisites
- Ubuntu 22.04 LTS server
- Static IP address
- SSH access

### Deployment Steps

#### 1. Connect to Server
```bash
ssh ubuntu@your_server_ip
```

#### 2. Install Dependencies
```bash
sudo apt update
sudo apt install -y python3.9 python3-pip git
```

#### 3. Clone Repository
```bash
git clone https://github.com/yourusername/statistical-modeling-agent.git
cd statistical-modeling-agent
```

#### 4. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 5. Configure Environment
```bash
cp .env.example .env
nano .env
# Add your credentials:
# TELEGRAM_BOT_TOKEN=...
# ANTHROPIC_API_KEY=...
```

#### 6. Create Systemd Service

**Create `/etc/systemd/system/telegram-bot.service`**:
```ini
[Unit]
Description=Statistical Modeling Agent Telegram Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/statistical-modeling-agent
Environment="PATH=/home/ubuntu/statistical-modeling-agent/venv/bin"
ExecStart=/home/ubuntu/statistical-modeling-agent/venv/bin/python src/bot/telegram_bot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 7. Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable telegram-bot
sudo systemctl start telegram-bot
sudo systemctl status telegram-bot
```

#### 8. Monitor Logs
```bash
sudo journalctl -u telegram-bot -f
```

---

## Post-Deployment Configuration

### Security Considerations

1. **Never Commit Secrets**:
   - `.env` should be in `.gitignore` (already done)
   - Use Railway/Render environment variables for production

2. **Restrict Bot Access** (optional):
   - Modify `src/bot/telegram_bot.py` to whitelist user IDs
   - Add rate limiting for API calls

3. **Enable HTTPS** (Railway/Render provide this automatically)

### Sharing Your Bot

1. **Find Your Bot Username**: Check @BotFather messages
2. **Share the Link**: `https://t.me/your_bot_username`
3. **Users Start Bot**: They send `/start` to begin

### Managing Costs

#### Railway Cost Breakdown:
```
Base Hobby Plan:              $5.00/month
Volumes (16GB × 730 hours):   $5.40/month ($0.000463/GB-hour)
──────────────────────────────────────────
Total:                        ~$10-12/month
```

#### Cost Optimization Tips:
1. Use `tensorflow-cpu` instead of `tensorflow` (saves 350MB)
2. Implement log rotation (keep only 7 days)
3. Delete old models (>90 days inactive)
4. Use cloud providers (RunPod/AWS) for heavy training, not Railway

---

## Monitoring and Maintenance

### Railway Built-In Monitoring

Railway provides:
- CPU and memory usage graphs
- Network traffic metrics
- Deployment logs
- Crash notifications

**Access**: Railway Dashboard → Project → Metrics

### Health Checks (Optional)

Add a health check endpoint to verify bot is running:

**In `src/bot/telegram_bot.py`**, add:
```python
import os
from aiohttp import web

class StatisticalModelingBot:
    async def start_health_server(self):
        """Health check endpoint for Railway."""
        app = web.Application()
        app.router.add_get('/health', self.health_check)
        runner = web.AppRunner(app)
        await runner.setup()
        port = int(os.getenv('PORT', 8080))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        self.logger.info(f"Health check server started on port {port}")

    async def health_check(self, request):
        """Health check handler."""
        if self.application and self.application.updater.running:
            return web.Response(text="OK", status=200)
        return web.Response(text="Bot not running", status=503)
```

### External Monitoring with UptimeRobot (Free)

1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. Add New Monitor:
   - Type: HTTP(s)
   - URL: `https://your-app.railway.app/health`
   - Monitoring Interval: 5 minutes
3. Set up email/SMS alerts for downtime

### Log Management

**View Logs**:
- **Railway**: Dashboard → Deployments → Logs
- **Render**: Dashboard → Logs
- **Self-hosted**: `sudo journalctl -u telegram-bot -f`

**Log Rotation** (already configured in `config.yaml`):
```yaml
logging:
  max_size: 10485760  # 10MB
  backup_count: 5      # Keep 5 backup files
```

---

## Troubleshooting

### Issue 1: Bot Not Responding

**Symptom**: Users message the bot but get no response

**Diagnosis**:
```bash
# Check Railway logs
railway logs --follow
# or Render dashboard → Logs
```

**Common Causes**:
1. **Multiple bot instances running**:
   - Stop local bot: `killall -9 python3`
   - Wait 30 seconds
   - Redeploy: `railway up`

2. **Missing environment variables**:
   - Check Railway Dashboard → Variables
   - Verify `TELEGRAM_BOT_TOKEN` is set

3. **Bot crashed**:
   - Check logs for Python exceptions
   - Railway auto-restarts on failure

### Issue 2: Models Not Persisting

**Symptom**: Trained models disappear after Railway restarts

**Solution**:
1. Verify volumes are created: Railway Dashboard → Volumes
2. Check volume mount paths match `config.yaml`:
   ```yaml
   ml_engine:
     models_dir: /app/models  # Must match volume mount path
   ```
3. Verify write permissions in logs:
   ```
   # Should see in logs:
   Created model directory: /app/models/user_12345
   Saved model: /app/models/user_12345/model_xxx.pkl
   ```

### Issue 3: Out of Memory (OOM)

**Symptom**: Bot crashes with exit code 137 or "Killed"

**Solutions**:
1. **Upgrade Railway plan**: Hobby ($5) → Pro ($20) for more RAM
2. **Reduce concurrent sessions**:
   ```yaml
   # config.yaml
   session:
     max_concurrent_sessions: 10  # Reduce from 100
   ```
3. **Offload training to cloud**:
   ```yaml
   cloud:
     enabled: true
     provider: runpod  # Use external GPU pods
   ```

### Issue 4: Telegram Conflict Error

**Error**: `telegram.error.Conflict: terminated by other getUpdates request`

**Cause**: Bot running in multiple places (local + Railway)

**Solution**:
```bash
# Stop ALL local instances
killall -9 python3
pkill -f telegram_bot.py

# Wait 30 seconds for Telegram API to reset
sleep 30

# Redeploy on Railway only
railway up
```

### Issue 5: Build Timeout

**Error**: `Build exceeded 30 minute timeout`

**Cause**: Large `tensorflow` package download

**Solution**:
Use `tensorflow-cpu` in `requirements-railway.txt` (already recommended):
```txt
tensorflow-cpu>=2.12.0,<2.16.0  # 150MB vs 500MB
```

### Issue 6: Environment Variables Not Loading

**Symptom**: `ConfigurationError: Required environment variable X is not set`

**Solution**:
1. Railway Dashboard → Variables → Check all required vars
2. Verify no typos in variable names
3. Redeploy: `railway up` or push to GitHub
4. Wait for new deployment to finish

### Issue 7: Cloud Training Fails

**Symptom**: `/train` with cloud option fails

**Diagnosis**:
```bash
# Check Railway logs for specific error
railway logs | grep -i "runpod\|aws"
```

**Common Issues**:
1. **Missing cloud credentials**:
   - Add `RUNPOD_API_KEY` or AWS keys to Railway variables
2. **Invalid credentials**:
   - Regenerate keys at runpod.io or AWS console
3. **Network volume not found**:
   - Verify `RUNPOD_NETWORK_VOLUME_ID` is correct

---

## Cost Comparison

### Monthly Costs

| Platform | Base Cost | Volume Cost | Total | Notes |
|----------|-----------|-------------|-------|-------|
| **Railway** | $5 | $5-7 | **$10-12** | Recommended - easy setup |
| **Render** | $7 | Free (limited) | **$7-10** | Budget option |
| **Fly.io** | $0-5 | $3-5 | **$5-8** | Complex setup |
| **DigitalOcean** | $12 | Included | **$12** | Full control |
| **AWS Lightsail** | $10 | $2 | **$12** | AWS ecosystem |
| **Heroku** | $25 | Included | **$25** | Not recommended |

### Cost Breakdown: Railway (Most Common)

```
Base Hobby Plan:                    $5.00/month
Persistent Volumes:
  - models (10GB × 730h × $0.000463): $3.38/month
  - data (5GB × 730h × $0.000463):    $1.69/month
  - sessions (1GB × 730h × $0.000463): $0.34/month
Compute (24/7 running):              Included
Bandwidth (<100GB):                  Included
─────────────────────────────────────────────────
Total:                               ~$10.41/month
```

### Cost Optimization

1. **Reduce Volume Sizes** (if low usage):
   - models: 10GB → 5GB (saves $1.69/month)
   - data: 5GB → 2GB (saves $1.01/month)

2. **Use External Storage**:
   - Store models in AWS S3 ($0.023/GB/month)
   - 10GB = $0.23/month vs $3.38/month on Railway

3. **Implement Model Cleanup**:
   ```python
   # Delete models older than 90 days
   DELETE_OLD_MODELS = True
   MODEL_RETENTION_DAYS = 90
   ```

4. **Log Rotation** (already enabled):
   - Keeps only recent logs
   - Prevents unbounded growth

---

## Quick Reference

### Railway Deployment Checklist

- [ ] Create Railway account at railway.app
- [ ] Create `Procfile`, `railway.json`, `.railwayignore`
- [ ] Create `requirements-railway.txt` with `tensorflow-cpu`
- [ ] Update `config/config.yaml` paths to `/app/models`, `/app/data`
- [ ] Push code to GitHub
- [ ] Connect GitHub repo to Railway
- [ ] Create 3 persistent volumes (models, data, sessions)
- [ ] Add environment variables (TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY)
- [ ] Wait for deployment to complete (3-5 minutes)
- [ ] Test bot with `/start` command
- [ ] Set up monitoring (optional: UptimeRobot)

### Essential Commands

```bash
# Railway CLI
railway login
railway init
railway up
railway logs --follow
railway status

# Self-Hosted Server
sudo systemctl status telegram-bot
sudo systemctl restart telegram-bot
sudo journalctl -u telegram-bot -f

# Stop Local Bot (before deploying)
killall -9 python3
pkill -f telegram_bot.py
```

### Support Resources

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Telegram Bot API**: [core.telegram.org/bots](https://core.telegram.org/bots)
- **This Project**: [GitHub Issues](https://github.com/yourusername/statistical-modeling-agent/issues)

---

## Next Steps After Deployment

1. **Monitor for 24 hours**: Check Railway metrics and logs
2. **Test all features**: `/train`, `/predict`, `/models`, `/cloud_models`
3. **Set up backups**: Weekly backup of `/app/models` to S3 (optional)
4. **Configure alerts**: Railway notifications for crashes/OOM
5. **Share bot**: Give users your bot's Telegram link
6. **Document**: Update README.md with production URL

---

## Summary

**Simplest Option**: Railway.app
- Cost: ~$10-12/month
- Setup: 15 minutes
- Maintenance: Minimal (auto-restarts, health checks)
- Best for: Most users

**Budget Option**: Render.com
- Cost: ~$7-10/month
- Setup: 15 minutes
- Similar to Railway, slightly less polished UI

**Full Control**: Self-hosted server
- Cost: $10-12/month
- Setup: 30-60 minutes
- Requires Linux knowledge
- Best for: Advanced users

**Recommendation**: Start with Railway.app. It's the easiest to set up, maintain, and scale. You can always migrate later if needed.
