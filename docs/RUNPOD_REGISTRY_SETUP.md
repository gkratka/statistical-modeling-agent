# RunPod Container Registry Setup Guide

**Goal**: Eliminate 5-8 minute Docker image downloads on every training run by using RunPod's container registry for instant pod startup.

**Time to Complete**: 15-20 minutes (one-time setup)
**Savings**: ~5-8 minutes per training run + $0.02-0.04 GPU cost savings
**ROI**: Breaks even after 3-4 training runs

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup](#step-by-step-setup)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Troubleshooting](#troubleshooting)
7. [Cost Analysis](#cost-analysis)
8. [Maintenance](#maintenance)

---

## Overview

### Problem
RunPod downloads the TensorFlow Docker image (~5-8GB) on every training run, adding:
- **5-8 minutes** to startup time
- **$0.02-0.04** in GPU costs per run
- **Risk of failure** due to Docker Hub rate limits

### Solution
Push your Docker image to RunPod's container registry **once**. All future pods pull instantly from RunPod's infrastructure.

### How RunPod Templates Work
**Important**: RunPod templates do **NOT** cache Docker images. They only store configuration (environment variables, volumes, docker_args). Even with a template, Docker images are downloaded on every pod creation.

To eliminate downloads, you must use RunPod Container Registry.

---

## Prerequisites

- **RunPod Account**: https://www.runpod.io/console/signup
- **RunPod API Key**: https://www.runpod.io/console/user/settings → API Keys
- **Docker Installed**: `docker --version` should work
- **Docker Running**: `docker info` should show system info
- **Git Repository**: Clone statistical-modeling-agent repository

---

## Step-by-Step Setup

### Step 1: Verify Docker Installation

```bash
# Check Docker is installed
docker --version
# Output: Docker version 20.10.x or higher

# Check Docker is running
docker info
# Should show system info, not errors
```

If Docker is not installed:
- **Mac**: Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
- **Linux**: `curl -fsSL https://get.docker.com | sh`
- **Windows**: Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)

### Step 2: Set RunPod API Key

```bash
# Export RunPod API key (get from https://www.runpod.io/console/user/settings)
export RUNPOD_API_KEY="your_runpod_api_key_here"

# Verify it's set
echo $RUNPOD_API_KEY
```

**Make it permanent** (add to `~/.bashrc` or `~/.zshrc`):
```bash
echo 'export RUNPOD_API_KEY="your_runpod_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Navigate to Project Directory

```bash
cd /path/to/statistical-modeling-agent
```

### Step 4: Build and Push to Registry

Run the automated push script:

```bash
./scripts/push_to_runpod_registry.sh YOUR_USERNAME v1.0.0
```

Replace:
- `YOUR_USERNAME`: Your RunPod username (from https://www.runpod.io/console/user/settings)
- `v1.0.0`: Version tag (use semantic versioning: v1.0.0, v1.1.0, etc.)

**What the script does**:
1. Builds Docker image from `docker/ml-training.Dockerfile` (~10 min)
2. Tags image for RunPod registry
3. Logs in to RunPod registry using your API key
4. Pushes image to `registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0` (~8 min for first push)

**Expected Output**:
```
[INFO] RunPod Registry Push Script
----------------------------------------
Username:     YOUR_USERNAME
Image Name:   ml-training
Version:      v1.0.0
Full Image:   registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0
----------------------------------------
[INFO] Step 1/4: Building Docker image...
[INFO] Step 2/4: Tagging image for RunPod registry...
[INFO] Step 3/4: Logging in to RunPod registry...
[INFO] Step 4/4: Pushing image to RunPod registry...
[WARN] This may take 5-10 minutes for first push (~8GB image)...
[INFO] ✅ Successfully pushed to RunPod registry!
```

---

## Configuration

### Step 5: Update Environment Variables

Edit `.env` file and add:

```bash
# RunPod Template (optional)
RUNPOD_TEMPLATE_ID=v3ie98gf2m

# RunPod Container Registry (REQUIRED for instant startup)
RUNPOD_REGISTRY_USERNAME=YOUR_USERNAME
RUNPOD_REGISTRY_PASSWORD=${RUNPOD_API_KEY}  # Use API key as password
```

### Step 6: Update Pod Manager to Use Registry Image

Edit `src/cloud/runpod_pod_manager.py`:

Find the line defining `docker_image` (around line 320-330):

```python
# BEFORE (Docker Hub - downloads every time):
docker_image = "tensorflow/tensorflow:2.13.0-gpu"

# AFTER (RunPod Registry - instant pulls):
docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0"
```

Replace `YOUR_USERNAME` with your actual RunPod username.

### Step 7: Restart the Bot

```bash
# Kill existing bot process
pkill -f telegram_bot.py

# Restart bot
python3 src/bot/telegram_bot.py
```

---

## Usage

### Training with Registry Image

Start a training run normally via Telegram:

```
/train
→ Choose Cloud Training
→ Upload dataset or provide path
→ Select model and hyperparameters
→ Confirm and start training
```

**Expected Behavior**:
- ✅ **Instant startup**: Pod starts in ~30 seconds (vs 5-8 minutes)
- ✅ **No download logs**: Training begins immediately
- ✅ **Cost savings**: ~$0.02-0.04 saved per run

### Verifying Registry Usage

Check pod logs in RunPod console or Telegram bot output:

```
# With Registry (GOOD):
2025-11-11 10:00:00 - Pod created: abc123xyz
2025-11-11 10:00:30 - Training started
2025-11-11 10:05:30 - Training complete

# Without Registry (BAD):
2025-11-11 10:00:00 - Pod created: abc123xyz
2025-11-11 10:00:05 - Pulling image tensorflow/tensorflow:2.13.0-gpu
2025-11-11 10:00:10 - Downloading layer 1/42 [=====>  ] 234MB/5.1GB
2025-11-11 10:05:45 - Image pull complete
2025-11-11 10:05:50 - Training started
```

---

## Troubleshooting

### Issue 1: Docker Build Fails

**Error**: `docker: command not found` or `Cannot connect to Docker daemon`

**Solution**:
```bash
# Check Docker is installed
docker --version

# Check Docker is running
docker info

# Start Docker Desktop (Mac/Windows)
open -a Docker  # Mac
# or start Docker Desktop manually

# Start Docker daemon (Linux)
sudo systemctl start docker
```

### Issue 2: Docker Login Fails

**Error**: `Error response from daemon: Get https://registry.runpod.io/v2/: unauthorized`

**Solution**:
```bash
# Verify API key is set
echo $RUNPOD_API_KEY

# Re-export with correct key
export RUNPOD_API_KEY="your_correct_api_key"

# Verify on RunPod console
# https://www.runpod.io/console/user/settings → API Keys

# Try login manually
echo $RUNPOD_API_KEY | docker login registry.runpod.io -u YOUR_USERNAME --password-stdin
```

### Issue 3: Push Fails with Timeout

**Error**: `EOF` or `context deadline exceeded`

**Solution**:
```bash
# Check network connection
ping registry.runpod.io

# Retry push (Docker resumes from last layer)
docker push registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0

# If repeatedly fails, push to Docker Hub first, then pull to RunPod
docker tag ml-training:v1.0.0 YOUR_USERNAME/ml-training:v1.0.0
docker push YOUR_USERNAME/ml-training:v1.0.0  # Push to Docker Hub
# Then pull to RunPod from Docker Hub (faster internal transfer)
```

### Issue 4: Pods Still Downloading Image

**Error**: Pods show "Pulling image..." in logs despite registry setup

**Solution**:
```bash
# 1. Verify docker_image in runpod_pod_manager.py uses registry URL
grep "docker_image" src/cloud/runpod_pod_manager.py
# Should show: docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0"

# 2. Verify .env has registry credentials
grep "RUNPOD_REGISTRY" .env
# Should show:
#   RUNPOD_REGISTRY_USERNAME=YOUR_USERNAME
#   RUNPOD_REGISTRY_PASSWORD=...

# 3. Restart bot to reload configuration
pkill -f telegram_bot.py
python3 src/bot/telegram_bot.py
```

### Issue 5: Image Not Found on Registry

**Error**: `Error: image not found: registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0`

**Solution**:
```bash
# List images in Docker
docker images | grep ml-training

# Verify image was pushed
docker pull registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0

# If pull fails, re-push
./scripts/push_to_runpod_registry.sh YOUR_USERNAME v1.0.0
```

---

## Cost Analysis

### Without Registry (Docker Hub)

**Per Training Run**:
- Download time: 5-8 minutes
- GPU cost during download: ~$0.02-0.04 (RTX A5000 at $0.29/hr)
- Risk: Docker Hub rate limits (100 pulls/6hrs for free accounts)

**Annual Cost (100 runs)**:
- Time wasted: 8-13 hours
- GPU cost: $2-4
- Opportunity cost: ~$40-130 (if your time is $10-20/hr)

### With Registry (RunPod Container Registry)

**One-Time Setup**:
- Build time: 10 minutes
- Push time: 8 minutes
- Configuration: 2 minutes
- **Total: ~20 minutes**

**Per Training Run**:
- Download time: **0 seconds** (instant pull)
- GPU cost savings: $0.02-0.04 per run
- **No rate limits**

**ROI Calculation**:
- Break-even: After 3-4 training runs
- Annual savings (100 runs): 8-13 hours + $2-4 + no rate limit failures

### Registry Storage Cost

RunPod Container Registry pricing (as of 2025-11):
- **Free tier**: 5GB storage, unlimited pulls
- **Pro tier**: $0.10/GB/month for storage over 5GB
- **ml-training image**: ~8GB = **$0.30/month** for storage

**Net savings**: ~$2-4/year GPU cost - $3.60/year storage = **Neutral to slight positive**, plus huge time savings

---

## Maintenance

### Updating the Image

When you update dependencies or code:

```bash
# Build and push new version
./scripts/push_to_runpod_registry.sh YOUR_USERNAME v1.1.0

# Update runpod_pod_manager.py
docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:v1.1.0"

# Restart bot
pkill -f telegram_bot.py
python3 src/bot/telegram_bot.py
```

### Version Management

Use semantic versioning for images:
- **v1.0.0**: Initial release
- **v1.1.0**: Add new dependencies (minor change)
- **v1.0.1**: Bug fixes (patch)
- **v2.0.0**: Major changes (breaking changes)

Keep multiple versions available:
```bash
# Production
docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0"

# Testing
docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:v1.1.0-beta"

# Latest (always points to newest)
docker_image = "registry.runpod.io/YOUR_USERNAME/ml-training:latest"
```

### Cleaning Up Old Images

Remove unused images from Docker:
```bash
# List all images
docker images | grep ml-training

# Remove specific version
docker rmi registry.runpod.io/YOUR_USERNAME/ml-training:v1.0.0

# Remove all unused images
docker image prune -a
```

Remove from RunPod registry:
- Log in to RunPod console: https://www.runpod.io/console/registry
- Navigate to YOUR_USERNAME/ml-training
- Delete old versions

---

## Advanced: Alternative Approaches

### Option 1: RunPod Pre-Cached Images (Free)

Use images RunPod maintains on their infrastructure:
```python
docker_image = "runpod/tensorflow:2.13.0-gpu"
```

**Pros**: Free, instant pulls
**Cons**: Limited to RunPod-maintained images, may not have exact versions

### Option 2: Accept Cold Starts (Current Default)

Continue using Docker Hub images:
```python
docker_image = "tensorflow/tensorflow:2.13.0-gpu"
```

**Pros**: No setup, maximum flexibility
**Cons**: 5-8 minute wait, $0.02-0.04 cost, rate limit risk

### Option 3: RunPod Container Registry (RECOMMENDED)

Use the setup described in this guide.

**Pros**: Instant pulls, custom images, no rate limits
**Cons**: One-time 20-minute setup, ~$0.30/month storage

---

## Summary

✅ **You've successfully set up RunPod Container Registry!**

**What you've accomplished**:
1. Built optimized ML training Docker image
2. Pushed to RunPod registry for instant access
3. Configured bot to use registry image
4. Eliminated 5-8 minute downloads on every training run

**Next steps**:
1. Start a training run via Telegram to verify instant startup
2. Monitor pod logs to confirm no downloads
3. Update image when adding new dependencies
4. Share this guide with team members

**Resources**:
- RunPod Console: https://www.runpod.io/console
- RunPod Docs: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- Project Issues: https://github.com/yourusername/statistical-modeling-agent/issues

---

**Questions?** Open an issue on GitHub or check CLAUDE.md for troubleshooting.
