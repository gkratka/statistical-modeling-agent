# PRD: Local Worker - Hybrid Architecture for Local ML Execution

## 1. Introduction/Overview

The Local Worker feature enables users to run ML training and prediction on their own machines while using the Telegram bot interface hosted on Railway. This solves the problem of users wanting to:
- Train models on large local datasets (no upload size limits)
- Keep sensitive data on their own machines
- Leverage local compute resources

**Architecture**: Bot runs on Railway (always online), ML Worker runs on user's machine (executes jobs locally).

## 2. Goals

1. Enable users to train/predict using local files without uploading to the cloud
2. Provide seamless UX where all interaction happens through Telegram
3. Require minimal user effort (single command to start worker)
4. Support auto-start persistence so users don't need to restart worker each session
5. Maintain full ML workflow capabilities (training, prediction, model management)

## 3. User Stories

### US1: First-time Setup
**As a** new user
**I want to** connect my local machine to the bot with a single command
**So that** I can start training models on my local data immediately

**Acceptance Criteria:**
- User sends `/start` or `/connect` in Telegram
- Bot displays a curl command with embedded one-time token
- User copies command to terminal and runs it
- Worker connects and bot confirms "Local worker connected!"
- User can now select "Local Path" option in `/train` workflow

### US2: Training with Local Data
**As a** user with large datasets
**I want to** train models using files on my computer
**So that** I don't have to upload large files through Telegram

**Acceptance Criteria:**
- User selects "Local Path" in `/train` workflow
- User enters file path (e.g., `/Users/me/data/dataset.csv`)
- Worker reads the file locally and executes training
- Training results appear in Telegram
- Model is saved locally on user's machine

### US3: Auto-start Worker
**As a** frequent user
**I want to** have the worker start automatically when I log in
**So that** I don't need to run the command each time

**Acceptance Criteria:**
- User sends `/worker autostart` in Telegram
- Bot provides instructions for Mac/Linux/Windows
- Worker runs in background on system startup
- User can disable with `/worker autostart off`

### US4: Prediction with Local Models
**As a** user with trained models
**I want to** make predictions using my local models and data
**So that** I can use the full ML workflow locally

**Acceptance Criteria:**
- User selects "Local Path" in `/predict` workflow
- User can select from locally stored models
- Prediction runs on user's machine
- Results displayed in Telegram

## 4. Functional Requirements

### 4.1 Bot-Side (Railway)

| ID | Requirement |
|----|-------------|
| FR1 | Bot MUST expose a WebSocket endpoint for worker connections |
| FR2 | Bot MUST generate one-time authentication tokens (valid for 5 minutes) |
| FR3 | Bot MUST associate connected workers with Telegram user IDs |
| FR4 | Bot MUST queue ML jobs and send them to connected workers |
| FR5 | Bot MUST receive job results from workers and display in Telegram |
| FR6 | Bot MUST detect worker disconnection and notify user |
| FR7 | Bot MUST show worker connection status in `/start` response |
| FR8 | `/connect` command MUST display the curl command with token |
| FR9 | Bot MUST serve the worker script at `/worker` endpoint |

### 4.2 Worker-Side (User's Machine)

| ID | Requirement |
|----|-------------|
| FR10 | Worker MUST be a single Python script (no pip install required) |
| FR11 | Worker MUST connect to bot via WebSocket using provided token |
| FR12 | Worker MUST authenticate with Telegram user ID + token |
| FR13 | Worker MUST listen for job commands (train, predict, list_models, etc.) |
| FR14 | Worker MUST execute ML operations using local files |
| FR15 | Worker MUST send job results back to bot |
| FR16 | Worker MUST handle reconnection if connection drops |
| FR17 | Worker MUST support `--autostart` flag to install as startup service |
| FR18 | Worker MUST store models in local `~/.statsbot/models/` directory |
| FR19 | Worker MUST validate file paths before accessing |

### 4.3 Authentication Flow

| ID | Requirement |
|----|-------------|
| FR20 | User sends `/connect` in Telegram |
| FR21 | Bot generates one-time token (UUID, expires in 5 min) |
| FR22 | Bot displays: `curl -sL https://bot.railway.app/worker \| python3 - --token=<TOKEN>` |
| FR23 | Worker sends token + machine identifier to bot |
| FR24 | Bot validates token and links worker to Telegram user |
| FR25 | Token is invalidated after successful use |

### 4.4 Job Protocol (WebSocket Messages)

```json
// Bot -> Worker: Job request
{
  "type": "job",
  "job_id": "uuid",
  "action": "train",
  "params": {
    "file_path": "/path/to/data.csv",
    "target_column": "price",
    "model_type": "random_forest"
  }
}

// Worker -> Bot: Job progress
{
  "type": "progress",
  "job_id": "uuid",
  "status": "training",
  "progress": 45,
  "message": "Training model..."
}

// Worker -> Bot: Job result
{
  "type": "result",
  "job_id": "uuid",
  "success": true,
  "data": {
    "model_id": "model_123",
    "metrics": {"r2": 0.85, "mse": 0.12}
  }
}
```

### 4.5 /start Workflow Integration

| ID | Requirement |
|----|-------------|
| FR26 | `/start` MUST show worker connection status |
| FR27 | If no worker connected, show "Connect local worker" button |
| FR28 | Button triggers `/connect` command flow |
| FR29 | If worker connected, show "Worker: Connected (machine-name)" |

## 5. Non-Goals (Out of Scope)

1. **Cloud ML execution** - This feature is for local execution only; cloud training via RunPod is a separate feature
2. **Multi-worker support** - One worker per user for MVP
3. **Worker-to-worker communication** - Workers don't talk to each other
4. **Windows GUI installer** - Command-line only for MVP
5. **Model syncing between machines** - Models stay on the machine where they were trained
6. **Web dashboard** - All UI is through Telegram

## 6. Design Considerations

### User Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Telegram  │────▶│   Railway   │◀────│   Worker    │
│   (User)    │     │   (Bot)     │     │  (Local)    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      │ /connect           │                    │
      │───────────────────▶│                    │
      │                    │                    │
      │ curl command       │                    │
      │◀───────────────────│                    │
      │                    │                    │
      │         User runs curl in terminal      │
      │                    │◀───────────────────│
      │                    │   WebSocket conn   │
      │                    │   + token auth     │
      │                    │                    │
      │ "Worker connected!"│                    │
      │◀───────────────────│                    │
      │                    │                    │
      │ /train             │                    │
      │───────────────────▶│                    │
      │                    │   Job: train       │
      │                    │───────────────────▶│
      │                    │                    │ (local execution)
      │                    │   Result           │
      │                    │◀───────────────────│
      │ Training complete! │                    │
      │◀───────────────────│                    │
```

### UI Messages

**Connect prompt (in /start):**
```
🖥️ Local Worker: Not connected

To train models using files on your computer, connect a local worker:
👉 /connect
```

**Connect command response:**
```
🔗 Connect Local Worker

Run this command in your terminal:

curl -sL https://statsbot.railway.app/worker | python3 - --token=abc123

⏱️ Token expires in 5 minutes

Requirements:
• Python 3.8+
• Mac or Linux (Windows: use WSL)
```

**Worker connected:**
```
✅ Local Worker Connected!

Machine: MacBook-Pro.local
Status: Ready

You can now use local file paths in /train and /predict.
```

## 7. Technical Considerations

### Dependencies

**Bot-side (add to requirements.txt):**
- `websockets` - WebSocket server
- `aiohttp` - HTTP endpoint for worker script

**Worker script (self-contained):**
- Uses only Python standard library + packages already in user's env
- Falls back gracefully if ML packages missing

### Security

1. **Token expiration** - Tokens expire after 5 minutes
2. **One-time use** - Tokens invalidated after successful auth
3. **Path validation** - Worker validates file paths (no traversal attacks)
4. **Local-only models** - Models stored locally, not uploaded to Railway

### Auto-start Implementation

**Mac (launchd):**
```bash
# Creates ~/Library/LaunchAgents/com.statsbot.worker.plist
python3 worker.py --autostart
```

**Linux (systemd user service):**
```bash
# Creates ~/.config/systemd/user/statsbot-worker.service
python3 worker.py --autostart
```

### File Structure

```
~/.statsbot/
├── config.json          # Worker config (token, bot URL)
├── models/              # Locally trained models
│   └── user_123/
│       └── model_abc/
└── logs/
    └── worker.log
```

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Worker connection success rate | > 95% |
| Time from `/connect` to worker ready | < 30 seconds |
| Job execution success rate | > 90% |
| User adoption of local worker feature | > 30% of active users |

## 9. Open Questions

1. **Q: Should we support Windows natively or require WSL?**
   - Recommendation: WSL for MVP, native Windows later

2. **Q: How to handle worker updates?**
   - Recommendation: Worker checks version on connect, prompts user to re-run curl if outdated

3. **Q: Should models be portable between machines?**
   - Recommendation: No for MVP; models tied to machine where trained

4. **Q: Rate limiting on WebSocket connections?**
   - Recommendation: Max 1 worker per user, reconnect cooldown of 5 seconds

---

*PRD Version: 1.0*
*Created: 2025-11-30*
*Status: Draft*
