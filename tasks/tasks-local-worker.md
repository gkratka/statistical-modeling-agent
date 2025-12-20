# Tasks: Local Worker - Hybrid Architecture Implementation

## Relevant Files

### New Files to Create
- `src/worker/websocket_server.py` - WebSocket server for worker connections (FR1, FR4-6, FR9)
- `src/worker/token_manager.py` - One-time token generation and validation (FR2, FR21, FR24-25)
- `src/worker/job_queue.py` - Job queuing and dispatching to workers (FR4-5)
- `src/worker/worker_manager.py` - Connected worker tracking per user (FR3, FR6)
- `src/bot/handlers/connect_handler.py` - /connect command handler (FR8, FR20, FR22)
- `worker/statsbot_worker.py` - Self-contained worker script served to users (FR10-19)
- `tests/unit/test_websocket_server.py` - WebSocket server tests
- `tests/unit/test_token_manager.py` - Token management tests
- `tests/unit/test_job_queue.py` - Job queue tests
- `tests/unit/test_worker_manager.py` - Worker manager tests
- `tests/integration/test_local_worker.py` - End-to-end worker tests

### Files to Modify
- `src/bot/telegram_bot.py` - Add WebSocket server startup alongside Telegram bot
- `src/bot/handlers.py` - Register /connect handler and update /start
- `requirements.txt` - Add websockets, aiohttp dependencies
- `config/config.yaml` - Add worker configuration section

### Notes
- Unit tests should be placed in `tests/unit/` directory
- Integration tests should be placed in `tests/integration/` directory
- Use `pytest tests/` to run all tests
- Worker script must be self-contained (no pip install required)

## Instructions for Completing Tasks

**IMPORTANT:** As you complete each task, check it off by changing `- [ ]` to `- [x]`. Update after completing each sub-task, not just parent tasks.

## Tasks

- [x] 0.0 Create feature branch
  - [x] 0.1 Create and checkout branch `feature/local-worker`

- [x] 1.0 Bot-side WebSocket Infrastructure (FR1, FR4-6, FR9)
  - [x] 1.1 Add `websockets` and `aiohttp` to requirements.txt
  - [x] 1.2 Create `src/worker/__init__.py` module
  - [x] 1.3 Create `src/worker/websocket_server.py` with WebSocket endpoint
  - [x] 1.4 Implement connection handling (accept, track, close)
  - [x] 1.5 Implement worker disconnection detection and notification
  - [x] 1.6 Add HTTP endpoint `/worker` to serve worker script
  - [x] 1.7 Integrate WebSocket server startup into `telegram_bot.py`
  - [x] 1.8 Write unit tests for `websocket_server.py`

- [x] 2.0 Authentication & Token Management (FR2-3, FR20-25)
  - [x] 2.1 Create `src/worker/token_manager.py`
  - [x] 2.2 Implement token generation (UUID, 5-minute expiry)
  - [x] 2.3 Implement token validation and one-time use invalidation
  - [x] 2.4 Implement user-to-token and token-to-user mapping
  - [x] 2.5 Create `src/worker/worker_manager.py` for connected worker tracking
  - [x] 2.6 Implement worker-to-user association on successful auth
  - [x] 2.7 Write unit tests for `token_manager.py`
  - [x] 2.8 Write unit tests for `worker_manager.py`

- [x] 3.0 Worker Script - Self-contained Python (FR10-19)
  - [x] 3.1 Create `worker/statsbot_worker.py` using only stdlib + common ML packages
  - [x] 3.2 Implement WebSocket connection with token authentication
  - [x] 3.3 Implement machine identifier detection (hostname)
  - [x] 3.4 Implement job listener (train, predict, list_models)
  - [x] 3.5 Implement local file path validation (security checks)
  - [x] 3.6 Implement ML training execution using local files
  - [x] 3.7 Implement prediction execution using local models
  - [x] 3.8 Implement result sending back to bot
  - [x] 3.9 Implement reconnection logic on connection drop
  - [x] 3.10 Implement local model storage in `~/.statsbot/models/`
  - [x] 3.11 Implement graceful fallback if ML packages missing
  - [x] 3.12 Write unit tests for worker script components

- [x] 4.0 Job Protocol Implementation (FR4-5, FR13-15)
  - [x] 4.1 Create `src/worker/job_queue.py`
  - [x] 4.2 Define JSON message schemas (job, progress, result)
  - [x] 4.3 Implement job creation and queuing
  - [x] 4.4 Implement job dispatch to connected worker
  - [x] 4.5 Implement progress message handling
  - [x] 4.6 Implement result message handling
  - [x] 4.7 Implement job timeout handling
  - [x] 4.8 Write unit tests for `job_queue.py`

- [x] 5.0 Telegram Handler Integration (FR7-8, FR26-29)
  - [x] 5.1 Create `src/bot/handlers/connect_handler.py`
  - [x] 5.2 Implement `/connect` command to generate token and display curl/irm command
  - [x] 5.3 Update `/start` handler to show worker connection status (FR7, FR26)
  - [x] 5.4 Add "Connect local worker" button when no worker connected (FR27-28)
  - [x] 5.5 Show "Worker: Connected (machine-name)" when connected (FR29)
  - [x] 5.6 Update `/train` workflow to route jobs to local worker when connected
  - [x] 5.7 Update `/predict` workflow to route jobs to local worker
  - [x] 5.8 Implement worker connection confirmation message to user
  - [x] 5.9 Implement worker disconnection notification to user
  - [x] 5.10 Register handlers in `handlers.py`

- [x] 6.0 Auto-start Support (Mac/Linux/Windows)
  - [x] 6.1 Implement `--autostart` flag parsing in worker script
  - [x] 6.2 Implement Mac launchd plist creation (`~/Library/LaunchAgents/`)
  - [x] 6.3 Implement Linux systemd user service creation (`~/.config/systemd/user/`)
  - [x] 6.4 Implement Windows Task Scheduler task creation
  - [x] 6.5 Implement `--autostart off` to remove auto-start
  - [x] 6.6 Add `/worker autostart` command in Telegram for instructions
  - [x] 6.7 Write tests for auto-start configuration generation

- [x] 7.0 Configuration & Documentation
  - [x] 7.1 Add worker configuration section to `config/config.yaml` (VERIFIED: already done)
  - [x] 7.2 Configure WebSocket port, token expiry, reconnect cooldown (VERIFIED: already done)
  - [x] 7.3 Create `~/.statsbot/` directory structure on worker first run (IMPLEMENTED: get_config_dir)
  - [x] 7.4 Write integration tests for full worker lifecycle (CREATED: tests/integration/test_local_worker.py - 15 tests passing)
  - [x] 7.5 Update CLAUDE.md with Local Worker architecture (ADDED: comprehensive architecture documentation)
  - [x] 7.6 Run all tests and verify passing (VERIFIED: all unit and integration tests passing)

---

*Generated from: docs/local-worker-prd.md v1.1*
*Created: 2025-11-30*
