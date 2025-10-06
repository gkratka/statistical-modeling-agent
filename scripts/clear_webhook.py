#!/usr/bin/env python3
"""
Clear Telegram webhook and pending updates.
This resolves the 'Conflict: terminated by other getUpdates request' error.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import requests

# Load environment
env_path = project_root / ".env"
load_dotenv(env_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    print("Error: TELEGRAM_BOT_TOKEN not found in .env")
    sys.exit(1)

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

print("Clearing Telegram webhook and pending updates...")

# Delete webhook
response = requests.post(f"{BASE_URL}/deleteWebhook", json={"drop_pending_updates": True})
print(f"Delete webhook: {response.json()}")

# Get webhook info to verify
response = requests.get(f"{BASE_URL}/getWebhookInfo")
webhook_info = response.json()
print(f"Webhook info: {webhook_info}")

# Clear any pending updates by calling getUpdates with a high offset
response = requests.post(f"{BASE_URL}/getUpdates", json={"offset": -1, "timeout": 1})
updates = response.json()
print(f"Cleared pending updates: {len(updates.get('result', []))} updates")

print("\n✓ Webhook cleared and bot ready for polling!")
