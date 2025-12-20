#!/usr/bin/env python3
"""Demo script to verify auto-start functionality.

This script demonstrates the auto-start configuration generation
for all supported platforms without actually installing anything.
"""

import sys
from pathlib import Path

# Add worker directory to path
worker_dir = Path(__file__).parent.parent / "worker"
sys.path.insert(0, str(worker_dir))

from autostart import (
    detect_platform,
    generate_launchd_plist,
    generate_systemd_service,
    generate_task_scheduler_xml,
)


def main():
    """Run auto-start demo."""
    print("=" * 70)
    print("StatsBot Worker Auto-start Configuration Demo")
    print("=" * 70)
    print()

    # Detect platform
    platform = detect_platform()
    print(f"Detected Platform: {platform.upper()}")
    print()

    # Sample configuration
    config = {
        "ws_url": "wss://statsbot.example.com/ws",
        "token": "demo-token-abc123xyz",
        "machine_id": "demo-machine",
    }

    script_path = "/home/user/statsbot_worker.py"

    print("Sample Configuration:")
    print(f"  WebSocket URL: {config['ws_url']}")
    print(f"  Token: {config['token']}")
    print(f"  Machine ID: {config['machine_id']}")
    print(f"  Script Path: {script_path}")
    print()

    # Generate configurations for all platforms
    print("-" * 70)
    print("Mac (launchd) Configuration")
    print("-" * 70)
    plist = generate_launchd_plist(script_path, config)
    print(plist)
    print()

    print("-" * 70)
    print("Linux (systemd) Configuration")
    print("-" * 70)
    service = generate_systemd_service(script_path, config)
    print(service)
    print()

    print("-" * 70)
    print("Windows (Task Scheduler) Configuration")
    print("-" * 70)
    xml = generate_task_scheduler_xml(script_path, config)
    print(xml)
    print()

    print("=" * 70)
    print("✅ All auto-start configurations generated successfully!")
    print("=" * 70)
    print()
    print("Usage:")
    print("  Install:  python3 statsbot_worker.py --token=YOUR_TOKEN --autostart")
    print("  Remove:   python3 statsbot_worker.py --autostart off")
    print()


if __name__ == "__main__":
    main()
