#!/usr/bin/env python3
"""
Manual test script for language detection in /start command.

Usage:
    python3 scripts/test_language_detection.py

Test scenarios:
1. Send "/start" → Should see English (default)
2. Send "Olá, preciso de ajuda" → Language detected as Portuguese
3. Send "/start" again → Should see Portuguese (if translations loaded)
4. Send "Hello, I need help" → Language switches to English
5. Send "/start" again → Should see English

Requirements:
- Bot must be running (python src/bot/telegram_bot.py)
- TELEGRAM_BOT_TOKEN in .env
- locales/pt.yaml with Portuguese translations (optional)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.language_detector import LanguageDetector


async def test_language_detector():
    """Test language detector with various inputs."""
    detector = LanguageDetector()

    test_cases = [
        ("Hello, I need help with my data", "en"),
        ("Olá, preciso de ajuda com análise", "pt"),
        ("Como usar este bot?", "pt"),
        ("What is the meaning of this?", "en"),
        ("Quero treinar um modelo", "pt"),
        ("/start", "en"),  # Short command, should default
    ]

    print("Testing LanguageDetector:\n")
    for text, expected_lang in test_cases:
        result = await detector.detect_language(text, user_id=12345)
        status = "✅" if result.language == expected_lang else "❌"
        print(f"{status} '{text[:40]}...'")
        print(f"   Detected: {result.language} (expected: {expected_lang})")
        print(f"   Confidence: {result.confidence:.2f}, Method: {result.method}\n")


def check_translations():
    """Check if translation files exist."""
    print("\nChecking translation files:\n")

    locales_dir = Path(__file__).parent.parent / "locales"
    if not locales_dir.exists():
        print("❌ locales/ directory not found")
        return

    for lang_code in ["en", "pt"]:
        yaml_file = locales_dir / f"{lang_code}.yaml"
        if yaml_file.exists():
            print(f"✅ {lang_code}.yaml found")
        else:
            print(f"❌ {lang_code}.yaml missing")


def print_manual_test_instructions():
    """Print instructions for manual testing with Telegram bot."""
    print("\n" + "="*60)
    print("MANUAL TEST INSTRUCTIONS")
    print("="*60)
    print("""
1. Start the bot:
   $ python3 src/bot/telegram_bot.py

2. In Telegram, test these scenarios:

   A. English (default):
      → Send: /start
      → Expected: Welcome message in English

   B. Portuguese detection:
      → Send: Olá, preciso de ajuda
      → Expected: Bot detects Portuguese
      → Send: /start
      → Expected: Welcome message in Portuguese (if pt.yaml exists)

   C. Language switch:
      → Send: Hello, I need help
      → Expected: Bot detects English
      → Send: /start
      → Expected: Welcome message in English

3. Check logs for language detection:
   Look for: "Language detected: user=..., lang=..., confidence=..."

4. Verify session persistence:
   → Language should persist across commands
   → No re-detection on every command (uses cached language)
""")


if __name__ == "__main__":
    print("Language Detection Test Script")
    print("="*60 + "\n")

    # Test detector
    asyncio.run(test_language_detector())

    # Check translations
    check_translations()

    # Print manual test instructions
    print_manual_test_instructions()
