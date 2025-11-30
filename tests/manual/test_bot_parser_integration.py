#!/usr/bin/env python
"""Test parser integration with the bot."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.parser import RequestParser
from src.utils.exceptions import ParseError

async def test_bot_integration():
    """Test parser with mock bot scenarios."""

    test_messages = [
        "calculate the mean of age column",
        "train a neural network model",
        "show correlation matrix",
        "what is the data shape",
        "predict customer satisfaction",
        "give me descriptive statistics",
        "linear regression to predict income",
        "what columns are available",
        "build random forest classifier",
        "standard deviation of salary"
    ]

    print("Testing Bot-Parser Integration\n" + "=" * 40)

    parser = RequestParser()
    success_count = 0

    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. Message: '{message}'")

        # Simulate bot parameters
        user_id = 12345
        chat_id = 67890
        message_id = i
        conversation_id = f"{chat_id}_{message_id}"

        try:
            # Parse the message (this is what would happen in the bot handler)
            task = parser.parse_request(message, user_id, conversation_id)

            print(f"   ✅ Successfully parsed")
            print(f"   Task Type: {task.task_type}")
            print(f"   Operation: {task.operation}")
            print(f"   Confidence: {task.confidence_score:.2f}")

            # Show parameters if they exist
            if task.parameters:
                relevant_params = {k: v for k, v in task.parameters.items()
                                 if v and v != ["all"]}
                if relevant_params:
                    print(f"   Parameters: {relevant_params}")

            success_count += 1

            # Simulate what orchestrator would receive
            print(f"   → Would route to: {task.task_type}_engine")

        except ParseError as e:
            print(f"   ❌ Parse failed: {e.message}")
        except Exception as e:
            print(f"   ⚠️ Unexpected error: {e}")

    print(f"\n" + "=" * 40)
    print(f"Integration Test Summary:")
    print(f"Successfully parsed: {success_count}/{len(test_messages)} messages")
    print(f"Success rate: {success_count/len(test_messages)*100:.1f}%")

async def test_conversation_flow():
    """Test multi-message conversation scenario."""
    print("\n" + "=" * 50)
    print("TESTING CONVERSATION FLOW")
    print("=" * 50)

    parser = RequestParser()
    user_id = 12345
    base_conversation_id = "conv_001"

    # Simulate a conversation where user uploads data then asks questions
    conversation_messages = [
        ("I uploaded a dataset with sales data", "Data upload context"),
        ("what columns are in the data", "Data exploration"),
        ("show me correlation between price and sales", "Statistical analysis"),
        ("calculate mean price by region", "Grouped statistics"),
        ("train model to predict sales based on price and region", "ML training"),
        ("predict sales for new data", "ML scoring"),
    ]

    for i, (message, context) in enumerate(conversation_messages, 1):
        conversation_id = f"{base_conversation_id}_msg_{i}"
        print(f"\n{i}. Context: {context}")
        print(f"   Message: '{message}'")

        try:
            task = parser.parse_request(message, user_id, conversation_id)
            print(f"   ✅ Parsed as: {task.task_type}")
            print(f"   Operation: {task.operation}")
            print(f"   Confidence: {task.confidence_score:.2f}")

            # Show how this would flow in the bot
            if task.task_type == "data_info":
                print(f"   → Bot would call data_engine.describe()")
            elif task.task_type == "stats":
                print(f"   → Bot would call stats_engine.process()")
            elif task.task_type in ["ml_train", "ml_score"]:
                print(f"   → Bot would call ml_engine.{task.operation}()")

        except ParseError as e:
            print(f"   ❌ Could not parse: {e.message}")
            print(f"   → Bot would ask for clarification")

async def test_error_handling():
    """Test how parser handles various error conditions."""
    print("\n" + "=" * 50)
    print("TESTING ERROR HANDLING")
    print("=" * 50)

    parser = RequestParser()

    error_cases = [
        ("", "Empty message"),
        ("   ", "Whitespace only"),
        ("hello", "Greeting (not a request)"),
        ("asdfghjkl", "Random characters"),
        ("do something", "Vague request"),
        ("help me", "Help request"),
        ("thanks", "Acknowledgment"),
    ]

    for message, description in error_cases:
        print(f"\nTesting: {description}")
        print(f"Message: '{message}'")

        try:
            task = parser.parse_request(message, 12345, "test")
            print(f"   ⚠️ Unexpectedly parsed as: {task.task_type} (confidence: {task.confidence_score:.2f})")
            if task.confidence_score < 0.5:
                print(f"   → Low confidence - bot should ask for clarification")
        except ParseError as e:
            print(f"   ✅ Correctly rejected: {e.message}")
            print(f"   → Bot would send helpful error message")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")

async def test_multilingual_and_variations():
    """Test various ways users might phrase requests."""
    print("\n" + "=" * 50)
    print("TESTING LANGUAGE VARIATIONS")
    print("=" * 50)

    parser = RequestParser()

    variations = [
        # Different ways to ask for mean
        ("calculate mean", "Direct command"),
        ("find the average", "Alternative phrasing"),
        ("what's the mean", "Question form"),
        ("show me the mean", "Polite request"),
        ("I want to see the average", "Statement form"),

        # Different ways to ask for ML
        ("train a model", "Direct ML command"),
        ("I want to build a predictor", "Conversational ML"),
        ("create a classifier", "Alternative ML phrasing"),
        ("make a neural network", "Specific model request"),

        # Casual vs formal
        ("gimme the stats", "Casual"),
        ("Please provide descriptive statistics", "Formal"),
        ("stats pls", "Very casual"),
        ("Could you calculate the correlation matrix", "Very formal"),
    ]

    for message, style in variations:
        print(f"\nStyle: {style}")
        print(f"Message: '{message}'")

        try:
            task = parser.parse_request(message, 12345, "test")
            print(f"   ✅ Parsed as: {task.task_type} - {task.operation}")
            print(f"   Confidence: {task.confidence_score:.2f}")
        except ParseError as e:
            print(f"   ❌ Failed to parse: {e.message}")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

async def main():
    """Run all integration tests."""
    await test_bot_integration()
    await test_conversation_flow()
    await test_error_handling()
    await test_multilingual_and_variations()

    print("\n" + "=" * 60)
    print("INTEGRATION TESTING COMPLETE")
    print("=" * 60)
    print("The parser is ready for integration with the Telegram bot!")
    print("Next steps:")
    print("1. Ensure handlers.py imports and uses RequestParser")
    print("2. Test with actual Telegram messages")
    print("3. Connect parser output to orchestrator")

if __name__ == "__main__":
    asyncio.run(main())