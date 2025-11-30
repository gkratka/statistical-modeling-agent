#!/usr/bin/env python
"""Interactive parser testing script."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.parser import RequestParser, DataSource
from src.utils.exceptions import ParseError

def test_parser_interactive():
    parser = RequestParser()
    user_id = 12345
    conversation_id = "test_conv"

    # Test cases with expected results
    test_cases = [
        # Statistical requests
        ("calculate mean for age column", "stats", "mean"),
        ("show correlation between age and income", "stats", "correlation"),
        ("give me descriptive statistics", "stats", "summary"),
        ("what's the standard deviation of salary", "stats", "std"),
        ("find median of customer_age", "stats", "median"),

        # ML requests
        ("train a model to predict income", "ml_train", "predict income"),
        ("build a random forest classifier", "ml_train", "random forest"),
        ("predict house prices using regression", "ml_score", "regression"),
        ("classify customers based on age and location", "ml_train", "classification"),
        ("train neural network model", "ml_train", "neural network"),

        # Data info requests
        ("show me the data", "data_info", "describe"),
        ("what columns are available", "data_info", "columns"),
        ("what is the shape of the dataset", "data_info", "shape"),

        # Edge cases
        ("", None, "should raise error"),
        ("random gibberish xyz", None, "should raise error"),
        ("do something", None, "should raise error or low confidence"),
    ]

    print("=" * 60)
    print("PARSER INTERACTIVE TEST")
    print("=" * 60)

    success_count = 0
    total_count = len(test_cases)

    for text, expected_type, description in test_cases:
        print(f"\nInput: '{text}'")
        print(f"Expected: {expected_type} - {description}")

        try:
            result = parser.parse_request(text, user_id, conversation_id)
            print(f"✅ Success!")
            print(f"  Type: {result.task_type}")
            print(f"  Operation: {result.operation}")
            print(f"  Parameters: {result.parameters}")
            print(f"  Confidence: {result.confidence_score:.2f}")

            if expected_type and result.task_type == expected_type:
                success_count += 1
                print(f"  ✓ Correct task type")
            elif expected_type:
                print(f"  ⚠️ Expected {expected_type}, got {result.task_type}")

        except ParseError as e:
            if expected_type is None:
                print(f"✅ Expected ParseError: {e.message}")
                success_count += 1
            else:
                print(f"❌ Unexpected ParseError: {e.message}")
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")

    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {success_count}/{total_count} tests behaved as expected")
    print(f"Success rate: {success_count/total_count*100:.1f}%")

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Type 'quit' to exit")
    print("Try your own requests to see how they're parsed!")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter request: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                print("Please enter a request (or 'quit' to exit)")
                continue

            result = parser.parse_request(user_input, user_id, conversation_id)
            print(f"✅ Parsed successfully!")
            print(f"  Task Type: {result.task_type}")
            print(f"  Operation: {result.operation}")
            print(f"  Parameters: {result.parameters}")
            print(f"  Confidence: {result.confidence_score:.2f}")

            # Show extracted columns if any
            if 'columns' in result.parameters and result.parameters['columns']:
                print(f"  Extracted columns: {result.parameters['columns']}")
            if 'target' in result.parameters and result.parameters['target']:
                print(f"  Target variable: {result.parameters['target']}")
            if 'features' in result.parameters and result.parameters['features']:
                print(f"  Features: {result.parameters['features']}")

        except ParseError as e:
            print(f"❌ Could not parse: {e.message}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

def test_with_data_source():
    """Test parser with a data source attached."""
    print("\n" + "=" * 60)
    print("TESTING WITH DATA SOURCE")
    print("=" * 60)

    parser = RequestParser()

    # Create a sample data source
    data_source = DataSource(
        file_id="test_123",
        file_name="sample.csv",
        file_type="csv",
        columns=["age", "income", "education", "satisfaction"],
        shape=(1000, 4)
    )

    test_requests = [
        "calculate mean for age",
        "show correlation matrix",
        "train model to predict satisfaction based on age and income"
    ]

    for request in test_requests:
        print(f"\nRequest: '{request}'")
        try:
            result = parser.parse_request(request, 12345, "test", data_source)
            print(f"✅ Parsed with data source")
            print(f"  Task: {result.task_type} - {result.operation}")
            print(f"  Data source: {result.data_source.file_name} ({result.data_source.shape})")
            print(f"  Available columns: {result.data_source.columns}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_parser_interactive()
    test_with_data_source()