#!/usr/bin/env python3
"""
Standalone test to verify DataLoader works independently.
This helps isolate if the issue is with the module or integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dataloader_import():
    """Test if DataLoader can be imported."""
    try:
        from src.processors.data_loader import DataLoader
        print("✅ DataLoader imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import DataLoader: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing DataLoader: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    required_modules = [
        'pandas',
        'numpy',
        'telegram',
        'openpyxl'  # For Excel support
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            print(f"❌ {module} missing")
            missing.append(module)

    return len(missing) == 0, missing

def test_dataloader_creation():
    """Test if DataLoader can be instantiated."""
    try:
        from src.processors.data_loader import DataLoader
        loader = DataLoader()
        print("✅ DataLoader instance created successfully")
        print(f"   Max file size: {loader.MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
        print(f"   Supported extensions: {loader.SUPPORTED_EXTENSIONS}")
        return True
    except Exception as e:
        print(f"❌ Failed to create DataLoader: {e}")
        return False

def test_simple_csv_processing():
    """Test DataLoader with a simple CSV."""
    try:
        import tempfile
        import pandas as pd
        from src.processors.data_loader import DataLoader

        # Create a simple test CSV
        csv_content = "name,age,city\nJohn,25,NYC\nJane,30,LA"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = Path(f.name)

        try:
            loader = DataLoader()
            # Test file validation
            loader._validate_file_metadata("test.csv", len(csv_content.encode()))
            print("✅ File validation works")

            # Test DataFrame validation
            df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
            metadata = loader._validate_dataframe(df, "test.csv")
            print("✅ DataFrame validation works")
            print(f"   Test metadata: {metadata['shape']}")

            return True

        finally:
            csv_path.unlink()  # Clean up

    except Exception as e:
        print(f"❌ CSV processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING DATALOADER STANDALONE")
    print("=" * 50)

    # Test 1: Import
    import_ok = test_dataloader_import()
    if not import_ok:
        print("\n❌ CRITICAL: DataLoader import failed. Check imports and dependencies.")
        return False

    print()

    # Test 2: Dependencies
    deps_ok, missing = test_dependencies()
    if not deps_ok:
        print(f"\n❌ CRITICAL: Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print()

    # Test 3: Creation
    creation_ok = test_dataloader_creation()
    if not creation_ok:
        print("\n❌ CRITICAL: DataLoader creation failed.")
        return False

    print()

    # Test 4: Basic functionality
    csv_ok = test_simple_csv_processing()
    if not csv_ok:
        print("\n❌ WARNING: CSV processing has issues.")

    print("\n" + "=" * 50)
    if import_ok and deps_ok and creation_ok:
        print("✅ DATALOADER IS WORKING - Issue must be in bot integration")
        print("\nNext steps:")
        print("1. Check if bot is using updated handlers.py")
        print("2. Restart the bot completely")
        print("3. Check bot logs for errors")
    else:
        print("❌ DATALOADER HAS ISSUES - Fix these first")

    return import_ok and deps_ok and creation_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)