#!/usr/bin/env python3
"""
Direct test of runpod_pod_manager to trigger diagnostic output.
This will show exactly what runpod module looks like when imported.
"""

import sys
print("=" * 80)
print("DIRECT RUNPOD_POD_MANAGER TEST")
print("=" * 80)

print("\n1. Checking sys.modules BEFORE import:")
print(f"  'runpod' in sys.modules: {'runpod' in sys.modules}")
if 'runpod' in sys.modules:
    print(f"  Type: {type(sys.modules['runpod'])}")

print("\n2. Importing runpod_pod_manager (this will trigger IMPORT DIAGNOSTIC)...")
print("-" * 80)
from src.cloud.runpod_pod_manager import RunPodPodManager, RUNPOD_AVAILABLE

print("-" * 80)
print(f"\n3. Import result:")
print(f"  RUNPOD_AVAILABLE: {RUNPOD_AVAILABLE}")

print("\n4. Checking sys.modules AFTER import:")
print(f"  'runpod' in sys.modules: {'runpod' in sys.modules}")
if 'runpod' in sys.modules:
    print(f"  Type: {type(sys.modules['runpod'])}")
    print(f"  Has create_pod: {hasattr(sys.modules['runpod'], 'create_pod')}")

print("\n" + "=" * 80)
print("TEST COMPLETE - Check output above for diagnostics")
print("=" * 80)
