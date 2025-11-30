#!/usr/bin/env python3
"""
Standalone test to verify runpod import and functionality.
This script isolates the runpod import to identify any issues.
"""

import sys

print("=" * 60)
print("RUNPOD IMPORT TEST")
print("=" * 60)

print("\n1. Check sys.modules before import:")
print(f"  'runpod' in sys.modules: {'runpod' in sys.modules}")
if 'runpod' in sys.modules:
    print(f"  Type: {type(sys.modules['runpod'])}")
    print(f"  Is None: {sys.modules['runpod'] is None}")

print("\n2. Attempting import runpod...")
try:
    import runpod
    print("  ✅ Import successful")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

print("\n3. Inspect imported module:")
print(f"  Type: {type(runpod)}")
print(f"  Is None: {runpod is None}")
print(f"  Module name: {runpod.__name__ if hasattr(runpod, '__name__') else 'N/A'}")
print(f"  Module file: {runpod.__file__ if hasattr(runpod, '__file__') else 'N/A'}")
print(f"  Module id: {id(runpod)}")

print("\n4. Check for create_pod:")
print(f"  Has create_pod: {hasattr(runpod, 'create_pod')}")
if hasattr(runpod, 'create_pod'):
    print(f"  create_pod type: {type(runpod.create_pod)}")
    print(f"  create_pod callable: {callable(runpod.create_pod)}")
    print(f"  Has __code__: {hasattr(runpod.create_pod, '__code__')}")
else:
    print("  ❌ create_pod NOT FOUND")

print("\n5. Check for mock indicators:")
print(f"  Has _mock_name: {hasattr(runpod, '_mock_name')}")
print(f"  Type name contains 'mock': {'mock' in type(runpod).__name__.lower()}")

print("\n6. First 20 public attributes:")
attrs = [a for a in dir(runpod) if not a.startswith('_')]
print(f"  {attrs[:20]}")

print("\n7. sys.modules check after import:")
print(f"  'runpod' in sys.modules: {'runpod' in sys.modules}")
if 'runpod' in sys.modules:
    print(f"  Same object: {sys.modules['runpod'] is runpod}")
    print(f"  sys.modules id: {id(sys.modules['runpod'])}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
