#!/usr/bin/env python3
"""
Test script - verify all imports work
"""

import sys
from pathlib import Path

print("=" * 80)
print("Testing Fine-Grained Data Valuation Imports")
print("=" * 80)

# Test basic packages
print("\n[1] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[2] Testing opendataval (modified)...")
try:
    from fine_grained.opendataval import *
    print("  ✓ opendataval imported")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[3] Testing ghostsuite...")
try:
    import ghostsuite
    print(f"  ✓ ghostsuite installed")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[4] Testing logix...")
try:
    from logix import *
    print("  ✓ logix-ai installed")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n[5] Testing core ML libraries...")
try:
    import numpy as np
    import pandas as pd
    import sklearn
    print(f"  ✓ numpy {np.__version__}")
    print(f"  ✓ pandas {pd.__version__}")
    print(f"  ✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ All imports successful!")
print("=" * 80)
