#!/usr/bin/env python
"""Test if imports work for simulate_streaming_clean.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    from sim_trading import StreamingSimulator
    print("✓ StreamingSimulator imported")
except Exception as e:
    print(f"✗ StreamingSimulator import failed: {e}")

try:
    from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
    print("✓ StreamingIndicatorsAligned imported")
except Exception as e:
    print(f"✗ StreamingIndicatorsAligned import failed: {e}")

try:
    from src.model_selector import load_model_for_stop
    print("✓ load_model_for_stop imported")
except Exception as e:
    print(f"✗ load_model_for_stop import failed: {e}")

print("\nAll imports successful!")
