print("Starting imports test...")
import sys
print("1. sys imported")
from pathlib import Path
print("2. pathlib imported")
sys.path.insert(0, str(Path(__file__).parent))
print("3. path modified")
import argparse
print("4. argparse imported")
import pandas as pd
print("5. pandas imported")

try:
    from sim_trading import StreamingSimulator
    print("6. StreamingSimulator imported")
except Exception as e:
    print(f"6. ERROR importing StreamingSimulator: {e}")

try:
    from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned
    print("7. StreamingIndicatorsAligned imported")
except Exception as e:
    print(f"7. ERROR importing StreamingIndicatorsAligned: {e}")

try:
    from src.model_selector import load_model_for_stop
    print("8. load_model_for_stop imported")
except Exception as e:
    print(f"8. ERROR importing load_model_for_stop: {e}")

print("All imports completed!")
