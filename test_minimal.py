import sys
print("Step 1: Starting", flush=True)

from pathlib import Path
print("Step 2: Path imported", flush=True)

sys.path.insert(0, str(Path(__file__).parent))
print("Step 3: Path modified", flush=True)

import pandas as pd
print("Step 4: Pandas imported", flush=True)

from sim_trading import StreamingSimulator
print("Step 5: StreamingSimulator imported", flush=True)

from sim_trading.streaming_indicators_aligned import StreamingIndicatorsAligned  
print("Step 6: StreamingIndicatorsAligned imported", flush=True)

from src.model_selector import load_model_for_stop
print("Step 7: load_model_for_stop imported", flush=True)

print("Step 8: Loading model...", flush=True)
model, metadata = load_model_for_stop(stop_atr=1.5, models_dir="models", latest=True)
print(f"Step 9: Model loaded! Features: {len(metadata['features'])}", flush=True)

print("Step 10: ALL IMPORTS SUCCESSFUL!", flush=True)
