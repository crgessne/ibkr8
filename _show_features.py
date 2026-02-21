import sys
sys.path.insert(0, "src")
from model_persistence import load_model
m, meta = load_model("models/rf_vwap_stop0.5_20260217_154448.pkl")
print("Features:")
for f in meta["features"]:
    print(f"  {f}")
print(f"\nTotal: {len(meta['features'])}")
