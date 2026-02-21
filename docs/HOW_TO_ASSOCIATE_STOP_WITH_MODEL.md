# How to Associate Stop Width with Models

## Quick Answer

**The stop width is stored in BOTH the filename AND the metadata!**

### Filename Pattern
```
rf_vwap_stop{STOP_ATR}_{TIMESTAMP}.pkl
              ^^^^^^^^
              Stop width here (dots replaced with underscores)

Examples:
- rf_vwap_stop0_25_20260209_153000.pkl  → Stop = 0.25 ATR
- rf_vwap_stop0_50_20260209_153000.pkl  → Stop = 0.50 ATR
- rf_vwap_stop1_00_20260209_153000.pkl  → Stop = 1.00 ATR
```

### Metadata
```python
metadata['stop_atr']  # e.g., 0.5
```

---

## Usage Examples

### 1. Load a Specific Stop Width Model (by filename)
```python
from model_persistence import load_model

# Load the 0.50 ATR stop model
model, metadata = load_model("models/rf_vwap_stop0_50_20260209_153000.pkl")

print(f"Stop width: {metadata['stop_atr']} ATR")  # 0.5
print(f"R:R ratio: {metadata['rr']:.2f}")
```

### 2. Find All Models for a Stop Width
```python
from model_selector import find_models_by_stop_atr

# Find all 0.5 ATR models
models = find_models_by_stop_atr("models", stop_atr=0.5)

for m in models:
    print(f"{m['filepath']} - Saved: {m['saved_at']}")
```

### 3. Load Latest Model for a Stop Width
```python
from model_selector import load_model_for_stop

# Get the most recent 0.5 ATR model
model, metadata = load_model_for_stop(stop_atr=0.5, latest=True)

print(f"Loaded: Stop {metadata['stop_atr']} ATR")
```

### 4. Find Best Model Across All Stop Widths
```python
from model_selector import get_best_model

# Find model with highest P&L (across all stop widths)
filepath, metadata = get_best_model("models", criterion="pnl")

print(f"Best model uses: {metadata['stop_atr']} ATR")
print(f"P&L: ${metadata['test_stats']['rf_threshold_0.5']['total_net_pnl']:,.0f}")
```

### 5. List All Models by Stop Width
```python
from model_selector import list_all_models_summary

# Shows all models grouped by stop width
list_all_models_summary("models")
```

**Output:**
```
================================================================================
AVAILABLE MODELS (9 total)
================================================================================

📊 Stop 0.25 ATR (1 model)
--------------------------------------------------------------------------------
  • rf_vwap_stop0_25_20260209_153000.pkl
    Saved: 2026-02-09 15:30:00
    Features: 42
    Performance: WR=68.0%, EV=+0.35, P&L=$15,000, Trades=500

📊 Stop 0.50 ATR (1 model)
--------------------------------------------------------------------------------
  • rf_vwap_stop0_50_20260209_153000.pkl
    Saved: 2026-02-09 15:30:00
    Features: 42
    Performance: WR=72.0%, EV=+0.45, P&L=$25,000, Trades=450

...
```

---

## In Your Paper Trading App

```python
from model_selector import load_model_for_stop

# At startup: Load the model for your chosen stop width
STOP_ATR = 0.5  # Your strategy's stop width
model, metadata = load_model_for_stop(stop_atr=STOP_ATR)

# Verify it matches
assert metadata['stop_atr'] == STOP_ATR, "Stop width mismatch!"

print(f"✓ Loaded model for stop {STOP_ATR} ATR")
print(f"  R:R: {metadata['rr']:.2f}")
print(f"  Features: {len(metadata['features'])}")

# In trading loop
def on_new_bar(indicators):
    features = [indicators[f] for f in metadata['features']]
    prob = model.predict_proba([features])[0, 1]
    
    if prob >= 0.5:
        # Use the SAME stop width the model was trained on
        stop_loss = entry_price - (metadata['stop_atr'] * indicators['atr'])
        take_profit = entry_price + (metadata['rr'] * metadata['stop_atr'] * indicators['atr'])
        
        enter_trade(stop=stop_loss, target=take_profit)
```

---

## Quick Reference Table

| Task | Function | Example |
|------|----------|---------|
| Load by filename | `load_model(path)` | `load_model("models/rf_vwap_stop0_50_*.pkl")` |
| Find by stop width | `find_models_by_stop_atr(dir, stop)` | `find_models_by_stop_atr("models", 0.5)` |
| Get latest for stop | `get_latest_model_by_stop(dir, stop)` | `get_latest_model_by_stop("models", 0.5)` |
| Get best overall | `get_best_model(dir, criterion)` | `get_best_model("models", "pnl")` |
| Load for trading | `load_model_for_stop(stop)` | `load_model_for_stop(0.5)` |
| List all models | `list_all_models_summary(dir)` | `list_all_models_summary("models")` |

---

## Files Created

✅ **`src/model_selector.py`** - Utility functions to find and select models by stop width

---

## Test It

```bash
# See all available models grouped by stop width
python -c "from src.model_selector import list_all_models_summary; list_all_models_summary('models')"

# Or run the module directly
python src/model_selector.py
```
