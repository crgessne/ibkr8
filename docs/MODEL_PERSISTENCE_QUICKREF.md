# Model Persistence - Quick Reference

## 🎯 Quick Start

### 1. Train and Save Models
```bash
python scripts/master_pipeline.py
```
**Output**: Models saved to `models/rf_vwap_stop*.pkl`

### 2. List Available Models
```python
from model_persistence import list_saved_models
models = list_saved_models("models")
```

### 3. Load a Model
```python
from model_persistence import load_model
model, metadata = load_model("models/rf_vwap_stop0_50_20260209_153000.pkl")
```

### 4. Make Predictions
```python
# Extract features (must match metadata['features'] order!)
features = [indicators[f] for f in metadata['features']]

# Get prediction probability
prob = model.predict_proba([features])[0, 1]

# Trading decision
if prob >= 0.5:
    enter_trade(stop_atr=metadata['stop_atr'])
```

---

## 📦 What's Saved

Each model includes:
- ✅ Trained RandomForest classifier
- ✅ Feature list (exact order)
- ✅ Stop ATR and R:R ratio
- ✅ Training/test statistics
- ✅ Performance metrics
- ✅ Feature importance rankings
- ✅ RF hyperparameters

---

## 📁 File Structure

```
models/
├── rf_vwap_stop0_50_20260209_153000.pkl   ← Binary model
├── rf_vwap_stop0_50_20260209_153000.json  ← Human-readable metadata
```

**Naming**: `rf_vwap_stop{STOP_ATR}_{TIMESTAMP}.pkl`

---

## 🔑 Key Metadata Fields

```python
metadata['stop_atr']           # 0.5
metadata['rr']                 # 2.5
metadata['features']           # ['vwap_pct', 'rsi', ...]
metadata['rf_params']          # {'n_estimators': 100, ...}
metadata['train_stats']        # Training set info
metadata['test_stats']         # Test set metrics
metadata['feature_importance_top10']  # Top features
```

---

## 🚀 Paper Trading Integration

```python
# Load once at startup
from model_persistence import load_model
model, metadata = load_model("models/rf_vwap_stop0_50_20260209_153000.pkl")

print(f"Loaded: Stop {metadata['stop_atr']} ATR, R:R {metadata['rr']:.2f}")

# In your trading loop
def on_new_bar(bar_history):
    # 1. Calculate indicators
    indicators = calculate_indicators(bar_history)
    
    # 2. Extract features IN CORRECT ORDER
    features = [indicators[f] for f in metadata['features']]
    
    # 3. Get prediction
    prob = model.predict_proba([features])[0, 1]
    
    # 4. Make decision (use same threshold as training)
    if prob >= 0.5:
        current_atr = indicators['atr']
        stop_loss = current_price - (metadata['stop_atr'] * current_atr)
        take_profit = current_price + (metadata['rr'] * metadata['stop_atr'] * current_atr)
        
        enter_trade(stop=stop_loss, target=take_profit)
```

---

## ⚠️ Critical: Feature Alignment

**MUST MATCH EXACTLY:**
```python
# Get feature order from model
required_features = metadata['features']

# Verify your features match
assert list(indicators.keys()) == required_features, "Feature mismatch!"

# Extract in correct order
features = [indicators[f] for f in required_features]
```

**Common mistakes:**
- ❌ Wrong feature order → Wrong predictions
- ❌ Missing features → Error
- ❌ Extra features → Error
- ❌ Different feature names → Error

---

## 🛠️ Utility Functions

```python
from model_persistence import (
    save_model,
    load_model,
    list_saved_models,
    get_model_summary,
    validate_model_compatibility
)

# List all models
models = list_saved_models("models")

# Quick metadata check (no model loading)
summary = get_model_summary("models/rf_vwap_stop0_50_20260209_153000.pkl")

# Validate compatibility
is_ok, issues = validate_model_compatibility(
    model_metadata=metadata,
    required_features=current_features,
    stop_atr=0.5
)
```

---

## 📊 Example Output

```
✓ Loaded model from: models/rf_vwap_stop0_50_20260209_153000.pkl
  - Model type: RandomForestClassifier
  - Saved at: 2026-02-09T15:30:00
  - Stop ATR: 0.5
  - Features: 42

Model Metadata:
  Stop ATR: 0.5
  R:R Ratio: 2.50
  Features: 42
  Training samples: 50,000
  Training date range: 2020-01-01 to 2023-12-31
  Training win rate: 65.0%
  Test samples: 10,000
  Test win rate (RF≥0.5): 68.0%
  Test EV (RF≥0.5): +0.45
  Test P&L (RF≥0.5): $25,000
```

---

## 🐛 Troubleshooting

### Model won't load
```python
FileNotFoundError: Model file not found
```
**Fix**: Check path, ensure models were trained

### Feature mismatch
```python
ValueError: X has 45 features but model was trained with 42
```
**Fix**: 
1. Check `metadata['features']`
2. Ensure indicator calculation produces same features
3. Extract features in exact same order

### Wrong predictions
**Check**:
1. ✅ Feature order matches
2. ✅ Feature types match (bool/int/float)
3. ✅ NaN handling matches training (fillna(0))
4. ✅ Indicator calculations are identical

---

## 📚 Documentation

- **User Guide**: `models/README.md`
- **API Reference**: `src/model_persistence.py`
- **Example**: `examples/load_model_example.py`
- **Implementation**: `docs/model_persistence_implementation.md`

---

## 🎯 Best Practices

1. **Train once, use many** - No need to retrain every run
2. **Version control** - Keep track of deployed model file
3. **Validate features** - Always check feature alignment
4. **Monitor performance** - Log predictions and actual outcomes
5. **Regular retraining** - Update models when performance degrades

---

## ✅ Checklist for Production

- [ ] Model trained and saved successfully
- [ ] Metadata JSON file readable
- [ ] Features list documented
- [ ] Feature calculation tested
- [ ] Feature order verified
- [ ] Prediction threshold documented (0.5)
- [ ] Stop ATR and R:R loaded from metadata
- [ ] Logging implemented for predictions
- [ ] Fallback strategy in place
- [ ] Model file backed up

---

## 🔗 Related Scripts

```bash
# Train models
python scripts/master_pipeline.py

# View example usage
python examples/load_model_example.py

# Run streaming simulator (future: with loaded model)
python examples/simulate_streaming.py
```

---

**Questions?** See `models/README.md` for comprehensive documentation.
