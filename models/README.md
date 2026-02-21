# Model Persistence System

This directory contains trained RandomForest models for the VWAP Reversion Strategy, along with their metadata and configuration.

## Overview

The model persistence system allows you to:
1. **Train once, use many times** - Train models in `master_pipeline.py` and save them to disk
2. **Load in paper trading** - Load pre-trained models in your live/paper trading application
3. **Version control** - Track model versions with timestamps and metrics
4. **Metadata tracking** - Store complete training configuration, feature lists, and performance metrics

## Directory Structure

```
models/
├── rf_vwap_stop0_25_20260208_120424.pkl    # Model file (binary)
├── rf_vwap_stop0_25_20260208_120424.json   # Metadata (human-readable)
├── rf_vwap_stop0_50_20260208_120424.pkl
├── rf_vwap_stop0_50_20260208_120424.json
└── ...
```

## File Naming Convention

Format: `rf_vwap_stop{STOP_ATR}_{TIMESTAMP}.pkl`

Examples:
- `rf_vwap_stop0_25_20260208_120424.pkl` - Stop = 0.25 ATR, trained on 2026-02-08 at 12:04:24
- `rf_vwap_stop0_50_20260208_120424.pkl` - Stop = 0.50 ATR, trained on 2026-02-08 at 12:04:24
- `rf_vwap_stop1_00_20260208_120424.pkl` - Stop = 1.00 ATR, trained on 2026-02-08 at 12:04:24

## Metadata Contents

Each `.json` file contains:

```json
{
  "stop_atr": 0.5,
  "rr": 2.5,
  "features": ["vwap_pct", "rsi", "bb_width", ...],
  "rf_params": {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 50,
    ...
  },
  "train_stats": {
    "n_samples": 50000,
    "date_range": ["2020-01-01", "2023-12-31"],
    "win_rate": 0.65
  },
  "test_stats": {
    "n_samples": 10000,
    "date_range": ["2024-01-01", "2024-12-31"],
    "win_rate": 0.68,
    "rf_threshold_0.5": {
      "n_trades": 500,
      "win_rate": 0.72,
      "ev": 0.45,
      "total_net_pnl": 25000,
      "avg_net_pnl_per_trade": 50
    }
  },
  "feature_importance_top10": [
    {"feature": "vwap_pct", "importance": 0.15},
    {"feature": "rsi", "importance": 0.12},
    ...
  ],
  "breakeven_wr": 0.286,
  "timestamp": "20260208_120424",
  "saved_at": "2026-02-08T12:04:24",
  "model_type": "RandomForestClassifier"
}
```

## Usage

### 1. Training and Saving Models

Run the master pipeline to train and save models:

```bash
python scripts/master_pipeline.py
```

This will:
- Train RF models for all configured stop widths
- Evaluate performance on test data
- Save each model with comprehensive metadata
- Output model paths when complete

### 2. Loading a Saved Model

```python
from src.model_persistence import load_model

# Load a specific model
model, metadata = load_model("models/rf_vwap_stop0_50_20260208_120424.pkl")

# Access model
probabilities = model.predict_proba(X_new)[:, 1]

# Access metadata
stop_atr = metadata['stop_atr']
features = metadata['features']
rr = metadata['rr']
```

### 3. Listing Available Models

```python
from src.model_persistence import list_saved_models

models = list_saved_models("models")
for model_info in models:
    print(f"Stop: {model_info['stop_atr']}, Features: {model_info['features_count']}")
```

### 4. Validating Model Compatibility

```python
from src.model_persistence import validate_model_compatibility

# Check if model is compatible with current data
is_compatible, issues = validate_model_compatibility(
    model_metadata=metadata,
    required_features=current_feature_list,
    stop_atr=0.5
)

if not is_compatible:
    print(f"Model incompatible: {issues}")
```

### 5. Example: Paper Trading Application

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_persistence import load_model
from streaming_indicators import StreamingIndicators

# Load model once at startup
model, metadata = load_model("models/rf_vwap_stop0_50_20260208_120424.pkl")
indicator_calc = StreamingIndicators()

print(f"Loaded model: Stop {metadata['stop_atr']} ATR, R:R {metadata['rr']:.2f}")
print(f"Required features: {len(metadata['features'])}")

# In your trading loop
def on_new_bar(bar_history):
    # Calculate indicators on rolling window
    indicators = indicator_calc.calculate(bar_history)
    
    # Extract features in correct order
    feature_values = [indicators[feat] for feat in metadata['features']]
    
    # Get prediction
    prob = model.predict_proba([feature_values])[0, 1]
    
    # Trading decision (use same threshold as training)
    if prob >= 0.5:
        stop_atr = metadata['stop_atr']
        rr = metadata['rr']
        
        # Calculate position sizing based on ATR
        current_atr = indicators['atr']
        stop_distance_dollars = stop_atr * current_atr
        target_distance_dollars = rr * stop_distance_dollars
        
        # Place order
        enter_trade(
            stop_loss=current_price - stop_distance_dollars,
            take_profit=current_price + target_distance_dollars
        )
```

## Best Practices

### Model Selection

1. **Choose by objective**:
   - Max P&L: Select model with highest `total_net_pnl`
   - Max EV per trade: Select model with highest `ev`
   - Risk-adjusted: Consider both P&L and number of trades

2. **Check recency**: Use models trained on recent data if market conditions change

3. **Validate on walk-forward**: Prefer models tested with walk-forward validation

### Production Deployment

1. **Feature alignment**: MUST match features exactly in production
   ```python
   assert list(current_features) == metadata['features'], "Feature mismatch!"
   ```

2. **Version control**: Track which model version is deployed
   ```python
   deployed_model_path = "models/rf_vwap_stop0_50_20260208_120424.pkl"
   with open("deployed_model.txt", "w") as f:
       f.write(deployed_model_path)
   ```

3. **Monitoring**: Log prediction probabilities and track performance
   ```python
   logger.info(f"Model prediction: prob={prob:.3f}, threshold=0.5, decision={'ENTER' if prob>=0.5 else 'SKIP'}")
   ```

4. **Fallback**: Have a strategy if model fails to load
   ```python
   try:
       model, metadata = load_model(model_path)
   except Exception as e:
       logger.error(f"Failed to load model: {e}")
       # Fall back to simple rules-based strategy
   ```

## Model Lifecycle

1. **Training** (`master_pipeline.py`)
   - Train on historical data
   - Evaluate on hold-out test set
   - Save model + metadata

2. **Validation** (optional)
   - Load model
   - Test on new unseen data
   - Verify feature compatibility

3. **Deployment** (paper/live trading)
   - Load model at startup
   - Calculate indicators in real-time
   - Make predictions on current bar
   - Execute trades based on predictions

4. **Monitoring**
   - Track live performance vs. backtest
   - Log prediction distributions
   - Monitor for model drift

5. **Retraining**
   - When performance degrades
   - When significant market regime change
   - Periodically (e.g., quarterly)

## Troubleshooting

### Model won't load

```python
FileNotFoundError: Model file not found
```
**Solution**: Check file path, ensure model was saved successfully

### Feature mismatch

```python
ValueError: X has 45 features but model was trained with 42
```
**Solution**: Check `metadata['features']` and ensure your indicator calculation produces the exact same features in the same order

### Version incompatibility

```python
ModuleNotFoundError: No module named 'sklearn.ensemble._forest'
```
**Solution**: Ensure same sklearn version as training (check `requirements.txt`)

### Model predictions look wrong

1. **Check feature order**: Features must be in exact same order as training
2. **Check feature types**: Ensure bool/int/float types match
3. **Check for NaN/inf**: Handle missing values same way as training (fillna(0))
4. **Verify indicators**: Print feature values and compare to training data ranges

## API Reference

See `src/model_persistence.py` for complete API documentation.

### Main Functions

- `save_model(model, filepath, metadata)` - Save model to disk
- `load_model(filepath)` - Load model from disk
- `list_saved_models(models_dir)` - List all models in directory
- `get_model_summary(filepath)` - Get metadata without loading model
- `validate_model_compatibility(metadata, features, stop_atr)` - Check compatibility

## Examples

See `examples/load_model_example.py` for a complete working example of:
- Listing available models
- Loading a model
- Inspecting metadata
- Making predictions
- Using in a paper trading context

Run it with:
```bash
python examples/load_model_example.py
```

## Security Notes

- Model files (`.pkl`) contain serialized Python objects
- Only load models from trusted sources
- Pickle files can execute arbitrary code when loaded
- Consider using `joblib` for large models (future enhancement)

## Future Enhancements

Potential improvements:
- [ ] Model versioning system (v1, v2, v3)
- [ ] Model comparison tool
- [ ] Automatic model selection based on recent performance
- [ ] Model ensembling (combine multiple models)
- [ ] Compressed storage for large models
- [ ] Cloud storage integration (S3, Azure Blob)
- [ ] Model A/B testing framework
