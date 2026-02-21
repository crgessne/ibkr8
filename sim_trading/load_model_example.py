"""
Example: Loading and Using a Saved RF Model

This script demonstrates how to load a trained RF model from disk
and use it for predictions in a separate application (e.g., paper trading).

Usage:
    python examples/load_model_example.py models/rf_vwap_stop0_50_*.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from model_persistence import load_model, list_saved_models, validate_model_compatibility


def main():
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found. Train models first using master_pipeline.py")
        print(f"\nUsage: python scripts/master_pipeline.py")
        return
    
    # List all available models
    print("\n" + "="*80)
    print("AVAILABLE TRAINED MODELS")
    print("="*80 + "\n")
    
    models_info = list_saved_models(str(models_dir))
    
    if len(models_info) == 0:
        print("No saved models found. Train models first using master_pipeline.py")
        return
    
    for i, info in enumerate(models_info, 1):
        print(f"{i}. {Path(info['filepath']).name}")
        print(f"   Stop ATR: {info['stop_atr']}")
        print(f"   Features: {info['features_count']}")
        print(f"   Saved: {info['saved_at']}")
        print()
    
    # Load the first model as an example
    example_model_path = models_info[0]['filepath']
    
    print("="*80)
    print(f"LOADING MODEL: {Path(example_model_path).name}")
    print("="*80 + "\n")
    
    model, metadata = load_model(example_model_path)
    
    print("\nModel Metadata:")
    print(f"  Stop ATR: {metadata['stop_atr']}")
    print(f"  R:R Ratio: {metadata['rr']:.2f}")
    print(f"  Features: {len(metadata['features'])}")
    print(f"  Training samples: {metadata['train_stats']['n_samples']:,}")
    print(f"  Training date range: {metadata['train_stats']['date_range'][0]} to {metadata['train_stats']['date_range'][1]}")
    print(f"  Training win rate: {metadata['train_stats']['win_rate']*100:.1f}%")
    print(f"  Test samples: {metadata['test_stats']['n_samples']:,}")
    print(f"  Test win rate (RF≥0.5): {metadata['test_stats']['rf_threshold_0.5']['win_rate']*100:.1f}%")
    print(f"  Test EV (RF≥0.5): {metadata['test_stats']['rf_threshold_0.5']['ev']:+.3f}")
    print(f"  Test P&L (RF≥0.5): ${metadata['test_stats']['rf_threshold_0.5']['total_net_pnl']:,.0f}")
    
    print("\nTop 10 Features by Importance:")
    for i, feat_info in enumerate(metadata['feature_importance_top10'], 1):
        print(f"  {i:2d}. {feat_info['feature']:30s} ({feat_info['importance']:.4f})")
    
    # Demonstrate making predictions
    print("\n" + "="*80)
    print("MAKING PREDICTIONS (EXAMPLE)")
    print("="*80 + "\n")
    
    # Create dummy data for demonstration
    n_samples = 5
    n_features = len(metadata['features'])
    
    print(f"Creating {n_samples} dummy samples with {n_features} features...")
    X_dummy = np.random.randn(n_samples, n_features)
    X_dummy_df = pd.DataFrame(X_dummy, columns=metadata['features'])
    
    # Make predictions
    probabilities = model.predict_proba(X_dummy_df)[:, 1]
    predictions = model.predict(X_dummy_df)
    
    print("\nPrediction Results:")
    print(f"{'Sample':<8} {'Probability':<15} {'Predicted Label':<20} {'Signal'}")
    print("-" * 60)
    
    for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
        signal = "ENTER TRADE ✓" if prob >= 0.5 else "SKIP"
        print(f"{i+1:<8} {prob:.4f}{'':11} {pred:<20} {signal}")
    
    print("\n" + "="*80)
    print("USAGE IN PAPER TRADING")
    print("="*80 + "\n")
    
    print("In your paper trading application:")
    print("""
from model_persistence import load_model

# Load the trained model once at startup
model, metadata = load_model("models/rf_vwap_stop0_50_20260208_120424.pkl")

# For each new bar:
#   1. Calculate indicators for the current bar
#   2. Create feature vector matching metadata['features']
#   3. Get prediction probability
#   4. Make trading decision based on threshold

def on_new_bar(bar_data):
    # Calculate features (must match training features exactly!)
    features = calculate_features(bar_data)  # Your indicator calculation
    
    # Get prediction
    prob = model.predict_proba([features])[0, 1]
    
    # Trading decision
    if prob >= 0.5:  # Use same threshold as training
        enter_trade(stop_atr=metadata['stop_atr'])
    """)
    
    print("\n✅ Example complete! Model loaded and ready for use.")


if __name__ == "__main__":
    main()
