"""
Export Master Pipeline Per-Bar Features for Comparison
Generates per-bar feature vectors and RF probabilities for 2024
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(assume_finite=True)

from src.model_selector import load_model_for_stop
from scripts.master_pipeline import calculate_core_indicators

def main():
    print("="*80)
    print("MASTER PIPELINE PER-BAR EXPORT")
    print("="*80)
    
    # Configuration
    year = 2024
    stop_atr = 1.5
    rf_threshold = 0.5
    
    print(f"\nConfiguration:")
    print(f"  Year: {year}")
    print(f"  Stop ATR: {stop_atr}")
    print(f"  RF Threshold: {rf_threshold}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    df["year"] = df["datetime"].dt.year
    df_year = df[df["year"] == year].reset_index(drop=True)
    print(f"Loaded {len(df_year):,} bars for {year}")
    
    # Load model
    print("\nLoading model...")
    model, metadata = load_model_for_stop(stop_atr=stop_atr, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    rr = float(metadata.get("rr", 1.2))
    model.n_jobs = 1  # Single-threaded
    
    print(f"Model loaded. Features: {len(feature_cols)}, RR: {rr}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    if 'date' not in df_year.columns:
        df_year['date'] = df_year['datetime'].dt.date
    
    df_year = calculate_core_indicators(df_year, verbose=False)
    print(f"Indicators calculated for {len(df_year):,} bars")
    
    # Export per-bar features
    print("\nExporting per-bar features...")
    per_bar_records = []
    
    lookback = 200
    for i in range(lookback, len(df_year)):
        bar = df_year.iloc[i]
        
        # Get ATR
        atr = bar.get('atr', 0.0)
        if pd.isna(atr) or atr == 0.0:
            continue
        
        # Check setup
        is_setup = bar.get('is_long_setup', False)
        
        # Get features
        feature_vector = [bar.get(col, np.nan) for col in feature_cols]
        has_all_features = not any(pd.isna(x) for x in feature_vector)
        
        # Calculate probability if we have all features
        prob = np.nan
        if has_all_features:
            prob = model.predict_proba([feature_vector])[0, 1]
        
        # Record this bar
        record = {
            'datetime': bar['datetime'],
            'is_setup': is_setup,
            'has_all_features': has_all_features,
            'prob': prob,
        }
        
        # Add all features
        for col in feature_cols:
            record[col] = bar.get(col, np.nan)
        
        per_bar_records.append(record)
        
        if (i - lookback) % 5000 == 0:
            print(f"  Progress: {i-lookback}/{len(df_year)-lookback} bars")
    
    # Save to CSV
    output_file = "data/master_per_bar_features_2024.csv"
    per_bar_df = pd.DataFrame(per_bar_records)
    per_bar_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Exported {len(per_bar_df):,} bars to {output_file}")
    print(f"  Setups: {per_bar_df['is_setup'].sum():,}")
    print(f"  With all features: {per_bar_df['has_all_features'].sum():,}")
    print(f"  Prob >= {rf_threshold}: {(per_bar_df['prob'] >= rf_threshold).sum():,}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
