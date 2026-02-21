"""
Export per-bar features and probabilities from master pipeline for diff analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.model_selector import load_model_for_stop
from scripts.master_pipeline import calculate_core_indicators

def export_master_per_bar(year=2024, stop_atr=1.5, rf_threshold=0.5):
    """Export per-bar features and RF probabilities from master pipeline"""
    
    print(f"Loading data for {year}...")
    df = pd.read_csv("data/tsla_5min_10years.csv")
    
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    
    df["year"] = df["datetime"].dt.year
    df_year = df[df["year"] == year].reset_index(drop=True)
    
    # Add date column
    if 'date' not in df_year.columns:
        df_year['date'] = df_year['datetime'].dt.date
    
    print(f"Calculating indicators for {len(df_year):,} bars...")
    df_year = calculate_core_indicators(df_year, verbose=False)
    
    # Load model
    print(f"Loading model (stop_atr={stop_atr})...")
    model, metadata = load_model_for_stop(stop_atr=stop_atr, models_dir="models", latest=True)
    feature_cols = metadata["features"]
    model.n_jobs = 1  # Single-threaded
    
    print(f"Extracting features and probabilities...")
    
    records = []
    lookback = 200
    
    for i in range(lookback, len(df_year)):
        bar = df_year.iloc[i]
        
        # Check if long setup
        is_setup = bar.get('is_long_setup', False)
        if not is_setup:
            continue
        
        # Get ATR
        atr = bar.get('atr', 0.0)
        if pd.isna(atr) or atr == 0.0:
            continue
        
        # Extract features
        feature_vector = [bar.get(col, np.nan) for col in feature_cols]
        if any(pd.isna(x) for x in feature_vector):
            continue
        
        # Get RF probability
        prob = model.predict_proba([feature_vector])[0, 1]
        
        # Record
        record = {
            'datetime': bar['datetime'],
            'is_setup': True,
            'prob': prob,
        }
        
        # Add all features
        for col in feature_cols:
            record[col] = bar.get(col, np.nan)
        
        records.append(record)
    
    # Save
    df_export = pd.DataFrame(records)
    output_path = f"data/master_per_bar_features_{year}.csv"
    df_export.to_csv(output_path, index=False)
    
    print(f"\n✓ Exported {len(df_export):,} bars to {output_path}")
    print(f"  - Bars with is_long_setup=True and valid features")
    print(f"  - Probability >= {rf_threshold}: {(df_export['prob'] >= rf_threshold).sum():,}")
    
    return output_path


if __name__ == "__main__":
    export_master_per_bar()
