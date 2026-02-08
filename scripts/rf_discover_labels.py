"""
RF that discovers best zone/stop combos from pre-computed labels.

Labels are generated upstream by indicators.calc_vwap_reversion_labels().
RF learns which features predict success for each zone/stop combo.

Usage:
    python rf_discover_labels.py [--regenerate_labels]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from indicators import calc_vwap_reversion_labels


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get feature columns (exclude labels, metadata, distance features)."""
    exclude_patterns = [
        'label', 'zone_', 'date', 'year',
        # Distance features - we don't want RF to learn "closer = better"
        'vwap_width_atr', 'vwap_width_pct', 'price_to_vwap', 'price_to_vwap_atr', 
        'price_to_vwap_pct', 'profit_potential_atr', 'vwap_dist_atr', 'vwap_dist_pct',
        # R:R features (derived from distance)
        'rr_', 'long_rr', 'short_rr', 'avg_rr', '_rr', '_be',
        # Zone booleans (derived from distance)
        'in_sweet_spot', 'in_tradeable_zone', 'over_extended', 'extreme_extension',
        # Non-features
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'atr',
        'bb_upper', 'bb_lower', 'bb_middle', 'price_below_vwap',
    ]
    
    feature_cols = []
    for c in df.columns:
        if any(p in c for p in exclude_patterns):
            continue
        if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            feature_cols.append(c)
    
    return feature_cols


def main():
    parser = argparse.ArgumentParser(description='RF discovery from pre-computed labels')
    parser.add_argument('--regenerate_labels', action='store_true',
                        help='Regenerate label columns (slow)')
    parser.add_argument('--test_year', type=int, default=2024,
                        help='First year of test set (default: 2024)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RF VWAP Reversion - Discover Best Zone/Stop Combos")
    print("=" * 80)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    labels_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_labels.csv"
    
    if labels_path.exists() and not args.regenerate_labels:
        print(f"\nLoading pre-computed labels from {labels_path}...")
        df = pd.read_csv(labels_path, parse_dates=['time'], index_col='time')
    else:
        print(f"\nLoading indicators from {data_path}...")
        df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
        
        # Clean inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        print("Generating labels for all zone/stop combinations...")
        df = calc_vwap_reversion_labels(df)
        
        print(f"Saving to {labels_path}...")
        df.to_csv(labels_path)
    
    print(f"Loaded {len(df):,} bars")
    
    # Get label columns
    label_cols = [c for c in df.columns if c.startswith('label_z')]
    print(f"\nFound {len(label_cols)} label columns")
    
    # Add date/year if needed
    if 'date' not in df.columns:
        if df.index.tz is not None:
            df['date'] = df.index.tz_localize(None).date
        else:
            df['date'] = df.index.date
    df['year'] = pd.to_datetime(df['date']).apply(lambda x: x.year)
    
    # Get feature columns
    feature_cols = get_feature_cols(df)
    print(f"Using {len(feature_cols)} features")
    
    # Train/test split
    train_mask = df['year'] < args.test_year
    test_mask = df['year'] >= args.test_year
    
    print(f"\nTrain: {train_mask.sum():,} samples ({df[train_mask]['year'].min()}-{df[train_mask]['year'].max()})")
    print(f"Test:  {test_mask.sum():,} samples ({df[test_mask]['year'].min()}-{df[test_mask]['year'].max()})")
    
    # Results storage
    results = []
    
    # Analyze each label column
    print("\n" + "=" * 80)
    print("ANALYZING ALL ZONE/STOP COMBINATIONS")
    print("=" * 80)
    
    for label_col in sorted(label_cols):
        # Parse zone and stop from column name
        # Format: label_z0_5_1_0_s0_4
        parts = label_col.replace('label_z', '').split('_s')
        zone_parts = parts[0].split('_')
        zone_min = float(f"{zone_parts[0]}.{zone_parts[1]}")
        zone_max = float(f"{zone_parts[2]}.{zone_parts[3]}")
        stop_atr = float(parts[1].replace('_', '.'))
        
        # Calculate R:R and breakeven
        mid_zone = (zone_min + zone_max) / 2
        rr = mid_zone / stop_atr
        breakeven = 1 / (1 + rr)
        
        # Get valid data for this label
        valid_mask = df[label_col].notna()
        n_total = valid_mask.sum()
        
        if n_total < 500:
            continue
        
        # Train data
        train_valid = valid_mask & train_mask
        test_valid = valid_mask & test_mask
        
        n_train = train_valid.sum()
        n_test = test_valid.sum()
        
        if n_train < 200 or n_test < 50:
            continue
        
        # Prepare data
        X_train = df.loc[train_valid, feature_cols].copy()
        y_train = df.loc[train_valid, label_col].astype(int)
        X_test = df.loc[test_valid, feature_cols].copy()
        y_test = df.loc[test_valid, label_col].astype(int)
        
        # Handle inf/nan
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Raw win rates
        raw_train_wr = y_train.mean()
        raw_test_wr = y_test.mean()
        raw_ev = raw_test_wr * rr - (1 - raw_test_wr)
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=50,
            min_samples_split=100,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        test_proba = rf.predict_proba(X_test)[:, 1]
        
        # RF filtered results
        for thresh in [0.5, 0.55, 0.6]:
            rf_mask = test_proba >= thresh
            n_rf = rf_mask.sum()
            if n_rf < 30:
                continue
            
            rf_wr = y_test[rf_mask].mean()
            rf_ev = rf_wr * rr - (1 - rf_wr)
            
            results.append({
                'zone': f"{zone_min}-{zone_max}",
                'stop': stop_atr,
                'rr': rr,
                'breakeven': breakeven,
                'n_test': n_test,
                'raw_wr': raw_test_wr,
                'raw_ev': raw_ev,
                'rf_thresh': thresh,
                'rf_n': n_rf,
                'rf_wr': rf_wr,
                'rf_ev': rf_ev,
                'positive_ev': rf_ev > 0,
            })
    
    # Convert to DataFrame and display
    results_df = pd.DataFrame(results)
    
    # Show positive EV combinations
    print("\n" + "=" * 80)
    print("POSITIVE EV COMBINATIONS")
    print("=" * 80)
    
    positive_ev = results_df[results_df['positive_ev'] == True].sort_values('rf_ev', ascending=False)
    
    if len(positive_ev) > 0:
        print(f"\nFound {len(positive_ev)} positive EV combinations:\n")
        display_cols = ['zone', 'stop', 'rr', 'breakeven', 'n_test', 'raw_wr', 'raw_ev', 
                       'rf_thresh', 'rf_n', 'rf_wr', 'rf_ev']
        
        # Format for display
        for _, row in positive_ev.head(20).iterrows():
            print(f"Zone {row['zone']:>10s} | Stop {row['stop']:.2f} | R:R {row['rr']:.2f}:1 | "
                  f"BE {row['breakeven']:.1%} | Raw {row['raw_wr']:.1%} ({row['raw_ev']:+.3f}R) | "
                  f"RF>={row['rf_thresh']:.2f}: {row['rf_wr']:.1%} ({row['rf_ev']:+.3f}R) n={row['rf_n']}")
    else:
        print("No positive EV combinations found")
    
    # Show best raw EV (already positive without RF)
    print("\n" + "=" * 80)
    print("BEST RAW EV (no RF needed)")
    print("=" * 80)
    
    raw_positive = results_df[results_df['raw_ev'] > 0].drop_duplicates(
        subset=['zone', 'stop']).sort_values('raw_ev', ascending=False)
    
    if len(raw_positive) > 0:
        print(f"\nFound {len(raw_positive)} raw positive EV combinations:\n")
        for _, row in raw_positive.head(10).iterrows():
            print(f"Zone {row['zone']:>10s} | Stop {row['stop']:.2f} | R:R {row['rr']:.2f}:1 | "
                  f"BE {row['breakeven']:.1%} | Raw WR {row['raw_wr']:.1%} | EV {row['raw_ev']:+.3f}R | "
                  f"n={row['n_test']}")
    
    # Save results
    results_path = Path(__file__).parent.parent / "data" / "rf_discovery_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best = results_df.loc[results_df['rf_ev'].idxmax()] if len(results_df) > 0 else None
    if best is not None:
        print(f"\nBest combination found:")
        print(f"  Zone: {best['zone']}")
        print(f"  Stop: {best['stop']} ATR")
        print(f"  R:R: {best['rr']:.2f}:1")
        print(f"  Breakeven WR: {best['breakeven']:.1%}")
        print(f"  Raw WR: {best['raw_wr']:.1%} -> EV = {best['raw_ev']:+.3f}R")
        print(f"  RF>={best['rf_thresh']:.2f} WR: {best['rf_wr']:.1%} -> EV = {best['rf_ev']:+.3f}R")
        print(f"  N trades: {best['rf_n']}")


if __name__ == "__main__":
    main()
