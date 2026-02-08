"""
RF Grid Search with CLEANED features (remove redundant distance metrics).

Removes:
- avg_rr (derived from vwap_width_atr)
- price_to_vwap_atr (redundant with vwap_width_atr)
- All derived R:R columns (long_rr_*, short_rr_*, rr_*)

Keeps:
- vwap_width_atr (single distance metric)
- Momentum indicators (RSI slopes, divergence, etc.)
- Volume indicators (rel_vol, vol_at_extension, etc.)
- VWAP dynamics (vwap_helping, vwap_slope_5, bars_from_vwap)
- Bar context (bar_range_atr, reversal_wick, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from label_generator import LabelConfig, generate_labels

STOP_ATRS = [0.25, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0]

def get_features_cleaned(df):
    """Get feature columns - REMOVE redundant distance metrics."""
    # Base exclusions
    exclude = ['date', 'year', 'open', 'high', 'low', 'close', 'volume', 
               'vwap', 'atr', 'bb_upper', 'bb_lower', 'bb_middle', 
               'price_below_vwap', 
               # Remove redundant distance metrics
               'price_to_vwap',       # Raw dollar distance
               'price_to_vwap_atr',   # REDUNDANT - same as vwap_width_atr but signed
               'price_to_vwap_pct',   # Percentage version
               'vwap_width_pct',      # Percentage version  
               'vwap_dist_pct',       # Another percentage version
               'vwap_dist_atr',       # Yet another distance metric
               'profit_potential_atr',# Same as vwap_width_atr
               'avg_rr',              # DERIVED from vwap_width_atr (distance/stop)
               # Remove zone booleans (we want continuous learning)
               'in_sweet_spot', 'in_tradeable_zone', 'over_extended', 'extreme_extension',
               'reversal_direction',  # Just the sign of distance
    ]
    
    # Also exclude label columns and ALL R:R columns (derived from distance)
    exclude_prefixes = ['label_', 'zone_', 'long_rr', 'short_rr', 'rr_']
    
    features = []
    for c in df.columns:
        if c in exclude or any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
            features.append(c)
    
    return features

def run_rf(df, stop_atr, features, test_year=2024):
    """Train RF for a single stop width."""
    label_col = f"label_s{stop_atr}".replace(".", "_")
    
    valid = df[label_col].notna()
    df_valid = df[valid].copy()
    
    if len(df_valid) < 500:
        return None
    
    X = df_valid[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_valid[label_col].astype(int)
    
    # Train/test split
    df_valid['year'] = pd.to_datetime(df_valid['date']).apply(lambda x: x.year)
    train_mask = df_valid['year'] < test_year
    test_mask = df_valid['year'] >= test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 200 or len(X_test) < 50:
        return None
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=50,
        min_samples_split=100, max_features='sqrt',
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Test predictions
    proba = rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    median_dist = df_valid['vwap_width_atr'].median()
    rr = median_dist / stop_atr
    breakeven = 1 / (1 + rr)
    
    raw_wr = y_test.mean()
    raw_ev = raw_wr * rr - (1 - raw_wr)
    
    result = {
        'stop_atr': stop_atr,
        'rr': rr,
        'breakeven_wr': breakeven,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'raw_wr': raw_wr,
        'raw_ev': raw_ev,
    }
    
    # RF filtered at thresholds
    for thresh in [0.4, 0.5, 0.55, 0.6]:
        mask = proba >= thresh
        n = mask.sum()
        if n >= 30:
            wr = y_test[mask].mean()
            ev = wr * rr - (1 - wr)
            result[f'rf{thresh}_n'] = n
            result[f'rf{thresh}_wr'] = wr
            result[f'rf{thresh}_ev'] = ev
        else:
            result[f'rf{thresh}_n'] = n
            result[f'rf{thresh}_wr'] = np.nan
            result[f'rf{thresh}_ev'] = np.nan
    
    # Feature importances (top 15)
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    result['top_features'] = ', '.join(importances['feature'].tolist())
    
    return result

def main():
    print("="*80)
    print("RF GRID SEARCH - CLEANED FEATURES (No Redundant Distance Metrics)")
    print("="*80)
    
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df):,} bars")
    
    # Generate labels
    print("\nGenerating labels...")
    config = LabelConfig(stop_atrs=STOP_ATRS)
    df = generate_labels(df, config)
    
    features = get_features_cleaned(df)
    print(f"\nUsing {len(features)} features (removed redundant distance metrics)")
    print("\nREMOVED:")
    print("  • price_to_vwap_atr (redundant with vwap_width_atr)")
    print("  • avg_rr (derived from distance)")
    print("  • All long_rr_*, short_rr_*, rr_* columns (derived)")
    print("\nKEPT:")
    print("  • vwap_width_atr (single distance metric)")
    print("  • Momentum, volume, VWAP dynamics, bar context")
    
    print(f"\n\nFeature list:")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    
    # Run RF for each stop
    print(f"\n\nRunning RF for {len(STOP_ATRS)} stop widths...\n")
    results = []
    
    for i, stop in enumerate(STOP_ATRS, 1):
        print(f"[{i}/{len(STOP_ATRS)}] Stop {stop} ATR...", end=" ")
        result = run_rf(df, stop, features)
        if result:
            results.append(result)
            print(f"Raw WR={result['raw_wr']:.1%}, Raw EV={result['raw_ev']:+.3f}, RF0.5 EV={result.get('rf0.5_ev', np.nan):+.3f}")
        else:
            print("skipped")
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(__file__).parent.parent / "data" / "rf_cleaned_features_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS COMPARISON: Cleaned Features vs Original")
    print("="*80)
    
    print("\nCLEANED FEATURES - ALL RESULTS (RF≥0.5):")
    print("Stop  │  R:R  │ Raw WR → RF WR │ Raw EV → RF EV │  Trades  ")
    print("──────┼───────┼────────────────┼────────────────┼──────────")
    for _, row in results_df.iterrows():
        print(f"{row['stop_atr']:0.2f}  │ {row['rr']:0.0f}:1  │ {row['raw_wr']:0.0%} → {row['rf0.5_wr']:0.0%} │ {row['raw_ev']:+6.3f} → {row['rf0.5_ev']:+6.3f} │ {int(row['rf0.5_n']):8,}")
    
    print("\nBEST SETUP:")
    best = results_df.loc[results_df['rf0.5_ev'].idxmax()]
    print(f"Stop: {best['stop_atr']} ATR")
    print(f"R:R: {best['rr']:.2f}:1")
    print(f"Raw WR: {best['raw_wr']:.1%} (EV={best['raw_ev']:+.3f})")
    print(f"RF≥0.5 WR: {best['rf0.5_wr']:.1%} (EV={best['rf0.5_ev']:+.3f}, N={best['rf0.5_n']:.0f})")
    
    print("\nTOP 15 FEATURES (0.25 ATR stop):")
    print(results_df.iloc[0]['top_features'])
    
    print("\n" + "="*80)
    print("📊 KEY QUESTION: Did removing redundant features change results?")
    print("="*80)
    print("\nIf results are SIMILAR:")
    print("  → Confirms redundancy (avg_rr, price_to_vwap_atr added no value)")
    print("  → Distance is still key, but model is cleaner")
    print("\nIf other features now rank higher:")
    print("  → We can see what ELSE matters beyond distance")
    print("  → Momentum/volume quality indicators get their due credit")

if __name__ == "__main__":
    main()
