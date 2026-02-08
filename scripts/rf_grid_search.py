"""
RF Grid Search over stop widths only.

RF uses ALL data and ALL features including vwap_width_atr.
RF learns what ATR distance + other features predict wins.
No pre-defined zones - just sweep stop widths.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from label_generator import LabelConfig, generate_labels


# Stop widths to evaluate
STOP_ATRS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]

# Min ATR distance to consider (filter out bars too close to VWAP)
MIN_ATR_DIST = 0.25


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get feature columns for RF."""
    exclude = [
        'date', 'year', 'time',
        # Raw price/vol - use normalized versions instead
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'atr',
        'bb_upper', 'bb_lower', 'bb_middle',
        # Booleans that are redundant
        'price_below_vwap', 'in_sweet_spot', 'in_tradeable_zone', 
        'over_extended', 'extreme_extension',
    ]
    
    # Exclude label columns and R:R columns (derived from distance)
    exclude_prefixes = ['label_', 'long_rr', 'short_rr', 'rr_']
    
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype not in ['float64', 'int64', 'float32', 'int32']:
            continue
        feature_cols.append(c)
    
    return feature_cols


def run_rf_for_stop(df: pd.DataFrame, stop_atr: float, feature_cols: list,
                    min_atr_dist: float = 0.25, test_year: int = 2024) -> dict:
    """
    Run RF for a specific stop width on ALL data (filtered by min ATR distance).
    
    RF sees vwap_width_atr as a feature and learns what distances work.
    """
    label_col = f"label_s{stop_atr}".replace(".", "_")
    
    # Filter to bars with sufficient extension
    mask = df['vwap_width_atr'] >= min_atr_dist
    df_filt = df[mask].copy()
    df_filt = df_filt.dropna(subset=[label_col])
    
    if len(df_filt) < 500:
        return None
    
    # Prepare features and labels
    X = df_filt[feature_cols].copy()
    y = df_filt[label_col].astype(int)
    
    # Handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Train/test split by year
    if 'date' not in df_filt.columns:
        df_filt['date'] = df_filt.index.date
    df_filt['year'] = pd.to_datetime(df_filt['date']).apply(lambda x: x.year)
    
    train_mask = df_filt['year'] < test_year
    test_mask = df_filt['year'] >= test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 200 or len(X_test) < 100:
        return None
    
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
    
    # Predictions on test set
    test_proba = rf.predict_proba(X_test)[:, 1]
    
    # For EV calculation, use average vwap_width_atr in test set
    avg_dist = df_filt.loc[test_mask, 'vwap_width_atr'].mean()
    rr = avg_dist / stop_atr
    breakeven = 1 / (1 + rr)
    
    # Raw stats
    raw_wr = y_test.mean()
    raw_ev = raw_wr * rr - (1 - raw_wr)
    
    results = {
        'stop_atr': stop_atr,
        'avg_dist_atr': avg_dist,
        'rr': rr,
        'breakeven': breakeven,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'raw_wr': raw_wr,
        'raw_ev': raw_ev,
    }
    
    # RF filtered at various thresholds
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
        rf_mask = test_proba >= thresh
        n = rf_mask.sum()
        if n >= 50:
            wr = y_test[rf_mask].mean()
            # Use actual avg distance for filtered trades
            avg_d = df_filt.loc[test_mask, 'vwap_width_atr'][rf_mask].mean()
            actual_rr = avg_d / stop_atr
            ev = wr * actual_rr - (1 - wr)
            results[f'rf{thresh}_n'] = n
            results[f'rf{thresh}_wr'] = wr
            results[f'rf{thresh}_avg_dist'] = avg_d
            results[f'rf{thresh}_rr'] = actual_rr
            results[f'rf{thresh}_ev'] = ev
        else:
            results[f'rf{thresh}_n'] = n
            results[f'rf{thresh}_wr'] = np.nan
            results[f'rf{thresh}_ev'] = np.nan
    
    # Feature importances
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    results['top_features'] = ', '.join(fi.head(5).index.tolist())
    results['feature_importances'] = fi.head(10).to_dict()
    
    return results, rf, X_test, y_test, test_proba


def main():
    print("="*80)
    print("RF GRID SEARCH: ATR Ranges × Stop Widths")
    print("="*80)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df):,} bars")
    
    # Generate labels
    print("\nGenerating labels...")
    config = LabelConfig(stop_atrs=STOP_ATRS)
    df = generate_labels(df, config)
    
    # Get feature columns
    feature_cols = get_feature_cols(df)
    print(f"\nUsing {len(feature_cols)} features:")
    print(f"  {feature_cols[:10]}...")
      # Check if vwap_width_atr is included
    if 'vwap_width_atr' in feature_cols:
        print("  [OK] vwap_width_atr included as feature")
    
    # Run grid search
    print(f"\nRunning grid: {len(ATR_RANGES)} ATR ranges × {len(STOP_ATRS)} stops = {len(ATR_RANGES)*len(STOP_ATRS)} combos")
    results = []
    
    total = len(ATR_RANGES) * len(STOP_ATRS)
    done = 0
    
    for atr_min, atr_max in ATR_RANGES:
        for stop_atr in STOP_ATRS:
            done += 1
            print(f"  [{done}/{total}] ATR {atr_min}-{atr_max}, Stop {stop_atr}...", end="", flush=True)
            
            result = run_rf_for_combo(df, atr_min, atr_max, stop_atr, feature_cols)
            
            if result:
                results.append(result)
                ev = result.get('rf0.5_ev', np.nan)
                ev_str = f"{ev:+.3f}" if not np.isnan(ev) else "N/A"
                print(f" Raw={result['raw_wr']:.1%}, RF0.5 EV={ev_str}")
            else:
                print(" skipped")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    output_path = Path(__file__).parent.parent / "data" / "rf_grid_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("POSITIVE EV SETUPS (RF ≥ 0.5 threshold)")
    print("="*80)
    
    positive = results_df[results_df['rf0.5_ev'] > 0].sort_values('rf0.5_ev', ascending=False)
    if len(positive) > 0:
        cols = ['atr_range', 'stop_atr', 'rr', 'breakeven', 'n_test', 
                'raw_wr', 'raw_ev', 'rf0.5_n', 'rf0.5_wr', 'rf0.5_ev']
        for _, row in positive.head(15).iterrows():
            print(f"ATR {row['atr_range']:<10} Stop {row['stop_atr']:.2f}  "
                  f"R:R {row['rr']:.2f}  BE {row['breakeven']:.1%}  "
                  f"N={row['n_test']:<5}  Raw {row['raw_wr']:.1%} ({row['raw_ev']:+.3f})  "
                  f"RF {row['rf0.5_wr']:.1%} ({row['rf0.5_ev']:+.3f}) n={row['rf0.5_n']:.0f}")
    else:
        print("No positive EV combinations found")
    
    print("\n" + "="*80)
    print("RAW POSITIVE EV (no RF filter)")
    print("="*80)
    raw_pos = results_df[results_df['raw_ev'] > 0].sort_values('raw_ev', ascending=False)
    if len(raw_pos) > 0:
        for _, row in raw_pos.head(10).iterrows():
            print(f"ATR {row['atr_range']:<10} Stop {row['stop_atr']:.2f}  "
                  f"R:R {row['rr']:.2f}  WR {row['raw_wr']:.1%}  EV {row['raw_ev']:+.3f}  N={row['n_test']}")
    else:
        print("No raw positive EV found")
    
    print("\n" + "="*80)
    print("TOP 10 BY RF 0.5 EV (regardless of sign)")
    print("="*80)
    top10 = results_df.dropna(subset=['rf0.5_ev']).sort_values('rf0.5_ev', ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"ATR {row['atr_range']:<10} Stop {row['stop_atr']:.2f}  "
              f"R:R {row['rr']:.2f}  RF WR {row['rf0.5_wr']:.1%}  EV {row['rf0.5_ev']:+.3f}  "
              f"Top: {row['top_features'][:50]}")


if __name__ == "__main__":
    main()
