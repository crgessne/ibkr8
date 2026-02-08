"""
RF with dynamic label based on stop width.

Win condition: Price touches VWAP before hitting stop (within stop_atr window), before EOD
Loss condition: Price hits stop OR EOD without touching VWAP

Key: vwap_width_atr is EXCLUDED from training features so RF learns 
quality signals, not just "closer to VWAP = better"

Usage:
    python rf_dynamic_label.py --stop_atr 2.0 --zone_min 1.5 --zone_max 2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import warnings
warnings.filterwarnings('ignore')


def compute_trade_outcome(df: pd.DataFrame, stop_atr: float) -> pd.Series:
    """
    For each bar, compute whether a VWAP reversion trade would WIN or LOSE.
    
    WIN: Price touches VWAP before hitting stop, before EOD
    LOSE: Price hits stop OR EOD comes without VWAP touch
    
    Args:
        df: DataFrame with 'close', 'high', 'low', 'vwap', 'atr', 'date' columns
        stop_atr: Stop distance in ATR units
        
    Returns:
        Series with 1 (win) or 0 (loss) for each bar
    """
    results = pd.Series(index=df.index, dtype=float)
    results[:] = np.nan
    
    # Pre-compute arrays for speed
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    vwap_arr = df['vwap'].values
    atr_arr = df['atr'].values
    date_arr = df['date'].values
    indices = df.index.tolist()
    
    n = len(df)
    result_arr = np.full(n, np.nan)
    
    i = 0
    while i < n:
        current_date = date_arr[i]
        
        # Find end of this day
        day_end = i
        while day_end < n and date_arr[day_end] == current_date:
            day_end += 1
        
        # Process all bars for this day
        for j in range(i, day_end):
            entry_price = close_arr[j]
            vwap = vwap_arr[j]
            atr = atr_arr[j]
            
            if np.isnan(atr) or atr <= 0 or np.isnan(vwap):
                continue
            
            is_long = entry_price < vwap
            stop_dist = stop_atr * atr
            
            if is_long:
                stop_price = entry_price - stop_dist
                target_price = vwap
            else:
                stop_price = entry_price + stop_dist
                target_price = vwap
            
            # Look at remaining bars in the day
            if j + 1 >= day_end:
                continue  # Last bar, can't evaluate
            
            hit_target = False
            hit_stop = False
            
            for k in range(j + 1, day_end):
                if is_long:
                    if low_arr[k] <= stop_price:
                        hit_stop = True
                        break
                    if high_arr[k] >= target_price:
                        hit_target = True
                        break
                else:
                    if high_arr[k] >= stop_price:
                        hit_stop = True
                        break
                    if low_arr[k] <= target_price:
                        hit_target = True
                        break
            
            result_arr[j] = 1 if hit_target else 0
        
        i = day_end
    
    return pd.Series(result_arr, index=df.index)


def main():
    parser = argparse.ArgumentParser(description='RF with dynamic stop-based labels')
    parser.add_argument('--stop_atr', type=float, default=2.0, 
                        help='Stop width in ATR (default: 2.0)')
    parser.add_argument('--zone_min', type=float, default=1.5,
                        help='Minimum VWAP distance in ATR (default: 1.5)')
    parser.add_argument('--zone_max', type=float, default=2.0,
                        help='Maximum VWAP distance in ATR (default: 2.0)')
    parser.add_argument('--test_year', type=int, default=2024,
                        help='First year of test set (default: 2024)')
    args = parser.parse_args()
    
    print(f"=" * 70)
    print(f"RF VWAP Reversion with Dynamic Labels")
    print(f"=" * 70)
    print(f"Stop width: {args.stop_atr} ATR")
    print(f"Zone: {args.zone_min}-{args.zone_max} ATR from VWAP")
    print(f"Test data: {args.test_year}+")
    print(f"=" * 70)
      # Load data
    data_path = Path(__file__).parent.parent / "data" / "tsla_5min_10years_indicators.csv"
    print(f"\nLoading {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['time'], index_col='time')
    print(f"Loaded {len(df):,} bars")
    
    # Clean inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Add date column
    if df.index.tz is not None:
        df['date'] = df.index.tz_localize(None).date
    else:
        df['date'] = df.index.date
    
    # Filter to zone
    print(f"\nFiltering to {args.zone_min}-{args.zone_max} ATR zone...")
    zone_mask = (df['vwap_width_atr'] >= args.zone_min) & (df['vwap_width_atr'] < args.zone_max)
    df_zone = df[zone_mask].copy()
    print(f"Bars in zone: {len(df_zone):,}")
    
    # Compute labels
    print(f"\nComputing trade outcomes with {args.stop_atr} ATR stop...")
    df_zone['label'] = compute_trade_outcome(df_zone, args.stop_atr)
    
    # Drop NaN labels
    df_zone = df_zone.dropna(subset=['label'])
    print(f"Bars with valid labels: {len(df_zone):,}")
    
    # Win rate
    raw_wr = df_zone['label'].mean()
    print(f"Raw win rate: {raw_wr:.1%}")
    
    # R:R calculation - profit is distance to VWAP, risk is stop
    mid_zone = (args.zone_min + args.zone_max) / 2
    rr = mid_zone / args.stop_atr
    breakeven_wr = 1 / (1 + rr)
    print(f"R:R ratio: {rr:.2f}:1")
    print(f"Breakeven WR: {breakeven_wr:.1%}")
    
    # EV calculation
    ev_r = raw_wr * rr - (1 - raw_wr)
    status = "POSITIVE" if ev_r > 0 else "negative"
    print(f"Raw EV per trade: {ev_r:+.3f}R ({status})")
    
    # === FEATURE SELECTION ===
    # Exclude vwap_width_atr and related distance features
    exclude_features = [
        'label', 'date',
        # Distance features - we don't want RF to learn "closer = better"
        'vwap_width_atr', 'vwap_width_pct', 'price_to_vwap', 'price_to_vwap_atr', 
        'price_to_vwap_pct', 'profit_potential_atr', 'vwap_dist_atr', 'vwap_dist_pct',
        # R:R features (derived from distance)
        'rr_0_5atr', 'rr_1_0atr', 'rr_1_5atr', 'rr_2_0atr', 'rr_2_5atr', 'rr_3_0atr',
        'long_rr_0_5atr', 'long_rr_1_0atr', 'long_rr_1_5atr', 'long_rr_2_0atr',
        'short_rr_0_5atr', 'short_rr_1_0atr', 'short_rr_1_5atr', 'short_rr_2_0atr',
        'avg_rr',
        # Zone booleans (derived from distance)
        'in_sweet_spot', 'in_tradeable_zone', 'over_extended', 'extreme_extension',
        # Non-features
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'atr',
        'bb_upper', 'bb_lower', 'bb_middle',
        'price_below_vwap',
    ]
    
    feature_cols = [c for c in df_zone.columns if c not in exclude_features 
                    and not c.startswith('long_rr') and not c.startswith('short_rr')
                    and df_zone[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"\nFeatures ({len(feature_cols)}):")
    for f in sorted(feature_cols):
        print(f"  - {f}")
    
    # Prepare data
    X = df_zone[feature_cols].copy()
    y = df_zone['label'].astype(int)
    
    # Handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Train/test split by date
    df_zone['year'] = pd.to_datetime(df_zone['date']).apply(lambda x: x.year)
    train_mask = df_zone['year'] < args.test_year
    test_mask = df_zone['year'] >= args.test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTrain: {len(X_train):,} samples ({df_zone[train_mask]['year'].min()}-{df_zone[train_mask]['year'].max()})")
    print(f"Test:  {len(X_test):,} samples ({df_zone[test_mask]['year'].min()}-{df_zone[test_mask]['year'].max()})")
    print(f"Train WR: {y_train.mean():.1%}")
    print(f"Test WR:  {y_test.mean():.1%}")
    
    # Train RF
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Predictions
    train_proba = rf.predict_proba(X_train)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]
    
    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 70)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # Results by threshold
    print("\n" + "=" * 70)
    print("RESULTS BY RF PROBABILITY THRESHOLD (TEST DATA)")
    print("=" * 70)
    print(f"{'Threshold':<12} {'N Trades':<12} {'Win Rate':<12} {'vs Raw':<12} {'EV (R)':<12} {'Status'}")
    print("-" * 70)
    
    raw_test_wr = y_test.mean()
    
    for thresh in [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = test_proba >= thresh
        n = mask.sum()
        if n < 50:
            continue
        wr = y_test[mask].mean()
        lift = wr - raw_test_wr
        ev = wr * rr - (1 - wr)
        status = "+EV" if ev > 0 else ""
        print(f"{thresh:<12.2f} {n:<12,} {wr:<12.1%} {lift:+11.1%} {ev:+11.3f} {status}")
    
    # Confusion matrix at 0.5
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (threshold=0.5)")
    print("=" * 70)
    y_pred = (test_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                  Predicted")
    print(f"                  Loss    Win")
    print(f"Actual Loss      {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual Win       {cm[1,0]:5d}  {cm[1,1]:5d}")
      # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_thresh = 0.5
    best_mask = test_proba >= best_thresh
    best_wr = y_test[best_mask].mean() if best_mask.sum() > 0 else 0
    best_ev = best_wr * rr - (1 - best_wr)
    
    print(f"Zone: {args.zone_min}-{args.zone_max} ATR")
    print(f"Stop: {args.stop_atr} ATR")
    print(f"R:R: {rr:.2f}:1 (breakeven @ {breakeven_wr:.1%})")
    print(f"Raw test WR: {raw_test_wr:.1%} -> EV = {raw_test_wr * rr - (1 - raw_test_wr):+.3f}R")
    print(f"RF>=0.5 WR: {best_wr:.1%} -> EV = {best_ev:+.3f}R")
    
    if best_ev > 0:
        print(f"\n** POSITIVE EXPECTED VALUE with RF filtering! **")
    else:
        print(f"\nStill negative EV - need tighter filters or different approach")


if __name__ == "__main__":
    main()
