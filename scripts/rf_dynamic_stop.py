# filepath: c:\Users\Administrator\ibkr8\scripts\rf_dynamic_stop.py
"""
RF analysis with DYNAMIC stop sized for 2:1 R:R.

Key insight: Instead of fixed 1.5 ATR stop, set stop = VWAP_distance / 2
This guarantees 2:1 R:R on every trade.

Question: Does the RF model help identify which 2:1 trades will hit target vs stop?
"""
print("="*60)
print("RANDOM FOREST WITH DYNAMIC 2:1 R:R STOP")
print("="*60)
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load raw data and recalculate outcomes with dynamic stops
print("\nLoading raw price data...")
df_prices = pd.read_csv('data/tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
print(f"Loaded {len(df_prices)} bars")

# Calculate indicators
from indicators import calc_all_indicators
df = calc_all_indicators(df_prices)
print(f"Calculated indicators")

# Filter to bars with meaningful extension (at least 0.5 ATR from VWAP)
min_extension = 0.5  # Minimum ATR from VWAP to consider a trade
df_ext = df[df['vwap_width_atr'] >= min_extension].copy()
print(f"Bars with extension >= {min_extension} ATR: {len(df_ext)}")

# Calculate dynamic stop for 2:1 R:R
# Stop distance = VWAP distance / 2
df_ext['dynamic_stop_atr'] = df_ext['vwap_width_atr'] / 2

print("\n" + "="*60)
print("DYNAMIC STOP DISTRIBUTION")
print("="*60)
print(f"VWAP distance (ATR): {df_ext['vwap_width_atr'].describe()}")
print(f"\nDynamic stop (ATR): {df_ext['dynamic_stop_atr'].describe()}")

# Simulate outcomes with dynamic stops
print("\n" + "="*60)
print("SIMULATING TRADES WITH DYNAMIC 2:1 STOPS")
print("="*60)

def simulate_dynamic_stop_trade(df_full, entry_idx, entry_row, max_bars=78):
    """
    Simulate a trade with dynamic stop sized for 2:1 R:R.
    
    Target: VWAP at entry
    Stop: entry_price +/- (vwap_distance / 2) in adverse direction
    """
    entry_price = entry_row['close']
    vwap_at_entry = entry_row['vwap']
    atr = entry_row['atr']
    vwap_dist = abs(entry_price - vwap_at_entry)
    stop_dist = vwap_dist / 2  # For 2:1 R:R
    
    # Determine direction
    is_long = entry_price < vwap_at_entry
    
    if is_long:
        target = vwap_at_entry
        stop = entry_price - stop_dist
    else:
        target = vwap_at_entry
        stop = entry_price + stop_dist
    
    # Get future bars
    try:
        entry_loc = df_full.index.get_loc(entry_idx)
    except:
        return None
    
    future_bars = df_full.iloc[entry_loc+1:entry_loc+1+max_bars]
    
    if len(future_bars) == 0:
        return None
    
    # Check each bar for target/stop hit
    for i, (bar_idx, bar) in enumerate(future_bars.iterrows()):
        if is_long:
            # Check stop first (more conservative)
            if bar['low'] <= stop:
                return {
                    'outcome': 'STOP',
                    'exit_price': stop,
                    'pnl': stop - entry_price,
                    'pnl_atr': (stop - entry_price) / atr,
                    'bars_held': i + 1,
                    'rr_achieved': -1.0  # Lost 1R
                }
            # Then check target
            if bar['high'] >= target:
                return {
                    'outcome': 'TARGET',
                    'exit_price': target,
                    'pnl': target - entry_price,
                    'pnl_atr': (target - entry_price) / atr,
                    'bars_held': i + 1,
                    'rr_achieved': 2.0  # Won 2R
                }
        else:  # Short
            # Check stop first
            if bar['high'] >= stop:
                return {
                    'outcome': 'STOP',
                    'exit_price': stop,
                    'pnl': entry_price - stop,
                    'pnl_atr': (entry_price - stop) / atr,
                    'bars_held': i + 1,
                    'rr_achieved': -1.0
                }
            # Then check target
            if bar['low'] <= target:
                return {
                    'outcome': 'TARGET',
                    'exit_price': target,
                    'pnl': entry_price - target,
                    'pnl_atr': (entry_price - target) / atr,
                    'bars_held': i + 1,
                    'rr_achieved': 2.0
                }
    
    # EOD exit
    exit_bar = future_bars.iloc[-1]
    if is_long:
        pnl = exit_bar['close'] - entry_price
    else:
        pnl = entry_price - exit_bar['close']
    
    return {
        'outcome': 'EOD',
        'exit_price': exit_bar['close'],
        'pnl': pnl,
        'pnl_atr': pnl / atr,
        'bars_held': len(future_bars),
        'rr_achieved': pnl / stop_dist if stop_dist > 0 else 0
    }

# Sample trades (every 6 bars = 30 min spacing to avoid overlap)
sample_indices = df_ext.index[::6]
print(f"Simulating {len(sample_indices)} trades...")

results = []
for idx in sample_indices:
    row = df_ext.loc[idx]
    result = simulate_dynamic_stop_trade(df, idx, row)
    if result:
        result['entry_time'] = idx
        result['vwap_width_atr'] = row['vwap_width_atr']
        result['stop_atr'] = row['dynamic_stop_atr']
        result['direction'] = 'LONG' if row['close'] < row['vwap'] else 'SHORT'
        # Add features for ML
        for col in ['rsi', 'rsi_extremity', 'bb_pct', 'rel_vol', 'vwap_helping',
                    'reversal_wick', 'reversal_close_position', 'bar_range_atr',
                    'consecutive_shrinking', 'momentum_divergence_5', 'extension_velocity_5',
                    'vol_declining', 'in_sweet_spot', 'in_tradeable_zone', 'over_extended']:
            if col in row.index:
                result[col] = row[col]
        results.append(result)

df_results = pd.DataFrame(results)
print(f"Completed {len(df_results)} trades")

# Analyze by VWAP extension zone
print("\n" + "="*60)
print("WIN RATE BY VWAP EXTENSION (with 2:1 dynamic stop)")
print("="*60)

# Filter to definitive outcomes
df_def = df_results[df_results['outcome'].isin(['TARGET', 'STOP'])].copy()
print(f"Definitive outcomes: {len(df_def)}")

zones = [
    (0.5, 1.0, '0.5-1.0 ATR'),
    (1.0, 1.5, '1.0-1.5 ATR (sweet)'),
    (1.5, 2.0, '1.5-2.0 ATR'),
    (2.0, 2.5, '2.0-2.5 ATR'),
    (2.5, 3.0, '2.5-3.0 ATR'),
    (3.0, 4.0, '3.0-4.0 ATR'),
    (4.0, 10.0, '>4.0 ATR'),
]

print(f"\n{'Zone':<20} {'Trades':>8} {'Wins':>8} {'WR':>8} {'Stop ATR':>10} {'Need WR':>8} {'EV/R':>8}")
print("-" * 80)

for low, high, label in zones:
    mask = (df_def['vwap_width_atr'] >= low) & (df_def['vwap_width_atr'] < high)
    sub = df_def[mask]
    if len(sub) > 10:
        wins = (sub['outcome'] == 'TARGET').sum()
        wr = 100 * wins / len(sub)
        avg_stop = sub['stop_atr'].mean()
        # With 2:1 R:R, need 33.3% WR to break even
        # EV = WR * 2 - (1-WR) * 1 = 3*WR - 1
        ev_per_r = (wr/100) * 2 - (1 - wr/100) * 1
        status = "✓" if wr > 33.3 else "✗"
        print(f"{label:<20} {len(sub):>8} {wins:>8} {wr:>7.1f}% {avg_stop:>9.2f} {'33.3%':>8} {ev_per_r:>+7.2f}R {status}")

# Overall stats
print("\n" + "="*60)
print("OVERALL RESULTS (2:1 Dynamic Stop)")
print("="*60)
total_wins = (df_def['outcome'] == 'TARGET').sum()
total_wr = 100 * total_wins / len(df_def)
print(f"Total trades: {len(df_def)}")
print(f"Win rate: {total_wr:.1f}%")
print(f"Required WR for breakeven (2:1): 33.3%")

ev = (total_wr/100) * 2 - (1 - total_wr/100) * 1
print(f"Expected Value: {ev:+.3f}R per trade")

# Actual P&L (sum of rr_achieved)
total_r = df_def['rr_achieved'].sum()
print(f"Total R earned: {total_r:+.1f}R")

# Now run RF
print("\n" + "="*60)
print("RANDOM FOREST ON 2:1 DYNAMIC STOP TRADES")
print("="*60)

feature_cols = ['vwap_width_atr', 'rsi', 'rsi_extremity', 'bb_pct', 'rel_vol',
                'vwap_helping', 'reversal_wick', 'reversal_close_position', 
                'bar_range_atr', 'consecutive_shrinking', 'momentum_divergence_5',
                'extension_velocity_5', 'vol_declining']
available = [c for c in feature_cols if c in df_def.columns]
print(f"Features: {len(available)}")

X = df_def[available].fillna(df_def[available].median())
y = (df_def['outcome'] == 'TARGET').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=30,
                            random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f"\nROC AUC: {auc:.3f}")
print(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Trading simulation
print("\n" + "="*60)
print("TRADING SIMULATION WITH RF FILTER")
print("="*60)

test_df = df_def.loc[X_test.index].copy()
test_df['pred_prob'] = y_prob

baseline_wr = 100 * (test_df['outcome'] == 'TARGET').sum() / len(test_df)
baseline_r = test_df['rr_achieved'].sum()
print(f"Baseline: {len(test_df)} trades, WR={baseline_wr:.1f}%, Total R={baseline_r:+.1f}")

print(f"\n{'Thresh':<8} {'Trades':>8} {'Wins':>8} {'WR':>8} {'Total R':>10} {'EV/trade':>10}")
print("-" * 60)
for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    sub = test_df[test_df['pred_prob'] >= thresh]
    if len(sub) >= 10:
        wins = (sub['outcome'] == 'TARGET').sum()
        wr = 100 * wins / len(sub)
        total_r = sub['rr_achieved'].sum()
        ev = total_r / len(sub)
        status = "✓" if wr > 33.3 else ""
        print(f"{thresh:<8} {len(sub):>8} {wins:>8} {wr:>7.1f}% {total_r:>+9.1f}R {ev:>+9.2f}R {status}")

# By zone + RF
print("\n" + "="*60)
print("SWEET SPOT (1-1.5 ATR) + RF FILTER")
print("="*60)
sweet = test_df[(test_df['vwap_width_atr'] >= 1.0) & (test_df['vwap_width_atr'] < 1.5)]
if len(sweet) > 5:
    print(f"Sweet spot trades: {len(sweet)}")
    for thresh in [0.3, 0.4, 0.5]:
        sub = sweet[sweet['pred_prob'] >= thresh]
        if len(sub) >= 5:
            wins = (sub['outcome'] == 'TARGET').sum()
            wr = 100 * wins / len(sub)
            total_r = sub['rr_achieved'].sum()
            print(f"  RF >= {thresh}: {len(sub)} trades, WR={wr:.1f}%, R={total_r:+.1f}")

# Feature importance
print("\n" + "="*60)
print("TOP FEATURES")
print("="*60)
imp = pd.DataFrame({'feature': available, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for _, row in imp.head(8).iterrows():
    bar = '*' * int(row['importance'] * 40)
    print(f"{row['feature']:<25} {row['importance']:.3f} {bar}")

print("\n" + "="*60)
print("KEY INSIGHT")
print("="*60)
print("""
With DYNAMIC 2:1 stops (stop = VWAP_dist / 2):
- Every trade has exactly 2:1 R:R by construction
- Need 33.3% win rate to break even
- Question: Can RF identify trades above 33.3% WR?
""")
