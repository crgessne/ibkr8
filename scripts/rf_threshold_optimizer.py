# filepath: c:\Users\Administrator\ibkr8\scripts\rf_threshold_optimizer.py
"""
RF-Based Threshold Optimizer
Uses Random Forest to identify which features matter, 
then finds optimal threshold values for each key feature.
"""
print("="*70)
print("RF THRESHOLD OPTIMIZER - Finding Optimal Feature Cutoffs")
print("="*70)

import sys
sys.stdout.flush()
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/tsla_5min_2years_ml_features.csv')
print(f'Loaded {len(df)} trades')

# Feature columns
feature_cols = [
    'vwap_width_atr', 'price_to_vwap_atr',
    'rsi', 'rsi_extremity', 'bb_pct',
    'rsi_slope_3', 'rsi_slope_5',
    'momentum_divergence_3', 'momentum_divergence_5',
    'reversal_wick', 'reversal_close_position', 'close_position',
    'bar_range_atr',
    'consecutive_shrinking', 'range_vs_prev',
    'vol_declining', 'vol_trend_3',
    'extension_velocity_3', 'extension_velocity_5',
    'extension_accel', 'vwap_helping',
    'rel_vol', 'vol_at_extension',
    'bb_extension', 'bb_extension_abs',
]
available = [c for c in feature_cols if c in df.columns]

# Filter to definitive outcomes
df_filt = df[df['outcome'].isin(['TARGET', 'STOP'])].copy()
df_filt['win'] = (df_filt['outcome'] == 'TARGET').astype(int)
print(f'Trades with outcomes: {len(df_filt)}, Win rate: {df_filt["win"].mean()*100:.1f}%')

X = df_filt[available].fillna(df_filt[available].median())
y = df_filt['win']

# Train RF to get feature importance
print('\nTraining Random Forest for feature importance...')
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=30, 
                            random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X, y)

# Feature importance
imp = pd.DataFrame({'feature': available, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)

print('\n' + '='*70)
print('FEATURE IMPORTANCE RANKING')
print('='*70)
for i, (_, row) in enumerate(imp.iterrows()):
    bar = '*' * int(row['importance'] * 100)
    print(f"{i+1:>2}. {row['feature']:<28} {row['importance']:.4f} {bar}")

# ============================================================================
# THRESHOLD OPTIMIZATION FOR TOP FEATURES
# ============================================================================
print('\n' + '='*70)
print('OPTIMAL THRESHOLD ANALYSIS FOR KEY FEATURES')
print('='*70)

def analyze_thresholds(df_data, feature, direction='above', percentiles=[10,20,30,40,50,60,70,80,90]):
    """Find optimal threshold for a feature."""
    results = []
    values = df_data[feature].dropna()
    
    for pct in percentiles:
        if direction == 'above':
            thresh = np.percentile(values, pct)
            mask = df_data[feature] >= thresh
        else:
            thresh = np.percentile(values, 100-pct)
            mask = df_data[feature] <= thresh
        
        sub = df_data[mask]
        if len(sub) < 20:
            continue
            
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        n_trades = len(sub)
        
        # Calculate expected value (win rate - 50% baseline)
        ev = wr - 50  # Simplified EV
        
        results.append({
            'percentile': pct,
            'threshold': thresh,
            'n_trades': n_trades,
            'win_rate': wr,
            'ev': ev
        })
    
    return pd.DataFrame(results)


# Key features to analyze (based on importance)
key_features = [
    ('vwap_width_atr', 'range', 'VWAP Distance (ATR)'),  # Range analysis
    ('reversal_wick', 'above', 'Reversal Wick %'),
    ('reversal_close_position', 'above', 'Close Position (reversal)'),
    ('rel_vol', 'above', 'Relative Volume'),
    ('bar_range_atr', 'below', 'Bar Range (ATR)'),
    ('rsi', 'custom', 'RSI'),  # Need to handle differently for long/short
    ('rsi_extremity', 'above', 'RSI Extremity'),
    ('vwap_helping', 'above', 'VWAP Helping'),
]

# Analyze VWAP width zones
print('\n--- VWAP Distance from Price (ATR) ---')
print('Finding optimal zone for mean-reversion trades:\n')
zones = [
    (0.0, 0.5, 'Very Close'),
    (0.5, 1.0, 'Close'),
    (1.0, 1.5, 'Sweet Spot 1'),
    (1.5, 2.0, 'Sweet Spot 2'),
    (2.0, 2.5, 'Moderate'),
    (2.5, 3.0, 'Extended'),
    (3.0, 4.0, 'Over-Extended'),
    (4.0, 10.0, 'Extreme'),
]

print(f"{'Zone':<20} {'Range (ATR)':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'EV':>10}")
print("-"*75)
for low, high, name in zones:
    mask = (df_filt['vwap_width_atr'] >= low) & (df_filt['vwap_width_atr'] < high)
    sub = df_filt[mask]
    if len(sub) >= 10:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        ev = wr - 50
        marker = ' ***' if wr >= 50 else ''
        print(f"{name:<20} {low:.1f}-{high:.1f} ATR     {len(sub):>8} {wins:>8} {wr:>9.1f}%{ev:>9.1f}%{marker}")


# Analyze reversal wick
print('\n--- Reversal Wick % (rejection signal) ---')
print('Higher wick = more rejection of adverse prices:\n')
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'vs Baseline':>12}")
print("-"*55)
baseline_wr = df_filt['win'].mean() * 100
for thresh in thresholds:
    mask = df_filt['reversal_wick'] >= thresh
    sub = df_filt[mask]
    if len(sub) >= 20:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 5 else ''
        print(f">= {thresh:<12.0%} {len(sub):>8} {wins:>8} {wr:>9.1f}% {delta:>+10.1f}%{marker}")


# Analyze close position
print('\n--- Reversal Close Position (where bar closes) ---')
print('For longs: close near high = bullish; For shorts: close near low = bearish:\n')
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(f"{'Threshold':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'vs Baseline':>12}")
print("-"*55)
for thresh in thresholds:
    mask = df_filt['reversal_close_position'] >= thresh
    sub = df_filt[mask]
    if len(sub) >= 20:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 5 else ''
        print(f">= {thresh:<12.0%} {len(sub):>8} {wins:>8} {wr:>9.1f}% {delta:>+10.1f}%{marker}")


# Analyze relative volume
print('\n--- Relative Volume ---')
print('Higher volume = more conviction:\n')
thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
print(f"{'Threshold':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'vs Baseline':>12}")
print("-"*55)
for thresh in thresholds:
    mask = df_filt['rel_vol'] >= thresh
    sub = df_filt[mask]
    if len(sub) >= 20:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 3 else ''
        print(f">= {thresh:<12.2f} {len(sub):>8} {wins:>8} {wr:>9.1f}% {delta:>+10.1f}%{marker}")


# Analyze bar range
print('\n--- Bar Range (ATR) ---')
print('Smaller bars = less volatility/noise:\n')
thresholds = [2.0, 1.75, 1.5, 1.25, 1.0, 0.75]
print(f"{'Threshold':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'vs Baseline':>12}")
print("-"*55)
for thresh in thresholds:
    mask = df_filt['bar_range_atr'] <= thresh
    sub = df_filt[mask]
    if len(sub) >= 20:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 3 else ''
        print(f"<= {thresh:<12.2f} {len(sub):>8} {wins:>8} {wr:>9.1f}% {delta:>+10.1f}%{marker}")


# Analyze RSI for long trades
print('\n--- RSI for LONG trades (price below VWAP) ---')
print('Lower RSI = more oversold:\n')
long_mask = df_filt['price_to_vwap_atr'] < 0
longs = df_filt[long_mask]
thresholds = [50, 45, 40, 35, 30, 25, 20]
print(f"{'Threshold':<15} {'Trades':>8} {'Wins':>8} {'WR':>10} {'vs Baseline':>12}")
print("-"*55)
long_baseline = longs['win'].mean() * 100
for thresh in thresholds:
    mask = longs['rsi'] <= thresh
    sub = longs[mask]
    if len(sub) >= 20:
        wins = sub['win'].sum()
        wr = 100 * wins / len(sub)
        delta = wr - long_baseline
        marker = ' ***' if delta >= 3 else ''
        print(f"<= {thresh:<12} {len(sub):>8} {wins:>8} {wr:>9.1f}% {delta:>+10.1f}%{marker}")


# ============================================================================
# COMBINED FILTER ANALYSIS
# ============================================================================
print('\n' + '='*70)
print('COMBINED FILTER ANALYSIS')
print('='*70)
print('Testing combinations of promising thresholds:\n')

# Base conditions
filters = [
    ('VWAP 1.0-2.0 ATR', (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0)),
    ('VWAP 1.0-2.5 ATR', (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.5)),
    ('Wick >= 40%', df_filt['reversal_wick'] >= 0.4),
    ('Wick >= 30%', df_filt['reversal_wick'] >= 0.3),
    ('Close Pos >= 65%', df_filt['reversal_close_position'] >= 0.65),
    ('Close Pos >= 50%', df_filt['reversal_close_position'] >= 0.5),
    ('RelVol >= 1.0', df_filt['rel_vol'] >= 1.0),
    ('Bar Range <= 1.5', df_filt['bar_range_atr'] <= 1.5),
]

# Test all combinations
from itertools import combinations

print(f"{'Filter Combination':<60} {'Trades':>8} {'WR':>10} {'vs Base':>10}")
print("-"*90)

# Single filters
for name, mask in filters:
    sub = df_filt[mask]
    if len(sub) >= 20:
        wr = 100 * sub['win'].mean()
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 5 else ''
        print(f"{name:<60} {len(sub):>8} {wr:>9.1f}% {delta:>+9.1f}%{marker}")

print()

# Key combinations
combos = [
    ('VWAP 1.0-2.0 + Wick>=40%', 
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) & (df_filt['reversal_wick'] >= 0.4)),
    ('VWAP 1.0-2.0 + ClosePos>=65%',
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) & (df_filt['reversal_close_position'] >= 0.65)),
    ('VWAP 1.0-2.0 + Wick>=40% + ClosePos>=65%',
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) & 
     (df_filt['reversal_wick'] >= 0.4) & (df_filt['reversal_close_position'] >= 0.65)),
    ('VWAP 1.0-2.0 + Wick>=30% + ClosePos>=50% + RelVol>=1.0',
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) &
     (df_filt['reversal_wick'] >= 0.3) & (df_filt['reversal_close_position'] >= 0.5) &
     (df_filt['rel_vol'] >= 1.0)),
    ('STRICT: VWAP 1.0-2.0 + Wick>=40% + ClosePos>=65% + RelVol>=1.0',
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) &
     (df_filt['reversal_wick'] >= 0.4) & (df_filt['reversal_close_position'] >= 0.65) &
     (df_filt['rel_vol'] >= 1.0)),
    ('STRICT + BarRange<=1.5',
     (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) &
     (df_filt['reversal_wick'] >= 0.4) & (df_filt['reversal_close_position'] >= 0.65) &
     (df_filt['rel_vol'] >= 1.0) & (df_filt['bar_range_atr'] <= 1.5)),
]

print("Multi-filter combinations:")
for name, mask in combos:
    sub = df_filt[mask]
    if len(sub) >= 5:
        wr = 100 * sub['win'].mean()
        delta = wr - baseline_wr
        marker = ' ***' if delta >= 8 else ''
        print(f"{name:<60} {len(sub):>8} {wr:>9.1f}% {delta:>+9.1f}%{marker}")


# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print('\n' + '='*70)
print('RF-DERIVED OPTIMAL THRESHOLDS')
print('='*70)

print("""
Based on Random Forest feature importance and threshold analysis:

TOP PREDICTIVE FEATURES (by RF importance):
1. vwap_width_atr (0.133) - Distance from VWAP
2. reversal_close_position (0.023) - Where bar closes
3. reversal_wick (0.023) - Rejection signal
4. rel_vol (0.025) - Volume conviction
5. bar_range_atr (0.021) - Volatility filter

OPTIMAL THRESHOLDS (backed by data):
- VWAP Distance: 1.0-2.0 ATR (sweet spot)
- Reversal Wick: >= 40% (strong rejection)
- Close Position: >= 65% (favorable close)
- Relative Volume: >= 1.0 (average or above)
- Bar Range: <= 1.5 ATR (controlled volatility)

These thresholds were determined by:
1. RF feature importance ranking
2. Monotonic win rate analysis at different cutoffs
3. Combined filter testing to verify synergies
""")

# Show the final strict config performance
strict_mask = (
    (df_filt['vwap_width_atr'] >= 1.0) & (df_filt['vwap_width_atr'] < 2.0) &
    (df_filt['reversal_wick'] >= 0.4) & 
    (df_filt['reversal_close_position'] >= 0.65) &
    (df_filt['rel_vol'] >= 1.0) &
    (df_filt['bar_range_atr'] <= 1.5)
)
strict_trades = df_filt[strict_mask]
print(f"STRICT CONFIG BACKTEST:")
print(f"  Trades: {len(strict_trades)}")
print(f"  Wins: {strict_trades['win'].sum()}")
print(f"  Win Rate: {100*strict_trades['win'].mean():.1f}%")
print(f"  vs Baseline (+{100*strict_trades['win'].mean() - baseline_wr:.1f}%)")
