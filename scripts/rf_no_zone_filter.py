# filepath: c:\Users\Administrator\ibkr8\scripts\rf_no_zone_filter.py
"""RF analysis WITHOUT ATR zone filtering - let RF discover zone importance."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys

print('=== RF ANALYSIS - NO ATR ZONE FILTERING ===')
print('Letting RF discover zone importance on its own\n')
sys.stdout.flush()

# Load data
df = pd.read_csv('C:/Users/Administrator/ibkr8/data/tsla_5min_10years_indicators.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df.set_index('time', inplace=True)
df['date'] = df.index.date
print(f'Loaded {len(df):,} bars')
sys.stdout.flush()

# Pre-compute targets for ALL bars once
print('Computing targets...')
sys.stdout.flush()

df['target'] = np.nan
dates = df['date'].unique()

for i, date in enumerate(dates):
    if i % 500 == 0:
        print(f'  Day {i}/{len(dates)}')
        sys.stdout.flush()
    
    day_mask = df['date'] == date
    day_df = df[day_mask]
    day_indices = day_df.index.tolist()
    
    for j, idx in enumerate(day_indices[:-1]):  # Skip last bar of day
        future = day_df.iloc[j+1:]
        entry_close = day_df.loc[idx, 'close']
        entry_vwap = day_df.loc[idx, 'vwap']
        
        if entry_close < entry_vwap:
            reverted = 1 if future['high'].max() >= entry_vwap else 0
        else:
            reverted = 1 if future['low'].min() <= entry_vwap else 0
        
        df.loc[idx, 'target'] = reverted

print('Targets computed!\n')
sys.stdout.flush()

# Features - include vwap_width_atr so RF can learn zone importance
feature_cols = [
    'vwap_width_atr',           # ATR zone - let RF discover importance
    'extension_velocity_3',
    'extension_velocity_5', 
    'rsi_extremity',
    'rsi',
    'momentum_divergence_3',
    'momentum_divergence_5',
    'vwap_slope_5',
    'bar_range_atr',
    'vol_at_extension',
    'bb_extension_abs',
]

# Drop NaN
df_valid = df.dropna(subset=['target'] + feature_cols).copy()
print(f'Valid rows: {len(df_valid):,}')
sys.stdout.flush()

# Time-based split: train 2016-2023, test 2024+
train_mask = df_valid.index.year <= 2023
test_mask = df_valid.index.year >= 2024

X_train = df_valid.loc[train_mask, feature_cols]
y_train = df_valid.loc[train_mask, 'target']
X_test = df_valid.loc[test_mask, feature_cols]
y_test = df_valid.loc[test_mask, 'target']

print(f'Train: {len(X_train):,} rows (2016-2023)')
print(f'Test: {len(X_test):,} rows (2024-2026)')
print(f'Train WR: {y_train.mean()*100:.1f}%')
print(f'Test WR: {y_test.mean()*100:.1f}%')
sys.stdout.flush()

# Train RF on ALL data (no zone filtering)
print('\n=== TRAINING RF ON ALL DATA (NO ZONE FILTER) ===')
sys.stdout.flush()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Feature importance
print('\n=== FEATURE IMPORTANCE (ALL DATA) ===')
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance.iterrows():
    print(f'  {row["feature"]:25s}: {row["importance"]:.3f}')
sys.stdout.flush()

# Predictions on test set
test_proba = rf.predict_proba(X_test)[:, 1]
df_test = df_valid.loc[test_mask].copy()
df_test['rf_proba'] = test_proba

# Overall results
print('\n=== OVERALL TEST RESULTS ===')
raw_wr = y_test.mean() * 100
print(f'Raw win rate (all test): {raw_wr:.1f}% (n={len(y_test):,})')

for thresh in [0.4, 0.5, 0.6, 0.7]:
    sel = df_test[df_test['rf_proba'] >= thresh]
    if len(sel) > 0:
        wr = sel['target'].mean() * 100
        print(f'RF >= {thresh}: {wr:.1f}% WR (n={len(sel):,}, lift={wr-raw_wr:+.1f}%)')
sys.stdout.flush()

# Now show results BY ATR ZONE (but model trained on all)
print('\n=== TEST RESULTS BY ATR ZONE (MODEL TRAINED ON ALL) ===')
print(f'{"Zone":>12} | {"Raw WR":>8} | {"RF>=0.5 WR":>10} | {"N":>8} | {"Lift":>6}')
print('-' * 60)
sys.stdout.flush()

bands = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
]

for lo, hi in bands:
    zone_mask = (df_test['vwap_width_atr'] >= lo) & (df_test['vwap_width_atr'] < hi)
    zone_df = df_test[zone_mask]
    
    if len(zone_df) < 50:
        continue
    
    raw_wr_zone = zone_df['target'].mean() * 100
    
    # RF selection within zone
    rf_sel = zone_df[zone_df['rf_proba'] >= 0.5]
    rf_wr = rf_sel['target'].mean() * 100 if len(rf_sel) > 0 else 0
    lift = rf_wr - raw_wr_zone if len(rf_sel) > 0 else 0
    
    print(f'{lo:.1f}-{hi:.1f} ATR | {raw_wr_zone:7.1f}% | {rf_wr:9.1f}% | {len(rf_sel):>8,} | {lift:+5.1f}%')
sys.stdout.flush()

# Show what zones RF tends to select
print('\n=== RF SELECTION DISTRIBUTION BY ZONE ===')
print('(What zones does RF prefer when proba >= 0.5?)')
rf_selected = df_test[df_test['rf_proba'] >= 0.5]
print(f'Total RF selected: {len(rf_selected):,}')
sys.stdout.flush()

for lo, hi in bands:
    zone_mask = (rf_selected['vwap_width_atr'] >= lo) & (rf_selected['vwap_width_atr'] < hi)
    n_zone = zone_mask.sum()
    pct = 100 * n_zone / len(rf_selected) if len(rf_selected) > 0 else 0
    
    # Compare to baseline distribution
    base_zone_mask = (df_test['vwap_width_atr'] >= lo) & (df_test['vwap_width_atr'] < hi)
    base_pct = 100 * base_zone_mask.sum() / len(df_test) if len(df_test) > 0 else 0
    
    diff = pct - base_pct
    print(f'  {lo:.1f}-{hi:.1f} ATR: {pct:5.1f}% of selections (vs {base_pct:5.1f}% baseline, {diff:+5.1f}%)')
sys.stdout.flush()

# High confidence selections
print('\n=== HIGH CONFIDENCE (RF >= 0.7) BY ZONE ===')
rf_high = df_test[df_test['rf_proba'] >= 0.7]
print(f'High confidence count: {len(rf_high):,}')
if len(rf_high) > 0:
    print(f'High confidence WR: {rf_high["target"].mean()*100:.1f}%')
    for lo, hi in bands:
        zone_mask = (rf_high['vwap_width_atr'] >= lo) & (rf_high['vwap_width_atr'] < hi)
        zone_df = rf_high[zone_mask]
        if len(zone_df) >= 10:
            wr = zone_df['target'].mean() * 100
            print(f'  {lo:.1f}-{hi:.1f} ATR: {wr:.1f}% WR (n={len(zone_df):,})')
sys.stdout.flush()

print('\nDone!')
