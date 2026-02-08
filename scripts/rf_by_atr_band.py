"""RF analysis by ATR band on 10-year data - FAST version."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys

print('=== RF RESULTS BY ATR BAND (10-YEAR DATA) ===\n')
sys.stdout.flush()

# Load data
df = pd.read_csv('C:/Users/Administrator/ibkr8/data/tsla_5min_10years_indicators.csv')
df['time'] = pd.to_datetime(df['time'], utc=True)
df.set_index('time', inplace=True)
df['date'] = df.index.date
print(f'Loaded {len(df):,} bars')
sys.stdout.flush()

# Pre-compute targets for ALL bars once (vectorized by day)
print('Computing targets...')
sys.stdout.flush()

df['target'] = np.nan
dates = df['date'].unique()

for i, date in enumerate(dates):
    if i % 500 == 0:
        print(f'  Processing day {i}/{len(dates)}...')
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

print('Targets computed!')
sys.stdout.flush()

# Features
feature_cols = ['extension_velocity_3', 'rsi_extremity', 'momentum_divergence_5', 
                'vwap_width_atr', 'rsi']

# Drop NaN features
df_valid = df.dropna(subset=['target'] + feature_cols)
print(f'Valid rows: {len(df_valid):,}')
sys.stdout.flush()

# ATR bands
bands = [
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 4.0),
]

print(f'\n{"ATR Band":>12} | {"Raw WR":>8} | {"RF>=0.5 WR":>10} | {"RF>=0.5 N":>9} | {"Lift":>6}')
print('-' * 60)
sys.stdout.flush()

for lo, hi in bands:
    mask = (df_valid['vwap_width_atr'] >= lo) & (df_valid['vwap_width_atr'] < hi)
    zone_df = df_valid[mask]
    
    if len(zone_df) < 100:
        continue
    
    # Time split
    train_mask = zone_df.index.year <= 2023
    test_mask = zone_df.index.year >= 2024
    
    X_train = zone_df.loc[train_mask, feature_cols]
    y_train = zone_df.loc[train_mask, 'target']
    X_test = zone_df.loc[test_mask, feature_cols]
    y_test = zone_df.loc[test_mask, 'target']
    
    if len(X_train) < 30 or len(X_test) < 10:
        continue
    
    # Train RF
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=30,
        random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    
    # Predict
    test_proba = rf.predict_proba(X_test)[:, 1]
    
    # Results
    raw_wr = y_test.mean() * 100
    rf_sel = y_test[test_proba >= 0.5]
    rf_wr = rf_sel.mean() * 100 if len(rf_sel) > 0 else 0
    lift = rf_wr - raw_wr
    
    print(f'{lo:.1f}-{hi:.1f} ATR | {raw_wr:7.1f}% | {rf_wr:9.1f}% | {len(rf_sel):>9,} | {lift:+5.1f}%')
    sys.stdout.flush()

print('\nDone!')
