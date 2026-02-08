# filepath: c:\Users\Administrator\ibkr8\scripts\rf_train_vs_test.py
"""
RF Train vs Test Comparison
Shows the same stats on training data vs test data to detect overfitting.
NOW WITH DOLLAR-BASED P&L CALCULATIONS!
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import P&L addon
from pnl_addon import add_pnl_columns, calculate_pnl_metrics, print_pnl_comparison

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
print(f'Trades with outcomes: {len(df_filt)}, Overall Win rate: {df_filt["win"].mean()*100:.1f}%')

# ADD P&L COLUMNS (100 shares, 0.25 ATR stop, 6R target)
print('\nAdding dollar-based P&L calculations...')
df_filt = add_pnl_columns(df_filt, shares_per_trade=100, stop_atr_width=0.25, target_rr=6.0)
print(f'  Average P&L per trade: ${df_filt["pnl_net"].mean():.2f}')
print(f'  Total P&L (all trades): ${df_filt["pnl_net"].sum():,.2f}')

# Define zones
zones = [
    (0.5, 1.0, '0.5-1.0 ATR'),
    (1.0, 1.5, '1.0-1.5 ATR'),
    (1.5, 2.0, '1.5-2.0 ATR'),
    (2.0, 2.5, '2.0-2.5 ATR'),
]

print()
print('='*80)
print('RF ANALYSIS: TRAINING vs TEST DATA COMPARISON')
print('='*80)
print()
print('Proper 70/30 train/test split for each zone')
print()

summary_rows = []

for low, high, zone_name in zones:
    zone_mask = (df_filt['vwap_width_atr'] >= low) & (df_filt['vwap_width_atr'] < high)
    zone_df = df_filt[zone_mask].copy()
    
    if len(zone_df) < 50:
        print(f'{zone_name}: Insufficient data ({len(zone_df)} trades)')
        continue
    
    X = zone_df[available].fillna(zone_df[available].median())
    y = zone_df['win']
    
    # Train/test split - preserve indices for P&L analysis
    train_idx, test_idx = train_test_split(zone_df.index, test_size=0.3, random_state=42)
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, 
                                random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    # Get predictions and store in original dataframe
    zone_df.loc[train_idx, 'rf_prob'] = rf.predict_proba(X_train)[:, 1]
    zone_df.loc[test_idx, 'rf_prob'] = rf.predict_proba(X_test)[:, 1]
    proba_train = zone_df.loc[train_idx, 'rf_prob']
    proba_test = zone_df.loc[test_idx, 'rf_prob']
    
    print(f'--- {zone_name} ---')
    print(f'Total: {len(zone_df)} | Train: {len(X_train)} | Test: {len(X_test)}')
    print()
    
    # Stats table header
    print(f'{"Metric":<25} {"TRAIN":>15} {"TEST":>15} {"Diff":>10}')
    print('-'*68)
    
    # Baseline win rates
    train_base_wr = y_train.mean() * 100
    test_base_wr = y_test.mean() * 100
    print(f'{"Baseline Win Rate":<25} {train_base_wr:>14.1f}% {test_base_wr:>14.1f}% {test_base_wr - train_base_wr:>+9.1f}%')
    
    # RF >= 0.5 filter
    train_rf50 = y_train[proba_train >= 0.5]
    test_rf50 = y_test[proba_test >= 0.5]
    train_rf50_wr = train_rf50.mean() * 100 if len(train_rf50) > 0 else 0
    test_rf50_wr = test_rf50.mean() * 100 if len(test_rf50) > 0 else 0
    print(f'{"RF>=0.5 Win Rate":<25} {train_rf50_wr:>14.1f}% {test_rf50_wr:>14.1f}% {test_rf50_wr - train_rf50_wr:>+9.1f}%')
    
    # N trades at RF >= 0.5
    n_train_50 = len(train_rf50)
    n_test_50 = len(test_rf50)
    print(f'{"N @ RF>=0.5":<25} {n_train_50:>15} {n_test_50:>15} {n_test_50 - n_train_50:>+10}')
    
    # Improvement over baseline
    train_improve = train_rf50_wr - train_base_wr
    test_improve = test_rf50_wr - test_base_wr
    print(f'{"Improvement vs Base":<25} {train_improve:>+14.1f}% {test_improve:>+14.1f}% {test_improve - train_improve:>+9.1f}%')
    
    # RF >= 0.6 filter
    train_rf60 = y_train[proba_train >= 0.6]
    test_rf60 = y_test[proba_test >= 0.6]
    train_rf60_wr = train_rf60.mean() * 100 if len(train_rf60) > 0 else 0
    test_rf60_wr = test_rf60.mean() * 100 if len(test_rf60) > 0 else 0
    print(f'{"RF>=0.6 Win Rate":<25} {train_rf60_wr:>14.1f}% {test_rf60_wr:>14.1f}% {test_rf60_wr - train_rf60_wr:>+9.1f}%')
    n_train_60 = len(train_rf60)
    n_test_60 = len(test_rf60)
    print(f'{"N @ RF>=0.6":<25} {n_train_60:>15} {n_test_60:>15} {n_test_60 - n_train_60:>+10}')
    
    # RF >= 0.4 filter
    train_rf40 = y_train[proba_train >= 0.4]
    test_rf40 = y_test[proba_test >= 0.4]
    train_rf40_wr = train_rf40.mean() * 100 if len(train_rf40) > 0 else 0
    test_rf40_wr = test_rf40.mean() * 100 if len(test_rf40) > 0 else 0
    print(f'{"RF>=0.4 Win Rate":<25} {train_rf40_wr:>14.1f}% {test_rf40_wr:>14.1f}% {test_rf40_wr - train_rf40_wr:>+9.1f}%')
    n_train_40 = len(train_rf40)
    n_test_40 = len(test_rf40)
    print(f'{"N @ RF>=0.4":<25} {n_train_40:>15} {n_test_40:>15} {n_test_40 - n_train_40:>+10}')
    
    print()
    
    # === DOLLAR P&L ANALYSIS ===
    print('-' * 68)
    print('DOLLAR P&L METRICS (100 shares, 0.25 ATR stop, 6R target)')
    print('-' * 68)
    
    # Calculate P&L for train/test splits
    train_zone_df = zone_df.loc[train_idx]
    test_zone_df = zone_df.loc[test_idx]
    
    # Unfiltered P&L
    train_pnl_base = calculate_pnl_metrics(train_zone_df)
    test_pnl_base = calculate_pnl_metrics(test_zone_df)
    
    # RF >= 0.5 filtered P&L
    train_pnl_rf50 = calculate_pnl_metrics(train_zone_df, filter_col='rf_prob', filter_threshold=0.5)
    test_pnl_rf50 = calculate_pnl_metrics(test_zone_df, filter_col='rf_prob', filter_threshold=0.5)
    
    # Print P&L comparison
    print(f'\n{"P&L Metric":<25} {"TRAIN":>15} {"TEST":>15} {"Diff":>10}')
    print('-'*68)
    print(f'{"Baseline Avg P&L/Trade":<25} ${train_pnl_base["avg_pnl"]:>14.2f} ${test_pnl_base["avg_pnl"]:>14.2f} ${test_pnl_base["avg_pnl"]-train_pnl_base["avg_pnl"]:>+9.2f}')
    print(f'{"Baseline Total P&L":<25} ${train_pnl_base["total_pnl"]:>14,.0f} ${test_pnl_base["total_pnl"]:>14,.0f} ${test_pnl_base["total_pnl"]-train_pnl_base["total_pnl"]:>+9,.0f}')
    print(f'{"RF>=0.5 Avg P&L/Trade":<25} ${train_pnl_rf50["avg_pnl"]:>14.2f} ${test_pnl_rf50["avg_pnl"]:>14.2f} ${test_pnl_rf50["avg_pnl"]-train_pnl_rf50["avg_pnl"]:>+9.2f}')
    print(f'{"RF>=0.5 Total P&L":<25} ${train_pnl_rf50["total_pnl"]:>14,.0f} ${test_pnl_rf50["total_pnl"]:>14,.0f} ${test_pnl_rf50["total_pnl"]-train_pnl_rf50["total_pnl"]:>+9,.0f}')
      # Calculate improvement
    if train_pnl_base["avg_pnl"] != 0:
        train_pnl_improve = ((train_pnl_rf50["avg_pnl"] - train_pnl_base["avg_pnl"]) / abs(train_pnl_base["avg_pnl"])) * 100
    else:
        train_pnl_improve = 0
    if test_pnl_base["avg_pnl"] != 0:
        test_pnl_improve = ((test_pnl_rf50["avg_pnl"] - test_pnl_base["avg_pnl"]) / abs(test_pnl_base["avg_pnl"])) * 100
    else:
        test_pnl_improve = 0
    
    print(f'{"P&L Improvement":<25} {train_pnl_improve:>+14.1f}% {test_pnl_improve:>+14.1f}% {test_pnl_improve-train_pnl_improve:>+9.1f}%')
    print()
    
    # Store for summary
    summary_rows.append({
        'Zone': zone_name,
        'Train Base WR': train_base_wr,
        'Test Base WR': test_base_wr,
        'Train RF50 WR': train_rf50_wr,
        'Test RF50 WR': test_rf50_wr,
        'Train N@50': n_train_50,
        'Test N@50': n_test_50,
        'WR Gap (Train-Test)': train_rf50_wr - test_rf50_wr,
        'Test Base Avg P&L': test_pnl_base["avg_pnl"],
        'Test RF50 Avg P&L': test_pnl_rf50["avg_pnl"],
        'Test P&L Improve %': test_pnl_improve,
    })

# Summary Table
print('='*80)
print('SUMMARY TABLE: WIN RATES')
print('='*80)
print()
print(f'{"Zone":<15} {"Train Base":>12} {"Test Base":>12} {"Train RF50":>12} {"Test RF50":>12} {"WR Gap":>10}')
print('-'*75)
for row in summary_rows:
    gap_marker = ' ***' if row['WR Gap (Train-Test)'] > 10 else ''
    print(f'{row["Zone"]:<15} {row["Train Base WR"]:>11.1f}% {row["Test Base WR"]:>11.1f}% {row["Train RF50 WR"]:>11.1f}% {row["Test RF50 WR"]:>11.1f}% {row["WR Gap (Train-Test)"]:>+9.1f}%{gap_marker}')

print()
print('='*80)
print('SUMMARY TABLE: DOLLAR P&L (TEST SET)')
print('='*80)
print()
print(f'{"Zone":<15} {"Base $/Trade":>15} {"RF50 $/Trade":>15} {"P&L Improve":>15} {"Trades@RF50":>12}')
print('-'*75)
for row in summary_rows:
    improve_marker = ' ✓' if row['Test P&L Improve %'] > 0 else ''
    print(f'{row["Zone"]:<15} ${row["Test Base Avg P&L"]:>14.2f} ${row["Test RF50 Avg P&L"]:>14.2f} {row["Test P&L Improve %"]:>+14.1f}%{improve_marker} {row["Test N@50"]:>12,}')

print()
print('='*80)
print('INTERPRETATION')
print('='*80)
print('''
WIN RATE TABLE:
- TRAIN stats show in-sample performance (RF sees this data during training)
- TEST stats show out-of-sample performance (RF never sees this data)
- WR Gap = Train RF50 WR - Test RF50 WR
- Large positive gap (*** marker) = overfitting (model memorized noise)
- Small/negative gap = model generalizes well

P&L TABLE (TEST SET ONLY):
- Base $/Trade = Average P&L per trade without RF filter
- RF50 $/Trade = Average P&L per trade with RF >= 0.5 filter
- P&L Improve = Percentage improvement from RF filtering
- ✓ = Positive improvement from RF filtering

POSITION SIZING: 100 shares, 0.25 ATR stop, 6R target (~1.5 ATR)
COSTS INCLUDED: $0.005/share commission + $0.01/share slippage

Key insight: If Train WR >> Test WR, the model is overfitting.
Real-world performance will be closer to TEST numbers.
''')

