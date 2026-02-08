"""RF analysis with 2:1 R:R filter - only take trades where VWAP target gives 2:1."""
print("="*60)
print("RANDOM FOREST WITH 2:1 R:R FILTER")
print("="*60)
import sys
sys.stdout.flush()
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/tsla_5min_2years_ml_features.csv')
print(f'Loaded {len(df)} trades')

# Check R:R distribution
print('\n' + '='*60)
print('R:R DISTRIBUTION (rr_theo = distance to VWAP / stop distance)')
print('='*60)
print(df['rr_theo'].describe())

# Filter to 2:1 R:R minimum
print('\n' + '='*60)
print('FILTERING TO R:R >= 2.0')
print('='*60)
df_2to1 = df[df['rr_theo'] >= 2.0].copy()
print(f'Trades with R:R >= 2:1: {len(df_2to1)} ({100*len(df_2to1)/len(df):.1f}% of total)')

# Check outcomes for 2:1 trades
df_2to1_outcomes = df_2to1[df_2to1['outcome'].isin(['TARGET', 'STOP'])]
print(f'With definitive outcomes: {len(df_2to1_outcomes)}')

wins = (df_2to1_outcomes['outcome'] == 'TARGET').sum()
losses = (df_2to1_outcomes['outcome'] == 'STOP').sum()
wr = 100 * wins / len(df_2to1_outcomes)
print(f'Win rate: {wins}/{len(df_2to1_outcomes)} = {wr:.1f}%')

# Calculate expected value with 2:1 R:R
# EV = WR * Win_Amount - (1-WR) * Loss_Amount
# With 2:1 R:R: Win = 2R, Loss = 1R
ev_per_r = (wr/100) * 2 - (1 - wr/100) * 1
print(f'\nExpected Value per R (2:1 R:R): {ev_per_r:.3f}R')
print(f'  (Positive = profitable, need WR > 33.3% for 2:1)')

# Actual P&L
total_pnl = df_2to1_outcomes['pnl'].sum()
avg_pnl = df_2to1_outcomes['pnl'].mean()
print(f'\nActual P&L: ${total_pnl:.2f} (avg ${avg_pnl:.2f}/trade)')

# Check actual R:R achieved
print('\n' + '='*60)
print('ACTUAL R:R ACHIEVED')
print('='*60)
print('Winners:')
winners = df_2to1_outcomes[df_2to1_outcomes['outcome'] == 'TARGET']
print(f'  Avg rr_achieved: {winners["rr_achieved"].mean():.2f}')
print(f'  Avg pnl: ${winners["pnl"].mean():.2f}')

print('Losers:')
losers = df_2to1_outcomes[df_2to1_outcomes['outcome'] == 'STOP']
print(f'  Avg rr_achieved: {losers["rr_achieved"].mean():.2f}')
print(f'  Avg pnl: ${losers["pnl"].mean():.2f}')

# The issue: are we actually getting 2:1 on wins?
print('\n' + '='*60)
print('THE PROBLEM: Theoretical vs Actual R:R')
print('='*60)
print(f'Theoretical R:R (at entry): {df_2to1_outcomes["rr_theo"].mean():.2f}')
print(f'Actual R:R achieved (winners): {winners["rr_achieved"].mean():.2f}')
print(f'Actual R:R achieved (losers): {losers["rr_achieved"].mean():.2f}')

# Profit Factor
gross_wins = winners['pnl'].sum()
gross_losses = abs(losers['pnl'].sum())
pf = gross_wins / gross_losses if gross_losses > 0 else 0
print(f'\nProfit Factor: {pf:.2f}')
print(f'  Gross Wins: ${gross_wins:.2f}')
print(f'  Gross Losses: ${gross_losses:.2f}')

# Now run RF on just the 2:1 trades
print('\n' + '='*60)
print('RANDOM FOREST ON 2:1 R:R TRADES')
print('='*60)

feature_cols = [
    'vwap_width_atr', 'price_to_vwap_atr',
    'rsi', 'rsi_extremity', 'bb_pct',
    'rsi_slope_3', 'rsi_slope_5',
    'momentum_divergence_3', 'momentum_divergence_5',
    'reversal_wick', 'reversal_close_position',
    'bar_range_atr',
    'consecutive_shrinking', 'range_vs_prev',
    'vol_declining', 'vol_trend_3',
    'extension_velocity_3', 'extension_velocity_5',
    'extension_accel', 'vwap_helping',
    'rel_vol', 'vol_at_extension',
    'bb_extension', 'bb_extension_abs',
]
available = [c for c in feature_cols if c in df_2to1_outcomes.columns]
print(f'Using {len(available)} features')

X = df_2to1_outcomes[available].fillna(df_2to1_outcomes[available].median())
y = (df_2to1_outcomes['outcome'] == 'TARGET').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, 
                            random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['STOP', 'TARGET']))

auc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f'ROC AUC: {auc:.3f}')
print(f'CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')

# Trading simulation with RF filter
print('\n' + '='*60)
print('TRADING SIMULATION: 2:1 R:R + RF FILTER')
print('='*60)

test_idx = X_test.index
test_df = df_2to1_outcomes.loc[test_idx].copy()
test_df['pred_prob'] = y_prob

baseline_wins = (test_df['outcome']=='TARGET').sum()
baseline_wr = 100 * baseline_wins / len(test_df)
baseline_pnl = test_df['pnl'].sum()
print(f'Baseline (all 2:1 trades): {len(test_df)} trades, WR={baseline_wr:.1f}%, PnL=${baseline_pnl:.2f}')

print('\nWith RF probability filter:')
print(f"{'Thresh':<8} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL':>12} {'PF':>8}")
for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
    sub = test_df[test_df['pred_prob'] >= thresh]
    if len(sub) > 5:
        wins = (sub['outcome']=='TARGET').sum()
        wr = 100*wins/len(sub)
        pnl = sub['pnl'].sum()
        gross_w = sub[sub['pnl'] > 0]['pnl'].sum()
        gross_l = abs(sub[sub['pnl'] < 0]['pnl'].sum())
        pf = gross_w / gross_l if gross_l > 0 else 0
        print(f'{thresh:<8} {len(sub):>8} {wins:>8} {wr:>7.1f}% ${pnl:>10.2f} {pf:>7.2f}')

# Feature importance
print('\n' + '='*60)
print('TOP 10 FEATURES FOR 2:1 TRADES')
print('='*60)
imp = pd.DataFrame({'feature': available, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for _, row in imp.head(10).iterrows():
    bar = '*' * int(row['importance'] * 50)
    print(f"{row['feature']:<28} {row['importance']:.3f} {bar}")
