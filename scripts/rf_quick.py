"""Quick RF analysis - no argparse."""
print("="*60)
print("RANDOM FOREST REVERSAL TRADE ANALYSIS")
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

# Feature columns
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
    'rr_theo', 'avg_rr',
]
available = [c for c in feature_cols if c in df.columns]
print(f'Using {len(available)} features')

# Filter to definitive outcomes
df_filt = df[df['outcome'].isin(['TARGET', 'STOP'])].copy()
print(f'Trades with outcomes: {len(df_filt)}')

X = df_filt[available].fillna(df_filt[available].median())
y = (df_filt['outcome'] == 'TARGET').astype(int)
print(f'Class balance: {100*y.mean():.1f}% wins')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

# Train RF
print('\nTraining Random Forest...')
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, 
                            random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

print('\n' + '='*60)
print('MODEL EVALUATION')
print('='*60)
print(classification_report(y_test, y_pred, target_names=['STOP', 'TARGET']))

auc = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
print(f'ROC AUC: {auc:.3f}')
print(f'CV AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')

# Feature importance
print('\n' + '='*60)
print('TOP 15 FEATURES')
print('='*60)
imp = pd.DataFrame({'feature': available, 'importance': rf.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for _, row in imp.head(15).iterrows():
    bar = '*' * int(row['importance'] * 50)
    print(f"{row['feature']:<28} {row['importance']:.3f} {bar}")

# Trading simulation
print('\n' + '='*60)
print('TRADING SIMULATION')
print('='*60)
test_idx = X_test.index
test_df = df_filt.loc[test_idx].copy()
test_df['pred_prob'] = y_prob

baseline_wr = 100 * (test_df['outcome']=='TARGET').mean()
baseline_pnl = test_df['pnl'].sum()
print(f'Baseline: {len(test_df)} trades, WR={baseline_wr:.1f}%, PnL=${baseline_pnl:.2f}')

print('\nPerformance by threshold:')
print(f"{'Thresh':<8} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL':>12} {'PF':>8}")
for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
    sub = test_df[test_df['pred_prob'] >= thresh]
    if len(sub) > 0:
        wins = (sub['outcome']=='TARGET').sum()
        wr = 100*wins/len(sub)
        pnl = sub['pnl'].sum()
        gross_wins = sub[sub['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(sub[sub['pnl'] < 0]['pnl'].sum())
        pf = gross_wins / gross_loss if gross_loss > 0 else 0
        print(f'{thresh:<8} {len(sub):>8} {wins:>8} {wr:>7.1f}% ${pnl:>10.2f} {pf:>7.2f}')

# Zone analysis
print('\n' + '='*60)
print('PERFORMANCE BY VWAP ZONE')
print('='*60)
test_df['vwap_width_atr'] = X_test['vwap_width_atr'].values

zones = [
    ('Sweet spot (1.0-1.5 ATR)', 1.0, 1.5),
    ('Moderate (1.5-2.5 ATR)', 1.5, 2.5),
    ('Extended (2.5-3.5 ATR)', 2.5, 3.5),
    ('Over-extended (>3.5 ATR)', 3.5, 100),
]

print(f"{'Zone':<30} {'N':>6} {'ActualWR':>10} {'AvgProb':>10}")
for zone_name, low, high in zones:
    mask = (test_df['vwap_width_atr'] >= low) & (test_df['vwap_width_atr'] < high)
    sub = test_df[mask]
    if len(sub) > 0:
        actual_wr = 100 * (sub['outcome']=='TARGET').mean()
        avg_prob = 100 * sub['pred_prob'].mean()
        print(f"{zone_name:<30} {len(sub):>6} {actual_wr:>9.1f}% {avg_prob:>9.1f}%")

# Combined strategy: Zone filter + RF filter
print('\n' + '='*60)
print('COMBINED STRATEGY: Sweet Spot Zone + RF Filter')
print('='*60)
sweet_spot = test_df[(test_df['vwap_width_atr'] >= 1.0) & (test_df['vwap_width_atr'] < 2.0)]
print(f"Sweet spot zone only: {len(sweet_spot)} trades, WR={(sweet_spot['outcome']=='TARGET').mean()*100:.1f}%")

for thresh in [0.5, 0.55, 0.6, 0.65]:
    combined = sweet_spot[sweet_spot['pred_prob'] >= thresh]
    if len(combined) > 0:
        wins = (combined['outcome']=='TARGET').sum()
        wr = 100*wins/len(combined)
        pnl = combined['pnl'].sum()
        gross_wins = combined[combined['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(combined[combined['pnl'] < 0]['pnl'].sum())
        pf = gross_wins / gross_loss if gross_loss > 0 else 0
        print(f"  + RF>={thresh}: {len(combined)} trades, WR={wr:.1f}%, PnL=${pnl:.2f}, PF={pf:.2f}")
