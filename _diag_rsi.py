"""Diagnose why RSI has low feature importance in the RF VWAP reversion model."""
import sys, warnings
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from master_pipeline import (
    load_and_validate_data, calculate_core_indicators, apply_setup_filter,
    DATA_FILE, TEST_YEAR,
)
from label_generator import LabelConfig, generate_labels

df = load_and_validate_data(DATA_FILE)
df = calculate_core_indicators(df, verbose=False)
config = LabelConfig(stop_atrs=[0.6])
df = generate_labels(df, config)

# Apply setup filter (same as best run)
label_col = 'label_s0_6'
valid = df[label_col].notna()
df_v = df[valid].copy()
mask = apply_setup_filter(
    df_v, stop_atr=0.6, min_dist_atr=0.5,
    min_minutes_session=15, max_minutes_session=360, min_rr_setup=1.0,
)
df_s = df_v[mask].copy()
df_s['year'] = pd.to_datetime(df_s['datetime']).dt.year
df_test = df_s[df_s['year'] >= 2024].copy()
y = df_test[label_col].astype(int)

print(f"Setup-filtered test bars: {len(df_test):,}")
print()

# ── 1. RSI distribution ──────────────────────────────────────────────────────
print("=== RSI Distribution (setup-filtered test set) ===")
print(df_test['rsi'].describe())
print()
pct_below30 = (df_test['rsi'] < 30).mean() * 100
pct_mid = ((df_test['rsi'] >= 30) & (df_test['rsi'] <= 70)).mean() * 100
pct_above70 = (df_test['rsi'] > 70).mean() * 100
n_below30 = (df_test['rsi'] < 30).sum()
n_mid = ((df_test['rsi'] >= 30) & (df_test['rsi'] <= 70)).sum()
n_above70 = (df_test['rsi'] > 70).sum()
n_extreme = (df_test['rsi_extreme'] == 1).sum()
print(f"RSI < 30:  {n_below30:>5,} ({pct_below30:.1f}%)")
print(f"RSI 30-70: {n_mid:>5,} ({pct_mid:.1f}%)")
print(f"RSI > 70:  {n_above70:>5,} ({pct_above70:.1f}%)")
print(f"rsi_extreme==1: {n_extreme:,} ({n_extreme/len(df_test)*100:.1f}%)")
print()

# ── 2. Correlation of RSI with dominant features ─────────────────────────────
key_feats = [
    'vwap_slope', 'vwap_slope_5', 'vol_pct_complete', 'price_to_vwap_atr',
    'momentum_3bar_atr', 'momentum_6bar_atr', 'ema20_slope_atr',
    'rsi', 'rsi_slope',
]
corr = df_test[key_feats].corr()
print("=== RSI Correlations with Key Features ===")
print(corr['rsi'].sort_values(ascending=False).to_string())
print()
print("=== RSI_SLOPE Correlations ===")
print(corr['rsi_slope'].sort_values(ascending=False).to_string())
print()

# ── 3. RSI vs label win rate (bucket analysis) ──────────────────────────────
print("=== Win Rate by RSI Bucket ===")
bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
df_test = df_test.copy()
df_test['rsi_bin'] = pd.cut(df_test['rsi'], bins=bins)
grouped = df_test.groupby('rsi_bin', observed=False)[label_col].agg(['mean', 'count'])
grouped.columns = ['win_rate', 'n_bars']
grouped['pct_of_total'] = grouped['n_bars'] / grouped['n_bars'].sum() * 100
grouped['win_rate'] = grouped['win_rate'].map(lambda x: f"{x*100:.1f}%")
print(grouped.to_string())
print()

# ── 4. RSI extreme vs not ────────────────────────────────────────────────────
print("=== Win Rate: RSI Extreme vs Not ===")
for ext_val in [0, 1]:
    sub = df_test[df_test['rsi_extreme'] == ext_val]
    wr = sub[label_col].mean()
    print(f"  rsi_extreme={ext_val}: WR={wr*100:.1f}% (n={len(sub):,})")
print()

# ── 5. Directional RSI ──────────────────────────────────────────────────────
print("=== Directional RSI (Oversold=Long, Overbought=Short) ===")
long_mask = df_test['is_long_setup'] == 1
short_mask = df_test['is_long_setup'] == 0

for desc, m, rsi_cond in [
    ("Long setups, RSI<30", long_mask & (df_test['rsi'] < 30), None),
    ("Long setups, RSI>=30", long_mask & (df_test['rsi'] >= 30), None),
    ("Short setups, RSI>70", short_mask & (df_test['rsi'] > 70), None),
    ("Short setups, RSI<=70", short_mask & (df_test['rsi'] <= 70), None),
]:
    n = int(m.sum())
    if n > 0:
        wr = df_test.loc[m, label_col].mean() * 100
    else:
        wr = float('nan')
    print(f"  {desc:30s}: WR={wr:5.1f}% (n={n:,})")
print()

# ── 6. Conditional importance: RSI signal GIVEN vwap_slope direction ────────
print("=== Win Rate: RSI x VWAP Slope Interaction ===")
vwap_slope_pos = df_test['vwap_slope'] > 0  # VWAP moving up
vwap_slope_neg = df_test['vwap_slope'] <= 0  # VWAP moving down

for slope_desc, slope_mask in [("VWAP slope > 0 (rising)", vwap_slope_pos),
                                ("VWAP slope <= 0 (falling)", vwap_slope_neg)]:
    print(f"  {slope_desc}:")
    for rsi_desc, rsi_mask in [("RSI<30", df_test['rsi'] < 30),
                                ("RSI 30-50", (df_test['rsi'] >= 30) & (df_test['rsi'] < 50)),
                                ("RSI 50-70", (df_test['rsi'] >= 50) & (df_test['rsi'] <= 70)),
                                ("RSI>70", df_test['rsi'] > 70)]:
        combined = slope_mask & rsi_mask
        n = int(combined.sum())
        if n > 10:
            wr = df_test.loc[combined, label_col].mean() * 100
        else:
            wr = float('nan')
        print(f"    {rsi_desc:12s}: WR={wr:5.1f}% (n={n:,})")
print()

# ── 7. Mutual information / unique info ──────────────────────────────────────
print("=== Unique Variance: RSI vs Momentum Features ===")
from sklearn.ensemble import RandomForestRegressor
# Quick single-feature RF importance test
single_feats = ['rsi', 'vwap_slope', 'vwap_slope_5', 'momentum_3bar_atr', 'momentum_6bar_atr', 'ema20_slope_atr']
y_reg = np.where(y.values == 1, 1.0, -1.0)

train_mask = df_s['year'] < 2024
df_train = df_s[train_mask].copy()
y_train_reg = np.where(df_train[label_col].astype(int).values == 1, 1.0, -1.0)

for feat in single_feats:
    X_single = df_train[[feat]].replace([np.inf, -np.inf], np.nan).fillna(0).values
    X_test_single = df_test[[feat]].replace([np.inf, -np.inf], np.nan).fillna(0).values
    rf_single = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=50, random_state=42, n_jobs=-1)
    rf_single.fit(X_single, y_train_reg)
    score = rf_single.score(X_test_single, y_reg)
    print(f"  {feat:25s}: R^2 = {score:+.4f}")

print()
print("=== RSI Variance After Residualizing on VWAP Slopes ===")
from sklearn.linear_model import LinearRegression
# How much of RSI is already explained by vwap_slope + vwap_slope_5?
X_slopes = df_test[['vwap_slope', 'vwap_slope_5', 'momentum_3bar_atr', 'momentum_6bar_atr']].fillna(0).values
lr = LinearRegression().fit(X_slopes, df_test['rsi'].fillna(50).values)
r2 = lr.score(X_slopes, df_test['rsi'].fillna(50).values)
print(f"  R^2 of RSI ~ vwap_slope + vwap_slope_5 + momentum_3/6: {r2:.4f}")
print(f"  => {r2*100:.1f}% of RSI variance is already captured by VWAP/momentum features")
print(f"  => Only {(1-r2)*100:.1f}% of RSI is 'unique' information")
