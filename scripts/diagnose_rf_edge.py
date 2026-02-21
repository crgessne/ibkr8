"""
Diagnose WHY the RF model can't separate winners from losers.

Key questions:
1. What is the RF actually learning? (feature importances + partial dependence)
2. Does the model's predicted probability correlate AT ALL with actual outcomes?
3. Are the features fundamentally uninformative, or is the model undertrained?
4. If we add better features (time-of-day, session context, trend), does edge appear?
5. What does the "conditional WR" look like when the RF is most confident?

This script also tests whether VWAP reversion has a conditional edge:
  - Only when price is >2 ATR from VWAP?
  - Only in the first 2 hours of the session?
  - Only on high-volume bars?
  - Only longs?
  - Combinations of the above?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from label_generator import LabelConfig, generate_labels

DATA_FILE = Path("data/tsla_5min_10years.csv")
STOP_ATR = 0.5  # Focus on one stop for clarity
TEST_YEAR = 2024
SHARES = 100
COST_RT = 3.0  # round-trip cost per trade

# ============================================================================
# LOAD + INDICATORS (reuse master_pipeline logic)
# ============================================================================
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['datetime'] = pd.to_datetime(df['time'], utc=True)
df['date'] = df['datetime'].dt.date

# ATR
high, low, close = df['high'], df['low'], df['close']
tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# VWAP
tp = (df['high'] + df['low'] + df['close']) / 3
pv = tp * df['volume']
df['vwap'] = df.groupby('date').apply(
    lambda g: pv.loc[g.index].cumsum() / df.loc[g.index, 'volume'].cumsum()
).reset_index(level=0, drop=True)

# Basic features (same as master_pipeline)
df['vwap_width_atr'] = abs(df['close'] - df['vwap']) / df['atr']
df['price_to_vwap_atr'] = (df['close'] - df['vwap']) / df['atr']
df['is_long_setup'] = (df['close'] < df['vwap']).astype(int)
df['vwap_slope'] = df['vwap'].diff(1)
df['vwap_slope_5'] = df['vwap'].diff(5)
df['vwap_helping'] = np.where(
    df['is_long_setup'], df['vwap_slope'] < 0, df['vwap_slope'] > 0
).astype(int)
df['rel_vol'] = df['volume'] / df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['volume'].shift(1)
df['vol_at_extension'] = df['volume'] / df['volume'].rolling(5).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0.0)
loss_s = (-delta).where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss_s.rolling(14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df['rsi'] = 100 - (100 / (1 + rs))
df['rsi_slope'] = df['rsi'].diff(3)
df['rsi_extreme'] = ((df['rsi'] < 30) | (df['rsi'] > 70)).astype(int)

df['bar_range_atr'] = (df['high'] - df['low']) / df['atr']
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
df['crossed_vwap'] = (df['is_long_setup'] != df['is_long_setup'].shift(1)).astype(int)
df['bars_from_vwap'] = df.groupby((df['crossed_vwap'] == 1).cumsum()).cumcount()

# ============================================================================
# NEW FEATURES the current pipeline is MISSING
# ============================================================================
print("Adding new features...")

# 1. Time-of-day features (crucial for intraday strategies)
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['minutes_into_session'] = (df['hour'] - 9) * 60 + df['minute'] - 30  # EST assumed
df['minutes_into_session'] = df['minutes_into_session'].clip(0, 390)

# Session phases: open (0-30min), morning (30-120), midday (120-270), close (270-390)
df['session_phase'] = pd.cut(
    df['minutes_into_session'], bins=[-1, 30, 120, 270, 391],
    labels=[0, 1, 2, 3]
).astype(float)

# 2. Daily cumulative volume profile
df['cum_vol_today'] = df.groupby('date')['volume'].cumsum()
df['total_vol_today'] = df.groupby('date')['volume'].transform('sum')
df['vol_pct_complete'] = df['cum_vol_today'] / df['total_vol_today']

# 3. How many times has price crossed VWAP today?
df['vwap_crosses_today'] = df.groupby('date')['crossed_vwap'].cumsum()

# 4. Day's range so far (context for "how extended are we?")
df['day_high'] = df.groupby('date')['high'].cummax()
df['day_low'] = df.groupby('date')['low'].cummin()
df['day_range_atr'] = (df['day_high'] - df['day_low']) / df['atr']
df['pct_of_day_range'] = np.where(
    df['day_high'] > df['day_low'],
    (df['close'] - df['day_low']) / (df['day_high'] - df['day_low']),
    0.5
)

# 5. VWAP as % of day range — is VWAP near top/bottom/middle of range?
df['vwap_in_day_range'] = np.where(
    df['day_high'] > df['day_low'],
    (df['vwap'] - df['day_low']) / (df['day_high'] - df['day_low']),
    0.5
)

# 6. Momentum into extension: last 3-bar price move in ATR
df['momentum_3bar_atr'] = (df['close'] - df['close'].shift(3)) / df['atr']
df['momentum_6bar_atr'] = (df['close'] - df['close'].shift(6)) / df['atr']

# 7. Bar-level mean reversion signal: close vs open direction relative to VWAP
df['bar_reverting'] = np.where(
    df['is_long_setup'],  # price below VWAP (long setup)
    (df['close'] > df['open']).astype(int),  # bullish bar = reverting toward VWAP
    (df['close'] < df['open']).astype(int),  # bearish bar = reverting toward VWAP
)

# 8. Consecutive bars on same side of VWAP
df['consecutive_same_side'] = df.groupby(
    (df['is_long_setup'] != df['is_long_setup'].shift(1)).cumsum()
).cumcount() + 1

# 9. Open-to-VWAP distance (was today's open above/below VWAP?)
daily_open = df.groupby('date')['open'].transform('first')
df['open_vs_vwap_atr'] = (daily_open - df['vwap']) / df['atr']

# 10. Prior bar outcome: did prior bar move toward or away from VWAP?
df['prior_bar_toward_vwap'] = np.where(
    df['is_long_setup'],
    (df['close'].shift(1) > df['close'].shift(2)).astype(float),
    (df['close'].shift(1) < df['close'].shift(2)).astype(float),
)

# 11. Trend context: 20-bar EMA slope
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema20_slope_atr'] = (df['ema20'] - df['ema20'].shift(5)) / df['atr']

# 12. Extension speed: how fast did price move away from VWAP?
df['extension_speed'] = df['vwap_width_atr'] / (df['bars_from_vwap'] + 1)

print(f"  Total features available: {len([c for c in df.columns if df[c].dtype in ['float64','int64','float32','int32','bool']])}")

# ============================================================================
# GENERATE LABELS
# ============================================================================
print("Generating labels...")
config = LabelConfig(stop_atrs=[STOP_ATR])
df = generate_labels(df, config)
label_col = f"label_s{STOP_ATR}".replace(".", "_")

# ============================================================================
# BUILD FEATURE SETS
# ============================================================================
base_features = [
    'vwap_width_atr', 'price_to_vwap_atr', 'is_long_setup',
    'vwap_slope', 'vwap_slope_5', 'vwap_helping',
    'rel_vol', 'vol_ratio', 'vol_at_extension',
    'rsi', 'rsi_slope', 'rsi_extreme',
    'bar_range_atr', 'close_position', 'crossed_vwap', 'bars_from_vwap',
]

new_features = [
    'minutes_into_session', 'session_phase',
    'vol_pct_complete', 'vwap_crosses_today',
    'day_range_atr', 'pct_of_day_range', 'vwap_in_day_range',
    'momentum_3bar_atr', 'momentum_6bar_atr',
    'bar_reverting', 'consecutive_same_side',
    'open_vs_vwap_atr', 'prior_bar_toward_vwap',
    'ema20_slope_atr', 'extension_speed',
]

all_features = base_features + new_features

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
valid = df[label_col].notna()
df_valid = df[valid].copy()
df_valid['year'] = df_valid['datetime'].dt.year
train_mask = df_valid['year'] < TEST_YEAR
test_mask = df_valid['year'] >= TEST_YEAR

y = df_valid[label_col].astype(int)

# ============================================================================
# TEST 1: RF with base features (current pipeline)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: RF with CURRENT features (baseline)")
print("="*80)

X = df_valid[base_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

rf1 = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=50,
    max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'
)
rf1.fit(X_train, y_train)
p1 = rf1.predict_proba(X_test)[:, 1]

auc1 = roc_auc_score(y_test, p1)
print(f"  AUC-ROC: {auc1:.4f}  (0.50 = random, need >0.55 for any edge)")

# Calibration: predicted vs actual by decile
for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    pct = np.percentile(p1, q)
    mask_above = p1 >= pct
    actual_wr = y_test.values[mask_above].mean()
    n = mask_above.sum()
    print(f"  RF proba >= {pct:.3f} (top {100-q}%, n={n}): actual WR = {actual_wr*100:.1f}%")

# Top decile vs bottom decile
top_10_mask = p1 >= np.percentile(p1, 90)
bot_10_mask = p1 <= np.percentile(p1, 10)
wr_top = y_test.values[top_10_mask].mean()
wr_bot = y_test.values[bot_10_mask].mean()
print(f"\n  TOP 10% predicted: WR = {wr_top*100:.1f}% (n={top_10_mask.sum()})")
print(f"  BOT 10% predicted: WR = {wr_bot*100:.1f}% (n={bot_10_mask.sum()})")
print(f"  SEPARATION: {(wr_top - wr_bot)*100:.1f}% ({wr_top/max(wr_bot,0.001):.2f}x)")

print("\n  Feature importances:")
imp1 = pd.DataFrame({'feat': base_features, 'imp': rf1.feature_importances_}).sort_values('imp', ascending=False)
for _, row in imp1.head(10).iterrows():
    print(f"    {row['feat']:25s} {row['imp']:.4f}")

# ============================================================================
# TEST 2: RF with ENHANCED features
# ============================================================================
print("\n" + "="*80)
print("TEST 2: RF with ENHANCED features (+time, session, trend, etc.)")
print("="*80)

X2 = df_valid[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)
X2_train, X2_test = X2[train_mask], X2[test_mask]

rf2 = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=50,
    max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'
)
rf2.fit(X2_train, y_train)
p2 = rf2.predict_proba(X2_test)[:, 1]

auc2 = roc_auc_score(y_test, p2)
print(f"  AUC-ROC: {auc2:.4f}  (baseline was {auc1:.4f}, delta = {auc2-auc1:+.4f})")

for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    pct = np.percentile(p2, q)
    mask_above = p2 >= pct
    actual_wr = y_test.values[mask_above].mean()
    n = mask_above.sum()
    print(f"  RF proba >= {pct:.3f} (top {100-q}%, n={n}): actual WR = {actual_wr*100:.1f}%")

top_10_mask2 = p2 >= np.percentile(p2, 90)
bot_10_mask2 = p2 <= np.percentile(p2, 10)
wr_top2 = y_test.values[top_10_mask2].mean()
wr_bot2 = y_test.values[bot_10_mask2].mean()
print(f"\n  TOP 10% predicted: WR = {wr_top2*100:.1f}% (n={top_10_mask2.sum()})")
print(f"  BOT 10% predicted: WR = {wr_bot2*100:.1f}% (n={bot_10_mask2.sum()})")
print(f"  SEPARATION: {(wr_top2 - wr_bot2)*100:.1f}% ({wr_top2/max(wr_bot2,0.001):.2f}x)")

print("\n  Feature importances (enhanced):")
imp2 = pd.DataFrame({'feat': all_features, 'imp': rf2.feature_importances_}).sort_values('imp', ascending=False)
for _, row in imp2.head(15).iterrows():
    print(f"    {row['feat']:25s} {row['imp']:.4f}")

# ============================================================================
# TEST 3: DOLLAR P&L by model confidence decile (is there a PROFITABLE decile?)
# ============================================================================
print("\n" + "="*80)
print("TEST 3: DOLLAR P&L BY MODEL CONFIDENCE DECILE (enhanced model, stop=0.5)")
print("="*80)

test_df = df_valid[test_mask].copy()
test_df['rf_proba'] = p2
test_df['label'] = y_test.values

# Per-trade economics
test_df['reward_dollars'] = test_df['vwap_width_atr'] * test_df['atr'] * SHARES
test_df['risk_dollars'] = STOP_ATR * test_df['atr'] * SHARES
test_df['gross_pnl'] = np.where(test_df['label'] == 1, test_df['reward_dollars'], -test_df['risk_dollars'])
test_df['net_pnl'] = test_df['gross_pnl'] - COST_RT

# Decile analysis
test_df['decile'] = pd.qcut(test_df['rf_proba'], 10, labels=False, duplicates='drop')

print(f"\n  {'Decile':>7} {'N':>6} {'Avg Proba':>10} {'WR':>7} {'Avg Net':>10} {'Total Net':>12} {'Breakeven':>10}")
print("  " + "-"*70)
for d in sorted(test_df['decile'].unique()):
    grp = test_df[test_df['decile'] == d]
    n = len(grp)
    avg_p = grp['rf_proba'].mean()
    wr = grp['label'].mean()
    avg_net = grp['net_pnl'].mean()
    total_net = grp['net_pnl'].sum()
    avg_reward = grp['reward_dollars'].mean()
    avg_risk = grp['risk_dollars'].mean()
    be_wr = avg_risk / (avg_reward + avg_risk) if (avg_reward + avg_risk) > 0 else float('nan')
    print(f"  {d:>7} {n:>6} {avg_p:>10.3f} {wr*100:>6.1f}% {avg_net:>10.1f} {total_net:>12,.0f} {be_wr*100:>9.1f}%")

# ============================================================================
# TEST 4: CONDITIONAL EDGE ANALYSIS (without model — pure setup filters)
# ============================================================================
print("\n" + "="*80)
print("TEST 4: CONDITIONAL EDGE (setup filters, no model)")
print("="*80)

base_wr = y_test.mean()
base_n = len(y_test)
base_ev = float(test_df['net_pnl'].mean())
print(f"\n  Base rate: WR={base_wr*100:.1f}% n={base_n} avg_net=${base_ev:.1f}")

conditions = {
    'dist > 2 ATR': test_df['vwap_width_atr'] > 2.0,
    'dist > 1.5 ATR': test_df['vwap_width_atr'] > 1.5,
    'dist > 1 ATR': test_df['vwap_width_atr'] > 1.0,
    'LONG only': test_df['is_long_setup'] == 1,
    'SHORT only': test_df['is_long_setup'] == 0,
    'First 30min': test_df['minutes_into_session'] <= 30,
    'First 1hr': test_df['minutes_into_session'] <= 60,
    'First 2hr': test_df['minutes_into_session'] <= 120,
    'Midday (2-4hr)': (test_df['minutes_into_session'] > 120) & (test_df['minutes_into_session'] <= 270),
    'Last 2hr': test_df['minutes_into_session'] > 270,
    'RSI extreme (<30 or >70)': test_df['rsi_extreme'] == 1,
    'High volume (>1.5x)': test_df['rel_vol'] > 1.5,
    'High volume (>2x)': test_df['rel_vol'] > 2.0,
    'VWAP crosses < 3 today': test_df['vwap_crosses_today'] < 3,
    'Consecutive same side > 5': test_df['consecutive_same_side'] > 5,
    'Consecutive same side > 10': test_df['consecutive_same_side'] > 10,
    'Reverting bar': test_df['bar_reverting'] == 1,
    'EMA20 trend aligned': (
        ((test_df['is_long_setup'] == 1) & (test_df['ema20_slope_atr'] > 0)) |
        ((test_df['is_long_setup'] == 0) & (test_df['ema20_slope_atr'] < 0))
    ),
    'EMA20 counter-trend': (
        ((test_df['is_long_setup'] == 1) & (test_df['ema20_slope_atr'] < -0.5)) |
        ((test_df['is_long_setup'] == 0) & (test_df['ema20_slope_atr'] > 0.5))
    ),
    'Day range < 1 ATR': test_df['day_range_atr'] < 1.0,
    'Day range > 3 ATR': test_df['day_range_atr'] > 3.0,
    'LONG + dist>1.5ATR': (test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > 1.5),
    'LONG + dist>1.5 + first2hr': (test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > 1.5) & (test_df['minutes_into_session'] <= 120),
    'LONG + dist>1ATR + RSI<30': (test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > 1.0) & (test_df['rsi'] < 30),
    'LONG + hi vol + dist>1ATR': (test_df['is_long_setup'] == 1) & (test_df['rel_vol'] > 1.5) & (test_df['vwap_width_atr'] > 1.0),
}

# Breakeven WR for stop 0.5 ATR varies per trade, compute per-group
print(f"\n  {'Condition':45s} {'N':>6} {'WR':>7} {'AvgNet':>9} {'TotNet':>12} {'vs Base':>8}")
print("  " + "-"*90)

results = []
for name, mask in conditions.items():
    grp = test_df[mask]
    n = len(grp)
    if n < 20:
        continue
    wr = grp['label'].mean()
    avg_net = grp['net_pnl'].mean()
    total_net = grp['net_pnl'].sum()
    vs_base = avg_net - base_ev
    results.append((name, n, wr, avg_net, total_net, vs_base))

# Sort by avg_net descending
results.sort(key=lambda x: x[3], reverse=True)
for name, n, wr, avg_net, total_net, vs_base in results:
    marker = " ***" if avg_net > 0 else ""
    print(f"  {name:45s} {n:>6} {wr*100:>6.1f}% {avg_net:>+9.1f} {total_net:>12,.0f} {vs_base:>+8.1f}{marker}")

# ============================================================================
# TEST 5: MAX 1 TRADE PER DAY (best signal only)
# ============================================================================
print("\n" + "="*80)
print("TEST 5: MAX 1 TRADE PER DAY (highest RF proba per day)")
print("="*80)

test_df['rank_in_day'] = test_df.groupby('date')['rf_proba'].rank(ascending=False, method='first')
best_per_day = test_df[test_df['rank_in_day'] == 1]

n_days = len(best_per_day)
wr_best = best_per_day['label'].mean()
avg_net_best = best_per_day['net_pnl'].mean()
total_net_best = best_per_day['net_pnl'].sum()
print(f"  1 trade/day (best proba): n={n_days} WR={wr_best*100:.1f}% avg_net=${avg_net_best:.1f} total=${total_net_best:,.0f}")

# Best per day, LONG only
best_long = test_df[(test_df['is_long_setup'] == 1)].copy()
if len(best_long) > 0:
    best_long['rank_in_day'] = best_long.groupby('date')['rf_proba'].rank(ascending=False, method='first')
    best_long_day = best_long[best_long['rank_in_day'] == 1]
    n_bld = len(best_long_day)
    wr_bld = best_long_day['label'].mean()
    avg_bld = best_long_day['net_pnl'].mean()
    total_bld = best_long_day['net_pnl'].sum()
    print(f"  1 LONG/day (best proba):  n={n_bld} WR={wr_bld*100:.1f}% avg_net=${avg_bld:.1f} total=${total_bld:,.0f}")

# Best per day with min distance > 1 ATR
best_dist = test_df[test_df['vwap_width_atr'] > 1.0].copy()
if len(best_dist) > 0:
    best_dist['rank_in_day'] = best_dist.groupby('date')['rf_proba'].rank(ascending=False, method='first')
    best_dist_day = best_dist[best_dist['rank_in_day'] == 1]
    n_bd = len(best_dist_day)
    wr_bd = best_dist_day['label'].mean()
    avg_bd = best_dist_day['net_pnl'].mean()
    total_bd = best_dist_day['net_pnl'].sum()
    print(f"  1 trade/day (dist>1ATR):  n={n_bd} WR={wr_bd*100:.1f}% avg_net=${avg_bd:.1f} total=${total_bd:,.0f}")

# Best LONG per day with dist > 1 ATR
best_combo = test_df[(test_df['is_long_setup'] == 1) & (test_df['vwap_width_atr'] > 1.0)].copy()
if len(best_combo) > 0:
    best_combo['rank_in_day'] = best_combo.groupby('date')['rf_proba'].rank(ascending=False, method='first')
    best_combo_day = best_combo[best_combo['rank_in_day'] == 1]
    n_bc = len(best_combo_day)
    wr_bc = best_combo_day['label'].mean()
    avg_bc = best_combo_day['net_pnl'].mean()
    total_bc = best_combo_day['net_pnl'].sum()
    print(f"  1 LONG/day (dist>1ATR):   n={n_bc} WR={wr_bc*100:.1f}% avg_net=${avg_bc:.1f} total=${total_bc:,.0f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
The RF model's job is to find bars where VWAP reversion probability EXCEEDS 
the breakeven rate. At stop 0.5 ATR, breakeven varies by distance but is 
roughly 36%. The base rate is ~32%.

Key metrics:
  - AUC (base features): {auc1:.4f}
  - AUC (enhanced features): {auc2:.4f}
  - Top decile WR (base): {wr_top*100:.1f}%
  - Top decile WR (enhanced): {wr_top2*100:.1f}%
  - Separation (enhanced): {(wr_top2-wr_bot2)*100:.1f}%

If AUC > 0.55 and top-decile WR > breakeven, the model HAS edge — it's just
not being used correctly (taking too many trades dilutes it).
""")

print("DONE.")
