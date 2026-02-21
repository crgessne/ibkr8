"""
Slippage / target-distance analysis on the 62-feature realized trades file.
Run:  .venv\Scripts\python.exe _check_slippage.py
"""
import pandas as pd, numpy as np, sys, os
os.chdir(r'C:\Users\Administrator\ibkr8')
sys.stdout = open('_check_slippage_out.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout

STOP_ATR  = 0.75
SLIP_LIVE  = 0.18   # $/share observed live (one-way)
SLIP_MODEL = 0.01   # $/share used in backtest (one-way)

df = pd.read_csv('data/trades_realized_y2024_stop0.75_selprob_0.50_kclassifier_20260218_192628.csv')
print('shape:', df.shape)
print('cols:', list(df.columns))
print()

# ── Derive distances ──────────────────────────────────────────────────────────
# stop_dist = risk_dollars / shares  (= 0.75 * ATR in $)
df['stop_dist']   = df['risk_dollars'] / df['shares'].replace(0, np.nan)
df['atr_est']     = df['stop_dist'] / STOP_ATR
# target_dist = vwap_dist_atr * ATR  (distance to VWAP in $)
df['target_dist'] = df['vwap_dist_atr'] * df['atr_est']
df['rr']          = df['target_dist'] / df['stop_dist']   # = vwap_dist_atr / 0.75

# win = hit VWAP target
df['win'] = (df['exit_reason'] == 'vwap').astype(int)

# ── Summaries ─────────────────────────────────────────────────────────────────
print("=== STOP DISTANCE (0.75 x ATR, $/share) ===")
print(df['stop_dist'].describe().round(4))

print("\n=== TARGET DISTANCE (vwap_dist x ATR, $/share) ===")
print(df['target_dist'].describe().round(4))

print("\n=== REWARD : RISK  (= target_dist / stop_dist) ===")
print(df['rr'].describe().round(4))

print("\n=== TARGET DIST PERCENTILES ===")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:2d}: ${df['target_dist'].quantile(p/100):.4f}")

print(f"\n=== % TRADES BELOW TARGET-DISTANCE THRESHOLDS ===")
for thr in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    n = (df['target_dist'] < thr).sum()
    print(f"  target < ${thr:.2f}: {n:4d} / {len(df)} ({n/len(df)*100:.1f}%)")

print(f"\n=== SLIPPAGE IMPACT ===")
print(f"  Backtest model: ${SLIP_MODEL:.2f}/share one-way  ->  ${SLIP_MODEL*2:.2f} round-trip")
print(f"  Live MKT obs:  ${SLIP_LIVE:.2f}/share one-way  ->  ${SLIP_LIVE*2:.2f} round-trip")
print(f"  Extra cost per share:  ${(SLIP_LIVE-SLIP_MODEL)*2:.2f} round-trip")
avg_shares = df['shares'].mean()
print(f"  Avg position size:     {avg_shares:.0f} shares")
print(f"  Extra cost per trade:  ${(SLIP_LIVE-SLIP_MODEL)*2*avg_shares:,.0f}")
print(f"  -> Need target_dist > ${SLIP_LIVE*2:.2f} to cover live slippage")

print(f"\n=== WIN RATE & NET P&L BY TARGET DISTANCE BUCKET ===")
bins   = [0, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 9999]
labels = ['<0.05','0.05-0.10','0.10-0.20','0.20-0.50','0.50-1.00','1.00-2.00','>2.00']
df['tgt_bucket'] = pd.cut(df['target_dist'], bins=bins, labels=labels)
grp = df.groupby('tgt_bucket', observed=True).agg(
    count     = ('win', 'count'),
    win_pct   = ('win', lambda x: round(x.mean()*100, 1)),
    avg_rr    = ('rr',  'mean'),
    total_pnl = ('net_pnl', 'sum'),
    avg_pnl   = ('net_pnl', 'mean'),
).round(2)
print(grp.to_string())

print(f"\n=== WIN RATE & NET P&L BY R:R BUCKET ===")
bins2   = [0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 9999]
labels2 = ['<0.25','0.25-0.50','0.50-0.75','0.75-1.00','1.00-1.50','1.50-2.00','>2.00']
df['rr_bucket'] = pd.cut(df['rr'], bins=bins2, labels=labels2)
grp2 = df.groupby('rr_bucket', observed=True).agg(
    count     = ('win', 'count'),
    win_pct   = ('win', lambda x: round(x.mean()*100, 1)),
    avg_tgt   = ('target_dist', 'mean'),
    total_pnl = ('net_pnl', 'sum'),
    avg_pnl   = ('net_pnl', 'mean'),
).round(2)
print(grp2.to_string())

print(f"\n=== CUMULATIVE P&L: FILTER BY MIN TARGET DISTANCE ===")
print(f"  {'min_tgt':>8}  {'trades':>7}  {'kept%':>6}  {'total_pnl':>14}  {'avg_pnl':>10}  {'wr%':>6}")
for min_t in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]:
    sub = df[df['target_dist'] >= min_t]
    if len(sub) == 0: continue
    print(f"  ${min_t:>6.2f}   {len(sub):>7}  {len(sub)/len(df)*100:>5.1f}%  "
          f"${sub['net_pnl'].sum():>13,.0f}  ${sub['net_pnl'].mean():>9,.0f}  "
          f"{sub['win'].mean()*100:>5.1f}%")

print(f"\n=== CUMULATIVE P&L: FILTER BY MIN R:R ===")
print(f"  {'min_rr':>8}  {'trades':>7}  {'kept%':>6}  {'total_pnl':>14}  {'avg_pnl':>10}  {'wr%':>6}")
for min_rr in [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]:
    sub = df[df['rr'] >= min_rr]
    if len(sub) == 0: continue
    print(f"  {min_rr:>8.2f}  {len(sub):>7}  {len(sub)/len(df)*100:>5.1f}%  "
          f"${sub['net_pnl'].sum():>13,.0f}  ${sub['net_pnl'].mean():>9,.0f}  "
          f"{sub['win'].mean()*100:>5.1f}%")

print(f"\n=== TOTAL SUMMARY ===")
print(f"  Total trades:       {len(df)}")
print(f"  Total net P&L:      ${df['net_pnl'].sum():>13,.0f}")
print(f"  Win rate (vwap):    {df['win'].mean()*100:.1f}%")
print(f"  Stop rate:          {(df['exit_reason']=='stop').mean()*100:.1f}%")
print(f"  EOD flatten rate:   {(df['exit_reason']=='eod').mean()*100:.1f}%")
print(f"  Avg target dist:    ${df['target_dist'].mean():.4f}")
print(f"  Median target dist: ${df['target_dist'].median():.4f}")
print(f"  Avg stop dist:      ${df['stop_dist'].mean():.4f}")
print(f"  Median stop dist:   ${df['stop_dist'].median():.4f}")
print(f"  Avg ATR est:        ${df['atr_est'].mean():.4f}")
print(f"  Avg vwap_dist_atr:  {df['vwap_dist_atr'].mean():.4f}")
print(f"  Avg realized R:R:   {df['per_trade_rr'].mean():.4f}")
