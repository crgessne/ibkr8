import pandas as pd, numpy as np, sys, os
os.chdir(r'C:\Users\Administrator\ibkr8')
sys.stdout = open('_inspect_trades_out.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout

df = pd.read_csv('data/trades_realized_y2024_stop0.75_selprob_0.50_kclassifier_20260218_192628.csv')
print('shape:', df.shape)
print('cols:', list(df.columns))
print()
print(df[['entry_price','exit_price','exit_reason','gross_pnl','net_pnl','shares',
          'risk_dollars','vwap_dist_atr','per_trade_rr']].head(10).to_string())
print()
print('exit_reason value_counts:')
print(df['exit_reason'].value_counts())
print()
print('vwap_dist_atr describe:')
print(df['vwap_dist_atr'].describe())
print()
print('per_trade_rr describe:')
print(df['per_trade_rr'].describe())
print()
print('risk_dollars describe:')
print(df['risk_dollars'].describe())
print()
# stop_dist in dollars
df['stop_dist'] = df['risk_dollars'] / df['shares'].replace(0, np.nan)
print('stop_dist ($/share) describe:')
print(df['stop_dist'].describe().round(4))
print()
# per_trade_rr = net_pnl / risk_dollars  -- confirm
check = (df['net_pnl'] / df['risk_dollars']).describe()
print('net_pnl/risk_dollars describe (should match per_trade_rr):')
print(check.round(4))
print()
# target_dist = stop_dist * R:R (if per_trade_rr = realized R:R)
# But per_trade_rr from backtest = ?
# Let's see: notional/shares = entry_price
df['calc_entry'] = df['notional'] / df['shares'].replace(0, np.nan)
print('calc_entry vs entry_price (first 5):')
print(df[['entry_price','calc_entry']].head())
print()
# exit_price - entry_price for winners vs losers
df['price_move'] = df['exit_price'] - df['entry_price']
df['is_long'] = df['is_long'].astype(bool)
df['signed_move'] = np.where(df['is_long'], df['price_move'], -df['price_move'])
print('signed_move describe:')
print(df['signed_move'].describe().round(4))
print()
# vwap_dist_atr: distance to vwap in ATR units
# So target_dist_dollars = vwap_dist_atr * ATR
# ATR = stop_dist / 0.75
df['atr_est'] = df['stop_dist'] / 0.75
df['target_dist'] = df['vwap_dist_atr'] * df['atr_est']
print('target_dist ($/share) describe:')
print(df['target_dist'].describe().round(4))
print()
print('=== TARGET DIST PERCENTILES ===')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f'  p{p:2d}: ${df["target_dist"].quantile(p/100):.4f}')
print()
print('=== TRADES BELOW VARIOUS TARGET THRESHOLDS ===')
for thr in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    n = (df['target_dist'] < thr).sum()
    print(f'  target < ${thr:.2f}: {n:4d} / {len(df)} ({n/len(df)*100:.1f}%)')
print()
df['win'] = (df['exit_reason'] == 'target').astype(int)
print('win (exit_reason==target) rate:', df['win'].mean().round(4))
# try other exit reason names
print('exit_reason unique:', df['exit_reason'].unique())
print()
print('=== WIN RATE & NET P&L BY TARGET DISTANCE BUCKET ===')
bins   = [0, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 9999]
labels = ['<0.05','0.05-0.10','0.10-0.20','0.20-0.50','0.50-1.00','1.00-2.00','>2.00']
df['tgt_bucket'] = pd.cut(df['target_dist'], bins=bins, labels=labels)
grp = df.groupby('tgt_bucket', observed=True).agg(
    count=('win','count'),
    win_rate=('win','mean'),
    avg_rr=('per_trade_rr','mean'),
    total_net_pnl=('net_pnl','sum'),
    avg_net_pnl=('net_pnl','mean'),
).round(2)
print(grp.to_string())
print()
print('=== TOTAL SUMMARY ===')
print(f'  Total trades:    {len(df)}')
print(f'  Total net P&L:   ${df["net_pnl"].sum():,.0f}')
print(f'  Win rate:        {df["win"].mean()*100:.1f}%')
print(f'  Avg target dist: ${df["target_dist"].mean():.4f}')
print(f'  Avg stop dist:   ${df["stop_dist"].mean():.4f}')
print(f'  Avg vwap_dist_atr: {df["vwap_dist_atr"].mean():.4f}')
print(f'  Avg per_trade_rr:  {df["per_trade_rr"].mean():.4f}')
