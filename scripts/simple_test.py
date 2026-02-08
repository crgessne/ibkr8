"""Simple parameter test."""
import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')
import pandas as pd
from indicators import calc_all_indicators, calc_theo_targets
from outcome_sim import simulate_all_setups

# Write to file directly
out = open(r'C:\Users\Administrator\ibkr8\data\results.txt', 'w')
out.write("Loading...\n")
out.flush()
df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
df_ind = calc_all_indicators(df)
out.write(f"Loaded {len(df)} bars\n")
out.flush()

# Test RSI < 25 for longs, RSI > 75 for shorts, with VWAP distance > 2 ATR
df_s = df_ind.copy()
df_s['long_setup'] = (
    (df_ind['rsi'] < 25) & 
    (df_ind['rel_vol'] >= 1.0) & 
    (df_ind['close'] < df_ind['bb_lower']) &
    (df_ind['vwap_dist_atr'] < -2.0)
)
df_s['short_setup'] = (
    (df_ind['rsi'] > 75) & 
    (df_ind['rel_vol'] >= 1.0) & 
    (df_ind['close'] > df_ind['bb_upper']) &
    (df_ind['vwap_dist_atr'] > 2.0)
)

n_long = df_s['long_setup'].sum()
n_short = df_s['short_setup'].sum()
out.write(f"Setups: {n_long} LONG, {n_short} SHORT\n")
out.flush()

df_t = calc_theo_targets(df_s, stop_atr_mult=2.0)
out.write("Running simulation...\n")
out.flush()
trades = simulate_all_setups(df_t)

out.write(f"Trades: {len(trades)}\n")
out.flush()
if len(trades) > 0:
    wr = trades['is_winner'].mean()*100
    ws = trades[trades['is_winner']]['pnl_per_share'].sum()
    ls = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
    pf = ws/ls if ls > 0 else 0
    out.write(f"Win Rate: {wr:.1f}%\n")
    out.write(f"Profit Factor: {pf:.2f}\n")
    out.write(f"Avg PnL: ${trades['pnl_per_share'].mean():.2f}\n")
out.write("Done\n")
out.close()
