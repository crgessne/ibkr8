import sys
import os
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')

import pandas as pd
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups

print("Loading 2-year data...")
df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
print(f"Loaded {len(df)} bars")
print(f"Date range: {df.index.min()} to {df.index.max()}")

print("\nCalculating indicators...")
df_ind = calc_all_indicators(df)
df_setups = identify_reversal_setups(df_ind, max_rsi_long=35.0, min_rsi_short=65.0, min_rel_vol=1.0)

long_setups = df_setups['long_setup'].sum()
short_setups = df_setups['short_setup'].sum()
print(f"Found {long_setups} LONG setups, {short_setups} SHORT setups")

print("\n" + "=" * 90)
print("STOP WIDTH PARAMETER SWEEP (2-Year Dataset)")
print("=" * 90)
print(f"{'Stop':<6} {'Trades':<8} {'Win%':<7} {'Long%':<7} {'Short%':<8} {'AvgPnL':<9} {'PF':<7} {'Tgt%':<7} {'Stp%':<7}")
print("-" * 90)

results = []
for stop_atr in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
    df_targets = calc_theo_targets(df_setups, stop_atr_mult=stop_atr, fallback_target_atr=1.5)
    trades = simulate_all_setups(df_targets)
    
    total = len(trades)
    winners = trades['is_winner'].sum()
    wr = winners / total * 100
    
    long_t = trades[trades['direction'] == 'LONG']
    short_t = trades[trades['direction'] == 'SHORT']
    long_wr = long_t['is_winner'].mean() * 100 if len(long_t) > 0 else 0
    short_wr = short_t['is_winner'].mean() * 100 if len(short_t) > 0 else 0
    
    avg_pnl = trades['pnl_per_share'].mean()
    
    win_sum = trades[trades['is_winner']]['pnl_per_share'].sum()
    loss_sum = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
    pf = win_sum / loss_sum if loss_sum > 0 else 999
    
    target_pct = (trades['outcome'] == 'TARGET').mean() * 100
    stop_pct = (trades['outcome'] == 'STOP').mean() * 100
    
    results.append({'stop': stop_atr, 'wr': wr, 'pf': pf, 'pnl': avg_pnl})
    print(f"{stop_atr:<6.2f} {total:<8} {wr:<7.1f} {long_wr:<7.1f} {short_wr:<8.1f} ${avg_pnl:<8.2f} {pf:<7.2f} {target_pct:<7.1f} {stop_pct:<7.1f}")

print("\n" + "=" * 90)
print("OPTIMAL PARAMETERS")
print("=" * 90)
best_wr = max(results, key=lambda x: x['wr'])
best_pf = max(results, key=lambda x: x['pf'] if x['pf'] < 900 else 0)
best_pnl = max(results, key=lambda x: x['pnl'])
print(f"Best Win Rate: {best_wr['wr']:.1f}% at stop={best_wr['stop']:.2f} ATR")
print(f"Best PF: {best_pf['pf']:.2f} at stop={best_pf['stop']:.2f} ATR")
print(f"Best Avg PnL: ${best_pnl['pnl']:.2f} at stop={best_pnl['stop']:.2f} ATR")
