"""Detailed analysis of 2-year reversal strategy performance."""
import sys
sys.path.insert(0, r'C:\Users\Administrator\ibkr8\src')

import pandas as pd
import numpy as np
from indicators import calc_all_indicators, identify_reversal_setups, calc_theo_targets
from outcome_sim import simulate_all_setups

print("Loading 2-year data...")
df = pd.read_csv(r'C:\Users\Administrator\ibkr8\data\tsla_5min_2years.csv', parse_dates=['time'], index_col='time')
print(f"Loaded {len(df)} bars")

print("\nCalculating indicators...")
df_ind = calc_all_indicators(df)

# Test different RSI thresholds
print("\n" + "=" * 100)
print("RSI THRESHOLD SWEEP (Stop=1.5 ATR)")
print("=" * 100)
print(f"{'RSI_Long':<10} {'RSI_Short':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'AvgPnL':<10}")
print("-" * 100)

for max_rsi_long in [25, 30, 35, 40]:
    for min_rsi_short in [60, 65, 70, 75]:
        df_setups = identify_reversal_setups(df_ind, max_rsi_long=max_rsi_long, 
                                              min_rsi_short=min_rsi_short, min_rel_vol=1.0)
        df_targets = calc_theo_targets(df_setups, stop_atr_mult=1.5)
        trades = simulate_all_setups(df_targets)
        
        if len(trades) < 50:
            continue
        
        total = len(trades)
        wr = trades['is_winner'].mean() * 100
        win_sum = trades[trades['is_winner']]['pnl_per_share'].sum()
        loss_sum = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
        pf = win_sum / loss_sum if loss_sum > 0 else 0
        avg_pnl = trades['pnl_per_share'].mean()
        
        print(f"<{max_rsi_long:<9} >{min_rsi_short:<9} {total:<8} {wr:<8.1f} {pf:<8.2f} ${avg_pnl:<9.2f}")

# Test without BB requirement
print("\n" + "=" * 100)
print("WITHOUT BB REQUIREMENT (RSI only, Stop=1.5 ATR)")
print("=" * 100)

# Modify setup identification - RSI only
df_setups_rsi = df_ind.copy()
df_setups_rsi['long_setup'] = (df_ind['rsi'] < 30) & (df_ind['rel_vol'] >= 1.0)
df_setups_rsi['short_setup'] = (df_ind['rsi'] > 70) & (df_ind['rel_vol'] >= 1.0)

df_targets = calc_theo_targets(df_setups_rsi, stop_atr_mult=1.5)
trades = simulate_all_setups(df_targets)

print(f"RSI-only setups: {len(trades)} trades")
print(f"Win Rate: {trades['is_winner'].mean()*100:.1f}%")
win_sum = trades[trades['is_winner']]['pnl_per_share'].sum()
loss_sum = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
print(f"Profit Factor: {win_sum/loss_sum:.2f}")
print(f"Avg PnL: ${trades['pnl_per_share'].mean():.2f}")

# Test with VWAP distance filter
print("\n" + "=" * 100)
print("WITH VWAP DISTANCE FILTER (must be >1 ATR from VWAP)")
print("=" * 100)

df_setups_vwap = df_ind.copy()
df_setups_vwap['long_setup'] = (
    (df_ind['rsi'] < 35) & 
    (df_ind['rel_vol'] >= 1.0) & 
    (df_ind['close'] < df_ind['bb_lower']) &
    (df_ind['vwap_dist_atr'] < -1.0)  # Price at least 1 ATR below VWAP
)
df_setups_vwap['short_setup'] = (
    (df_ind['rsi'] > 65) & 
    (df_ind['rel_vol'] >= 1.0) & 
    (df_ind['close'] > df_ind['bb_upper']) &
    (df_ind['vwap_dist_atr'] > 1.0)  # Price at least 1 ATR above VWAP
)

df_targets = calc_theo_targets(df_setups_vwap, stop_atr_mult=1.5)
trades = simulate_all_setups(df_targets)

print(f"VWAP-filtered setups: {len(trades)} trades")
if len(trades) > 0:
    print(f"Win Rate: {trades['is_winner'].mean()*100:.1f}%")
    win_sum = trades[trades['is_winner']]['pnl_per_share'].sum()
    loss_sum = abs(trades[~trades['is_winner']]['pnl_per_share'].sum())
    print(f"Profit Factor: {win_sum/loss_sum:.2f}" if loss_sum > 0 else "N/A")
    print(f"Avg PnL: ${trades['pnl_per_share'].mean():.2f}")

# Test by year
print("\n" + "=" * 100)
print("PERFORMANCE BY YEAR (Standard Setup, Stop=1.5 ATR)")
print("=" * 100)

df_setups = identify_reversal_setups(df_ind, max_rsi_long=35.0, min_rsi_short=65.0, min_rel_vol=1.0)
df_targets = calc_theo_targets(df_setups, stop_atr_mult=1.5)
trades = simulate_all_setups(df_targets)
trades['year'] = pd.to_datetime(trades['entry_time']).dt.year

print(f"{'Year':<8} {'Trades':<8} {'Win%':<8} {'PF':<8} {'AvgPnL':<10}")
print("-" * 50)

for year in sorted(trades['year'].unique()):
    year_trades = trades[trades['year'] == year]
    wr = year_trades['is_winner'].mean() * 100
    win_sum = year_trades[year_trades['is_winner']]['pnl_per_share'].sum()
    loss_sum = abs(year_trades[~year_trades['is_winner']]['pnl_per_share'].sum())
    pf = win_sum / loss_sum if loss_sum > 0 else 0
    avg_pnl = year_trades['pnl_per_share'].mean()
    print(f"{year:<8} {len(year_trades):<8} {wr:<8.1f} {pf:<8.2f} ${avg_pnl:<9.2f}")
